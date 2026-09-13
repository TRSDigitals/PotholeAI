import os
import uuid
import traceback
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"  # Easy to change if your model is elsewhere.
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1 GB
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_model = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model():
    global _model
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Trained pothole model not found. Please place your trained best.pt file inside models/."
        )
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def enhance_night(frame: np.ndarray) -> np.ndarray:
    """Mild low-light enhancement; it does not change or retrain the AI model."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    return enhanced


def severity_from_area(box_area: float, frame_area: float) -> str:
    ratio = box_area / max(frame_area, 1.0)
    # Explainable visual estimate, not physical pothole depth/damage measurement.
    if ratio < 0.015:
        return "LOW"
    if ratio < 0.05:
        return "MEDIUM"
    return "HIGH"


def draw_safe_visualization(frame: np.ndarray) -> None:
    """Perspective-style visualization only; this is NOT road segmentation."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    pts = np.array(
        [
            [int(w * 0.38), int(h * 0.48)],
            [int(w * 0.62), int(h * 0.48)],
            [int(w * 0.94), int(h * 0.98)],
            [int(w * 0.06), int(h * 0.98)],
        ],
        dtype=np.int32,
    )
    cv2.polylines(overlay, [pts], True, (0, 220, 0), 4)
    cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)

    cv2.rectangle(frame, (20, 20), (255, 75), (0, 120, 0), -1)
    cv2.putText(frame, "ROAD SAFE", (38, 57), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)


def draw_detection_header(frame: np.ndarray, count: int) -> None:
    h, w = frame.shape[:2]
    box_w = min(w - 40, 420)
    cv2.rectangle(frame, (20, 20), (20 + box_w, 78), (20, 20, 160), -1)
    cv2.putText(
        frame,
        f"POTHOLE DETECTED | COUNT: {count}",
        (35, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def create_writer(output_path: Path, fps: float, width: int, height: int):
    # H.264 is preferred for browser compatibility. Some OpenCV builds do not provide it.
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError("Could not create an MP4 video writer. Install an OpenCV build with video codec support.")


def process_video(input_path: Path, output_path: Path, night_mode: bool = True):
    model = get_model()
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("The uploaded video could not be opened. Please try a valid road video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("The uploaded video has invalid frame dimensions.")

    writer, codec = create_writer(output_path, fps, width, height)
    total_detections = 0
    max_in_frame = 0
    frames_with_potholes = 0
    processed_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if night_mode:
                frame_for_ai = enhance_night(frame)
            else:
                frame_for_ai = frame

            results = model.predict(
                source=frame_for_ai,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
            )
            result = results[0]
            boxes = result.boxes
            frame_count = 0
            frame_area = float(width * height)

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy.tolist()
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(x1 + 1, min(x2, width))
                    y2 = max(y1 + 1, min(y2, height))

                    conf = float(box.conf[0].cpu().item())
                    cls_id = int(box.cls[0].cpu().item()) if box.cls is not None else 0
                    class_name = result.names.get(cls_id, "Pothole") if hasattr(result, "names") else "Pothole"
                    area = float((x2 - x1) * (y2 - y1))
                    severity = severity_from_area(area, frame_area)

                    # Red pothole bounding box.
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"{class_name} | {conf * 100:.1f}% | {severity}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    label_y = max(30, y1)
                    cv2.rectangle(frame, (x1, label_y - th - 12), (x1 + tw + 10, label_y), (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, label_y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    frame_count += 1

            if frame_count == 0:
                draw_safe_visualization(frame)
            else:
                frames_with_potholes += 1
                draw_detection_header(frame, frame_count)
                total_detections += frame_count
                max_in_frame = max(max_in_frame, frame_count)

            # Frame counter is useful for a project demonstration and does not affect detection.
            cv2.putText(
                frame,
                f"Frame: {processed_frames + 1}/{total_frames if total_frames > 0 else '?'}",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            processed_frames += 1
    finally:
        cap.release()
        writer.release()

    if processed_frames == 0:
        raise ValueError("No readable frames were found in the uploaded video.")

    return {
        "total_detections": total_detections,
        "max_potholes_in_frame": max_in_frame,
        "frames_processed": processed_frames,
        "frames_with_potholes": frames_with_potholes,
        "fps": round(float(fps), 2),
        "width": width,
        "height": height,
        "codec": codec,
    }


@app.route("/")
def index():
    model_ready = MODEL_PATH.is_file()
    return render_template("index.html", model_ready=model_ready)


@app.route("/analyze", methods=["POST"])
def analyze():
    if not MODEL_PATH.is_file():
        return jsonify({"success": False, "error": "Trained pothole model not found. Please place your trained best.pt file inside models/."}), 400

    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video selected."}), 400

    video = request.files["video"]
    if not video or not video.filename:
        return jsonify({"success": False, "error": "Empty filename. Please select a video."}), 400
    if not allowed_file(video.filename):
        return jsonify({"success": False, "error": "Unsupported video format. Use MP4, AVI, MOV, MKV, WEBM or M4V."}), 400

    job_id = uuid.uuid4().hex
    safe_name = secure_filename(video.filename)
    input_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    output_path = OUTPUT_DIR / f"processed_{job_id}.mp4"
    video.save(str(input_path))

    night_mode = request.form.get("night_mode", "on") == "on"
    try:
        stats = process_video(input_path, output_path, night_mode=night_mode)
        return jsonify({
            "success": True,
            "message": "Video analysis completed successfully.",
            "video_url": f"/outputs/{output_path.name}",
            "download_url": f"/download/{output_path.name}",
            "stats": stats,
        })
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except (ValueError, RuntimeError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception:
        app.logger.error("Unexpected processing error:\n%s", traceback.format_exc())
        return jsonify({"success": False, "error": "Processing failed. Check that the trained model and video are valid, then try again."}), 500
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/outputs/<path:filename>")
def output_video(filename):
    return send_from_directory(OUTPUT_DIR, filename, mimetype="video/mp4", as_attachment=False, max_age=0)


@app.route("/download/<path:filename>")
def download_video(filename):
    return send_from_directory(OUTPUT_DIR, filename, mimetype="video/mp4", as_attachment=True, download_name=filename, max_age=0)


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"success": False, "error": "Video is too large. Maximum upload size is 1 GB."}), 413


if __name__ == "__main__":
    print("AI-Based Pothole Detection & Road Safety Visualization System")
    print(f"Model path: {MODEL_PATH}")
    if MODEL_PATH.is_file():
        print("Model status: trained best.pt found")
    else:
        print("Model status: MISSING — place your existing trained best.pt in models/")
    app.run(host="127.0.0.1", port=5000, debug=False)
