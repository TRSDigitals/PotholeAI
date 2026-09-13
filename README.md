# AI-Based Pothole Detection & Road Safety Visualization System

A GitHub-ready final-year ECE project that analyzes road/vehicle video using an **already-trained YOLO pothole detection model**. The system produces a processed MP4 with red pothole bounding boxes, confidence percentages, approximate visual severity, pothole counts, and a green road-safe visualization when no pothole is detected in a frame.

> **Model rule:** This project does not retrain the model and does not include a fabricated model. Put your existing trained `best.pt` at `models/best.pt`.

## 1. Project overview

Road potholes are a common road-safety problem. Manual inspection is slow and difficult to scale. This project uses computer vision and an existing trained YOLO pothole model to analyze a road video frame-by-frame and visualize detected potholes.

The web application is built with Flask. OpenCV handles video input/output and frame processing, while Ultralytics YOLO performs inference using the student's trained `best.pt` model.

## 2. Problem statement

Traditional road inspection can require significant human effort and may miss potholes between inspection periods. An automated vision system can assist by identifying visible potholes in road videos and presenting the result in an easy-to-understand visual form.

## 3. Objectives

- Detect potholes in uploaded road videos using an already-trained YOLO model.
- Draw clear RED bounding boxes around detected potholes.
- Display detection confidence percentages.
- Estimate LOW / MEDIUM / HIGH visual severity from bounding-box area.
- Count detections and calculate the maximum potholes detected in one frame.
- Show a GREEN perspective-style road boundary and `ROAD SAFE` when no pothole is detected in a frame.
- Generate a processed MP4 that can be played in the browser and downloaded.
- Support optional browser-based continuous camera recording followed by analysis.
- Provide optional low-light enhancement for night/low-light video.

## 4. Features

- Professional responsive Flask web interface.
- Upload MP4, AVI, MOV, MKV, WEBM and M4V videos.
- Browser camera recording using `MediaRecorder` when supported.
- Existing trained YOLO `best.pt` model loading.
- Red pothole bounding boxes.
- Confidence percentage on each detection.
- LOW / MEDIUM / HIGH approximate visual severity.
- `POTHOLE DETECTED` frame status.
- `ROAD SAFE` frame status.
- Green perspective-style safe-road visualization.
- Processed MP4 browser player.
- Download button for processed MP4.
- Statistics for total detections, maximum detections in one frame, frames processed, and frames containing potholes.
- Clear model, video and processing error messages.
- No training pipeline is included because the trained model already exists.

## 5. Technologies used

- Python
- Flask
- Ultralytics YOLO
- OpenCV
- NumPy
- HTML5
- CSS3
- JavaScript / MediaRecorder API

## 6. System workflow

```text
Road video / camera recording
          |
          v
     Flask upload
          |
          v
 OpenCV reads frames
          |
          v
 Optional low-light enhancement
          |
          v
 Existing trained YOLO best.pt
          |
          v
 Pothole detections + confidence
          |
          v
 Red boxes + severity + frame status
          |
          +------ no detection ------> Green road-safe visualization
          |
          v
 OpenCV writes processed MP4
          |
          v
 Browser playback + download + statistics
```

## 7. Project structure

```text
AI-Pothole-Detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── best.pt                 # Place your real trained model here
│   └── MODEL_INSTRUCTIONS.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── uploads/
│   └── .gitkeep
└── outputs/
    └── .gitkeep
```

## 8. Installation

Python 3.10 or 3.11 is recommended for a smooth Ultralytics setup.

### Windows

```bash
cd AI-Pothole-Detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
cd AI-Pothole-Detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 9. Model placement instructions

Copy your **actual already-trained pothole YOLO model** to:

```text
models/best.pt
```

The application checks this file before loading the model. If it is missing, the UI shows:

> Trained pothole model not found. Please place your trained best.pt file inside models/.

You do **not** need to provide or retrain the approximately 2.6M+ training images again.

### Large model and GitHub

A large `.pt` file may be unsuitable for ordinary GitHub file limits. Keep the application source in GitHub and use Git LFS or a GitHub Release/model-storage solution for a large `best.pt` when appropriate. The application still expects the local file at `models/best.pt` when it runs.

## 10. Run the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 11. Upload a road video

1. Start the Flask application.
2. Open `http://127.0.0.1:5000`.
3. Click **Browse Video** and select a road/vehicle video.
4. Keep **Night / low-light enhancement** enabled for dark or low-light videos if useful.
5. Click **Analyze Road**.
6. Wait while OpenCV processes the video frame-by-frame.
7. Watch the processed video in the result section.
8. Click **Download Processed Video** to save the MP4.

## 12. Continuous camera recording

The web page also has **Record Road Video**. It uses the browser's camera permission and continuously records until **Stop Recording** is clicked. The recorded WebM file is then submitted to the same Flask/YOLO processing pipeline.

This is intentionally implemented as browser recording rather than pretending that a normal Flask page can continuously access a camera without browser permission.

For best results, mount the camera/phone securely and keep the road visible. Long recordings require more processing time and storage.

## 13. Expected output

When a pothole is detected:

- A **RED bounding box** is drawn around it.
- Confidence is displayed as a percentage.
- Approximate severity is displayed as LOW, MEDIUM or HIGH.
- The frame header shows `POTHOLE DETECTED` and the current frame count.

When no pothole is detected:

- A **GREEN perspective-style road boundary** is drawn.
- `ROAD SAFE` is displayed.

The output video is saved temporarily under `outputs/` and is served by Flask for browser playback/download.

## 14. Severity explanation

Severity is a simple, explainable **visual estimate** based on the detected bounding-box area divided by the frame area:

| Approx. box area / frame area | Severity |
|---:|---|
| `< 1.5%` | LOW |
| `1.5% – < 5%` | MEDIUM |
| `>= 5%` | HIGH |

This is **not** a certified civil-engineering road-damage measurement. Bounding-box area does not directly measure pothole depth, structural damage, road material condition, or repair priority.

## 15. Night / low-light operation

The application includes an optional CLAHE-based low-light enhancement before YOLO inference. This can improve visual contrast in some night videos, but it cannot guarantee night-time detection accuracy.

Actual night performance depends on the trained `best.pt` model, camera exposure, headlights, motion blur, weather, road surface, and training data. If the trained model was not trained for night conditions, enhancement cannot magically make it a night-trained model.

## 16. Limitations

- The AI result is limited by the accuracy and class definitions of the provided trained `best.pt` model.
- A detection model cannot guarantee that every real pothole will be detected.
- Confidence is model confidence, not a physical probability of road damage.
- Severity is only an approximate visual estimate.
- `ROAD SAFE` means no pothole was detected by the model in that frame; it does **not** mean the road is physically certified safe.
- The green boundary is a visualization, not semantic road segmentation.
- Very dark, blurred, rainy, occluded, or low-resolution footage may reduce detection quality.
- MP4 browser playback depends partly on the codecs available in the local OpenCV/FFmpeg build. The application prefers H.264 (`avc1`) and falls back to `mp4v` if necessary.
- Long videos consume more CPU/GPU time and disk space.
- The browser recording feature requires camera permission and a browser supporting `MediaRecorder`.

## 17. Future enhancements

- GPS coordinates for detected pothole locations.
- Road-condition heat maps.
- Mobile/edge deployment using an optimized YOLO model.
- Automatic report generation with timestamps and GPS.
- Tracking the same pothole across multiple frames to reduce duplicate counts.
- Real road segmentation and lane detection.
- Depth estimation for more meaningful damage assessment.
- Cloud dashboard for road-maintenance teams.
- Automatic night/weather classification.
- Model quantization and GPU acceleration for real-time edge processing.

## 18. Final-year project description

**AI-Based Pothole Detection & Road Safety Visualization System** is a computer-vision-based road monitoring application developed using Python, Flask, OpenCV, NumPy and Ultralytics YOLO. The system accepts a road or vehicle video and applies an already-trained pothole detection model to individual video frames. Detected potholes are highlighted using red bounding boxes, with confidence percentages and an explainable approximate severity category. When the model does not detect a pothole in a frame, the application displays a green perspective-style road visualization and `ROAD SAFE`. The processed video is generated as an MP4 and can be viewed or downloaded through the web interface. The project is intended as an academic decision-support and visualization system rather than a certified civil-engineering inspection instrument.

## 19. Verification status

If `best.pt` is not included in the repository/package, the **application code is prepared**, but **AI pothole detection itself is not verified** until your actual trained model is placed at `models/best.pt` and run on a suitable video.

No generic YOLO model is substituted for the pothole model.

## 20. Troubleshooting

### Model missing

Confirm:

```text
AI-Pothole-Detection/models/best.pt
```

### `ModuleNotFoundError`

Activate the virtual environment and run:

```bash
pip install -r requirements.txt
```

### Video cannot be opened

Try an MP4 recorded with a common H.264 video codec.

### Browser does not play the processed MP4

The application first requests `avc1` and then falls back to `mp4v`. If your local OpenCV installation lacks a browser-friendly codec, install an OpenCV/FFmpeg environment that provides H.264 encoding support.

### Night accuracy is poor

Keep night enhancement enabled, improve camera exposure/lighting, and verify that the trained model itself contains suitable night/low-light examples. Do not claim night accuracy merely because enhancement is enabled.
