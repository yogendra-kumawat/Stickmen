# 🕺 Real-Time Stickman Generator

[![Platform](https://img.shields.io/badge/Platform-PC%20%2F%20Webcam-blue.svg)]()
[![Language](https://img.shields.io/badge/Language-Python%203.9-green.svg)]()
[![Libraries](https://img.shields.io/badge/Libraries-MediaPipe%20%7C%20OpenCV-red.svg)]()
[![Mode](https://img.shields.io/badge/Mode-Real--Time%20Video-orange.svg)]()

A real-time computer vision project that detects anyone standing in front of a webcam and renders them as a **live animated stickman** on a black canvas — using Google's MediaPipe Pose Landmarker to track body joints frame by frame.

---

## 🧭 Table of Contents

- [Project Overview](#-project-overview)
- [How It Works](#-how-it-works)
- [Pose Landmarks Used](#-pose-landmarks-used)
- [Code Walkthrough](#-code-walkthrough)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Future Improvements](#-future-improvements)

---

## 📖 Project Overview

The program captures live webcam footage, runs **MediaPipe Pose Landmarker** on every frame, and maps detected body keypoints to a simplified stickman drawn with OpenCV lines and a circle (for the head). The real person's video feed is hidden — only the stickman is shown on a pure black background, giving a clean artistic effect.

**Key Features:**
- 🎥 Real-time webcam capture and pose detection
- 🖤 Stickman rendered on a **pure black canvas** (background removed)
- 👥 Supports detection of **up to 6 people** simultaneously
- 🪞 **Mirror-flipped** so movement feels natural (like a mirror)
- ⚡ Continuous loop with graceful error handling for dropped frames

---

## ⚙️ How It Works

```
Webcam Frame
     │
     ▼
cv2.flip()  ──────────────────────► Mirror image (feels natural)
     │
     ▼
mp.Image()  ──────────────────────► Convert to MediaPipe format
     │
     ▼
PoseLandmarker.detect()  ─────────► Extract 33 body landmarks (x, y, z)
     │
     ▼
pointer()   ──────────────────────► Draw lines + circle on black canvas
     │
     ▼
cv2.imshow()  ────────────────────► Display live stickman window
```

---

# video playback 

<img width="400" height="296" alt="Screencast from 2026-02-06 13-07-23" src="https://github.com/user-attachments/assets/f68d835d-ebc8-484e-baf5-32fdbd80f9bd" />

---

## 🦴 Pose Landmarks Used

MediaPipe provides 33 landmarks. This project uses the following to construct the stickman:

| Landmark Index | Body Part         | Role in Stickman       |
|----------------|-------------------|------------------------|
| `0`            | Nose              | Head center            |
| `7` / `8`      | Left / Right ear  | Head radius reference  |
| `11` / `12`    | Left / Right shoulder | Shoulder midpoint  |
| `13` / `14`    | Left / Right elbow    | Arm joints         |
| `15` / `16`    | Left / Right wrist    | Arm endpoints      |
| `23` / `24`    | Left / Right hip      | Hip midpoint       |
| `25` / `26`    | Left / Right knee     | Leg joints         |
| `27` / `28`    | Left / Right ankle    | Leg endpoints      |

### Stickman Skeleton Map

```
        O          ← Head (circle, radius from ear distance)
        |
   ─────┼─────    ← Shoulders (midpoint calculated)
   |         |
   |         |    ← Upper arms
   |         |
   └─────────┘    ← Elbows → Wrists

        |
        |          ← Torso (shoulder mid → hip mid)
        |

   ─────┼─────    ← Hips (midpoint calculated)
   |         |
   |         |    ← Thighs
   |         |
   └─────────┘    ← Knees → Ankles
```

---

## 🔍 Code Walkthrough

### 1. Model Setup

```python
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=6,                        # Detect up to 6 people at once
    min_pose_detection_confidence=0.3   # Lower threshold = more sensitive detection
)
landmarker = PoseLandmarker.create_from_options(options)
```

### 2. The `avg()` Helper

Calculates the **midpoint x-coordinate** between two landmarks (used for shoulder and hip centers):

```python
def avg(landmarks, m, per, l, p):
    return int((int(landmarks[l].x * m) + int(landmarks[p].x * m)) / per)
```

| Parameter | Meaning                          |
|-----------|----------------------------------|
| `m`       | Image width (pixels)             |
| `per`     | Divisor (`2` for midpoint)       |
| `l`, `p`  | Landmark indices to average      |

### 3. The `pointer()` Drawing Function

Draws the stickman onto a **black canvas** (`back`) instead of the real image:

```python
def pointer(landmarks, image):
    color = (237, 16, 160)   # Hot pink / magenta stickman color
    edit = image             # Black canvas passed in
    n, m, _ = np.shape(edit) # n = height, m = width

    # Limb pairs: [from_landmark, to_landmark]
    lld = [[16,14],[14,12],[11,13],[13,15],   # Arms
           [28,26],[26,24],[23,25],[25,27]]    # Legs

    # Head radius = distance between ears
    rad = int((landmarks[7].x - landmarks[8].x) * m)

    for p, q in lld:
        # Special cases: connect limbs through body midpoints
        # Right arm upper → shoulder midpoint → spine
        # Left arm upper  → shoulder midpoint → spine
        # Legs            → hip midpoint
        ...
    
    # Draw head circle
    cv2.circle(edit, (head_x, head_y), rad, color, 30, 1)
```

### 4. Main Loop

```python
video = cv2.VideoCapture(0)   # Open default webcam
while True:
    try:
        _, image = video.read()
        image = cv2.flip(image, 1)                          # Mirror
        back = (np.ones(image.shape) * 0).astype(np.uint8) # Black canvas
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = landmarker.detect(mp_image)
        edit = pointer(result.pose_landmarks[0], back)      # Draw stickman
        cv2.imshow("marks", edit)
        if cv2.waitKey(1) & 0xFF == ord(" "):               # Space to quit
            break
    except:
        continue   # Skip bad/dropped frames silently
```

> **Press `SPACE`** to exit the live window.

---

## 🚀 Setup & Installation

### Requirements

- Python 3.9+
- A working webcam

### Install Dependencies

```bash
pip install mediapipe opencv-python numpy
```

### Download the Model

Download the MediaPipe Pose Landmarker model file and place it in the project root:

```bash
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task \
     -O pose_landmarker.task
```

Or download manually from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models) and rename the file to `pose_landmarker.task`.

### Project Structure

```
stickman/
├── main.ipynb            # Main notebook
├── pose_landmarker.task  # MediaPipe model file (download separately)
└── README.md
```

---

## ▶️ Usage

1. Ensure your webcam is connected and accessible.
2. Open and run `main.ipynb` in Jupyter:
   ```bash
   jupyter notebook main.ipynb
   ```
3. Run all cells — a window named **"marks"** will open showing the live stickman.
4. Stand in front of your camera and watch yourself become a stickman!
5. Press **`SPACE`** to close the window and stop the program.

### Tips for Best Results

| Tip | Why |
|-----|-----|
| Ensure good lighting | MediaPipe needs clear contrast to detect poses |
| Stand ~1–2 meters from camera | Full body should be visible in frame |
| Lower `min_pose_detection_confidence` to `0.2` for dim lighting | More sensitive detection |
| Raise it to `0.6`+ for less noise | Reduces false detections |

 
---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `mediapipe` | ≥ 0.10 | Pose landmark detection |
| `opencv-python` | ≥ 4.x | Webcam capture & drawing |
| `numpy` | ≥ 1.x | Image array manipulation |

---

*Built with ❤️ using MediaPipe and OpenCV.*
