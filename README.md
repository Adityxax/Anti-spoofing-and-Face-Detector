# 🛡️ Anti-Spoofing & Face Detection System

A real-time **Face Detection + Anti-Spoofing** system built using computer vision and deep learning. This project detects human faces from a live camera feed or images and determines whether the face is **real or spoofed** (e.g., photo, video replay, screen attack).

Designed for use cases like:

* Secure authentication systems
* Attendance systems
* Access control
* Identity verification pipelines

---

## 🚀 Features

* **Real-time face detection** using YOLOv8
* **Anti-spoofing detection** (real vs fake face)
* Works with **webcam and image input**
* High-performance inference using **PyTorch**
* Face landmark and recognition support
* Modular and easy-to-extend codebase

---

## 🧠 Tech Stack

| Technology            | Usage                                    |
| --------------------- | ---------------------------------------- |
| OpenCV (cv2)          | Image processing & video stream handling |
| NumPy                 | Numerical computations                   |
| Dlib                  | Face landmarks & detection support       |
| Ultralytics (YOLOv8n) | Face detection model                     |
| PyTorch               | Deep learning inference                  |
| CvZone                | UI overlays & bounding box utilities     |
| Face Recognition      | Face encoding & matching                 |
| Pandas                | Data handling & logging                  |
| CMake                 | Dependency build support                 |

---

## 📁 Project Structure

```text
Anti-spoofing-and-Face-Detector/
├── Datasets/
│   ├── DataCollect/        # Raw captured data
│   └── SplitData/          # Organized datasets (train/val/test)
├── models/                 # Pre-trained and fine-tuned models
├── runs/                   # Training logs and weights
├── DataCollection.py
├── SplitData.py
├── Train.py
└── main.py
```

---

## 🛠️ Components

| Script | Purpose |
| :--- | :--- |
| `DataCollection.py` | Captures images and generates normalized YOLO labels from a webcam feed. |
| `SplitData.py` | Automatically splits the collected data into Train, Validation, and Test sets. |
| `Train.py` | Handles the model training process using Ultralytics YOLO. |
| `main.py` | Real-time inference script for live detection and anti-spoofing testing. |

---

## 🏁 Quick Start

1.  **Collect Data**: Run `DataCollection.py` to capture "real" vs "fake" faces.
2.  **Organize**: Run `SplitData.py` to generate the YOLO-ready directory structure.
3.  **Train**: Run `Train.py` to build your custom model.
4.  **Inference**: Run `main.py` to start detection.

---

## 🧪 How It Works

1. **YOLOv8/11** detects faces in each frame
2. Detected face is passed to **Anti-Spoof Model**
3. Facial landmarks and texture analysis are extracted
4. Deep learning model classifies the face as **Real or Fake**
5. Results are displayed using **CvZone overlays**

---

## 📊 Output Example

```
[INFO] Face Detected
[INFO] Liveness: REAL
[INFO] Confidence: 96.3%
```

---

## 🔐 Use Cases

* Biometric Authentication Systems
* Online Exam Proctoring
* Smart Door Locks
* Attendance Systems
* Security Surveillance

---

## 🛠️ Future Improvements

* Add Mobile Camera Support
* Improve Spoof Detection Accuracy
* Add API Support (FastAPI / Flask)
* Cloud Deployment (Docker + AWS)
* Face Database Integration

---

## 🤝 Contributing

Pull requests are welcome. If you have major changes, please open an issue first to discuss what you'd like to improve.

---

## 👤 Author

**Adi**
Computer Vision & AI Enthusiast

---

## ⭐ Star This Repo

If this project helped you, consider giving it a star. It makes my day and tells the algorithm I’m not completely useless.

---
*Created with ❤️ for premium computer vision development.*
