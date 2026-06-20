# 🤟 Real-Time Sign Language Translator

A real-time Sign Language Translation system that detects hand gestures using a webcam and converts them into meaningful text. The project uses **MediaPipe** for hand landmark detection, **TensorFlow/Keras** for gesture classification, and **Streamlit** for an interactive user interface.

---

## 📌 Project Overview

Communication between sign language users and non-sign language users can be challenging. This project aims to bridge that gap by recognizing hand signs in real time and translating them into readable text.

The system captures hand gestures through a webcam, extracts hand landmarks, predicts the corresponding sign using a trained deep learning model, and displays the translated output instantly.

---

## 🚀 Features

* Real-time hand gesture recognition
* Live webcam feed
* Hand landmark detection using MediaPipe
* Deep learning-based sign classification
* Sentence generation from detected signs
* Interactive Streamlit interface
* Custom dataset support
* High prediction accuracy (~99%)

---

## 🛠️ Technologies Used

| Technology         | Purpose                          |
| ------------------ | -------------------------------- |
| Python             | Programming Language             |
| OpenCV             | Video Capture & Image Processing |
| MediaPipe          | Hand Landmark Detection          |
| TensorFlow / Keras | Deep Learning Model              |
| Scikit-Learn       | Data Processing                  |
| Pandas & NumPy     | Data Handling                    |
| Streamlit          | Web Application                  |

---

## 📂 Project Structure

```text
RealTimeSignTranslator/
│
├── dataset/
│   └── dataset.csv
│
├── models/
│   ├── sign_model.keras
│   └── label_encoder.pkl
│
├── collect_data.py
├── train.py
├── app.py
├── streamlit_app.py
├── requirements.txt
├── .env
└── README.md
```

---

## 🔄 Project Workflow

### 1. Data Collection

* Webcam captures hand gestures.
* MediaPipe extracts 21 hand landmarks.
* Each landmark contains:

  * X coordinate
  * Y coordinate
  * Z coordinate

Total Features:

```text
21 Landmarks × 3 Coordinates = 63 Features
```

* Features are stored in `dataset.csv`.

---

### 2. Model Training

The collected dataset is used to train a Neural Network model.

Input:

```text
63 Hand Landmark Features
```

Output:

```text
Predicted Sign Class
```

Model:

* TensorFlow Sequential Neural Network
* Dense Layers
* Softmax Output Layer

---

### 3. Real-Time Prediction

The system:

1. Captures webcam frames
2. Detects hand landmarks
3. Extracts 63 features
4. Sends features to the trained model
5. Predicts the corresponding sign
6. Displays the translated word

---

## 🧠 Architecture

```text
Webcam
   │
   ▼
OpenCV
   │
   ▼
MediaPipe
(21 Hand Landmarks)
   │
   ▼
Feature Extraction
(63 Features)
   │
   ▼
TensorFlow Model
   │
   ▼
Predicted Sign
   │
   ▼
Sentence Generation
   │
   ▼
Streamlit Interface
```

---

## 📊 Dataset Information

Dataset Type:

```text
Custom Dataset
```

Features:

```text
63 Features
```

Samples:

```text
1899 Samples
```

Format:

```csv
x1,y1,z1,x2,y2,z2,...,x21,y21,z21,label
```

---

## 🎯 Model Performance

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | ~99%   |
| Validation Accuracy | ~99%   |
| Final Accuracy      | 99.47% |

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/mohamedriayzkhan/RealTimeSignTranslator.git
cd RealTimeSignTranslator
```

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

---

## 📸 Collect Training Data

```bash
python collect_data.py
```

Enter the sign label:

```text
HELLO
```

The script collects hand landmark samples and stores them in:

```text
dataset/dataset.csv
```

---

## 🏋️ Train Model

```bash
python train.py
```

Generated Files:

```text
models/sign_model.keras
models/label_encoder.pkl
```

---

## 🎥 Run Desktop Application

```bash
python app.py
```

---

## 🌐 Run Streamlit Application

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## 📷 Demo

### Input

Hand Gesture:

```text
HELLO
```

### Output

```text
HELLO
```

Sentence Formation:

```text
HELLO HOW ARE YOU
```

---

## ⚠️ Challenges Faced

* Hand detection under different lighting conditions
* Dataset collection consistency
* Real-time prediction stability
* Streamlit deployment compatibility
* Avoiding duplicate predictions

---

## 🔮 Future Enhancements

* Full Indian Sign Language (ISL) support
* Text-to-Speech conversion
* AI-powered sentence correction
* Mobile application
* Multi-hand gesture recognition
* Cloud deployment

---

## 👨‍💻 Developer

**Mohamed Riayz Khan**

B.Sc Information Technology
Rathinam College of Arts and Science

GitHub:

[mohamedriayzkhan](https://github.com/mohamedriayzkhan?utm_source=chatgpt.com)

---

## 📜 License

This project is developed for educational and research purposes.

---

⭐ If you found this project useful, consider giving it a star on GitHub! ⭐
