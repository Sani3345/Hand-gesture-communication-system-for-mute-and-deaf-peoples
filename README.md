# Hand-gesture-communication-system-for-mute-and-deaf-peoples
# 🤟 Hand Gesture Communication System for Mute and Deaf People

A real-time **Hand Gesture Communication System** that uses computer vision and deep learning to recognize hand gestures through a webcam and convert them into meaningful words and sentences.

The project is designed to help people communicate more easily using hand gestures.

## ✨ Features

* 🎥 Real-time hand gesture recognition using a webcam
* 🤟 Hand landmark detection using **MediaPipe**
* 🧠 GRU-based sequence recognition for dynamic gestures
* 📊 **94.12% test accuracy** on the GRU sequence model
* 🔤 Supports 10 sequence gestures
* 💬 Converts recognized gestures into words and simple sentences
* ⚡ Real-time prediction and confidence display
* 🧹 Clear communication using the `C` key
* ❌ Exit the application using the `Q` key

## 🤟 Supported Gestures

The current GRU sequence model recognizes:

| No. | Gesture   |
| --- | --------- |
| 1   | COME      |
| 2   | FOOD      |
| 3   | HAPPY     |
| 4   | HELP      |
| 5   | I         |
| 6   | NO        |
| 7   | SORRY     |
| 8   | THANK YOU |
| 9   | WANT      |
| 10  | WATER     |

## 🧠 How It Works

The system follows these steps:

```text
Webcam
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks
   ↓
63 Features (x, y, z)
   ↓
30-Frame Sequence
   ↓
GRU Deep Learning Model
   ↓
Gesture Prediction
   ↓
Word / Sentence
```

Each detected hand contains **21 landmarks**.
Every landmark provides three coordinates:

```text
x, y, z
```

Therefore:

```text
21 × 3 = 63 features
```

The GRU model receives:

```text
30 frames × 63 features
```

and predicts one of the 10 supported gestures.

## 🛠️ Technologies Used

* **Python**
* **OpenCV**
* **MediaPipe**
* **NumPy**
* **TensorFlow / Keras**
* **GRU (Gated Recurrent Unit)**
* **Scikit-learn**
* **Git & GitHub**

## 📁 Project Structure

```text
Hand-gesture-communication-system-for-mute-and-deaf-peoples/
│
├── model/
│   ├── isl_gru_model.keras
│   ├── labels.txt
│   └── sign_model.pkl
│
├── dataset/
│   └── sign_data.csv
│
├── sequence_dataset/
│   ├── COME.csv
│   ├── FOOD.csv
│   ├── HAPPY.csv
│   ├── HELP.csv
│   ├── I.csv
│   ├── NO.csv
│   ├── SORRY.csv
│   ├── THANK YOU.csv
│   ├── WANT.csv
│   └── WATER.csv
│
├── collect_data.py
├── collect_sequence.py
├── train_model.py
├── train_sequence_model.py
├── train_gru.py
├── gesture_detector.py
├── live_detection.py
├── live_sequence_detection.py
├── final_isl_detector.py
├── Camera test.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sani3345/Hand-gesture-communication-system-for-mute-and-deaf-peoples.git
```

### 2. Open the project folder

```bash
cd Hand-gesture-communication-system-for-mute-and-deaf-peoples
```

### 3. Create a virtual environment

```bash
python3 -m venv venv
```

### 4. Activate the virtual environment

#### macOS / Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Sequence Gesture Detector

After installing the dependencies and ensuring the trained GRU model is available:

```bash
python live_sequence_detection.py
```

The webcam will open and start detecting hand gestures.

### Controls

| Key | Action              |
| --- | ------------------- |
| `C` | Clear communication |
| `Q` | Quit application    |

## 📊 Model Performance

The GRU sequence model was evaluated on a test dataset and achieved:

**Test Accuracy: 94.12%**

Model input:

```text
(30, 63)
```

Model output:

```text
10 classes
```

## 💬 Example

Recognized gestures can be combined into simple communication such as:

```text
I → WANT → WATER

Output:
I want water.
```

Other examples include:

```text
I → WANT → FOOD
```

```text
HELP
```

```text
THANK YOU
```

## 🔮 Future Improvements

* Add more ISL gestures and vocabulary
* Improve recognition under different lighting conditions
* Support two-hand gestures
* Add text-to-speech output
* Add a graphical user interface
* Improve sentence formation using NLP
* Collect a larger and more diverse dataset
* Deploy the system as a desktop or web application

## 🎯 Project Goal

The main goal of this project is to demonstrate how **computer vision and deep learning** can be used to create an accessible communication system based on hand gestures.

## 👩‍💻 Author

**Saniya Kumari**

---

⭐ If you find this project useful, consider giving the repository a star!

