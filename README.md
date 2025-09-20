# 🎭 Face Emotion Detection with Streamlit

This project is a **Streamlit web application** that detects **faces** in an uploaded image and predicts the **emotion** of each detected face using a pre-trained deep learning model (trained on FER2013 dataset).  

It uses:
- **[MTCNN](https://github.com/ipazc/mtcnn)** for accurate face detection  
- **TensorFlow/Keras** for loading and running the emotion recognition model  
- **OpenCV** for image preprocessing and drawing bounding boxes  
- **Streamlit** for the interactive web app  

---

## 🚀 Features
- Upload an image (`jpg`, `jpeg`, or `png`)  
- Detects all faces in the image  
- Crops and preprocesses each face  
- Predicts emotions from **7 categories**:  
  - Angry  
  - Disgust  
  - Fear  
  - Happy  
  - Sad  
  - Surprise  
  - Neutral  
- Displays cropped faces in a **grid view** with predicted labels  
- Displays the **final image with bounding boxes and emotion labels**  

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/face-emotion-detection.git
cd face-emotion-detection
2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

3. Install dependencies
pip install -r requirements.txt


If you don’t have a requirements.txt, install manually:

pip install streamlit tensorflow opencv-python mtcnn pillow numpy

📂 Project Structure
face-emotion-detection/
│── models/
│   └── emotion_model.h5       # Pre-trained FER2013 emotion recognition model
│── app.py                     # Streamlit app (main code)
│── README.md                  # Project documentation
│── requirements.txt           # Python dependencies

▶️ Usage

Run the Streamlit app:

streamlit run app.py


Then open the provided local URL in your browser (e.g. http://localhost:8501).

📊 Example Workflow

Upload a photo with one or more faces

The app detects the faces using MTCNN

Each face is cropped and resized to 48x48 grayscale

The pre-trained CNN model predicts the emotion

The app shows:

Cropped faces in a grid view with emotion labels

The final annotated image with bounding boxes + labels

📦 Requirements

Python 3.8+

TensorFlow (tested on 2.x)

Streamlit

OpenCV

MTCNN

Pillow

NumPy

📝 Notes

The model was trained on FER2013, so results may vary on real-world images.

False positives may still occur; filtering is applied to reduce them.

You can replace models/emotion_model.h5 with your own trained model if desired.
