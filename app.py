import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from mtcnn import MTCNN
from PIL import Image

# Load emotion model
MODEL_PATH = "models/emotion_model.h5"
model = load_model(MODEL_PATH)

# FER2013 classes
classes = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Initialize MTCNN detector
detector = MTCNN()

st.title("Face Emotion Detection")
st.write("Upload an image and see the detected emotion!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file:
    # Convert to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MTCNN expects RGB

    # Detect faces
    results = detector.detect_faces(img_rgb)

    # Apply filtering: high confidence + reasonable aspect ratio + min size
    filtered_faces = []
    for res in results:
        confidence = res['confidence']
        x, y, w, h = res['box']

        if confidence < 0.95:  # ignore weak detections
            continue
        if w < 50 or h < 50:  # ignore tiny detections (like watches)
            continue
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:  # ignore weird shapes
            continue

        filtered_faces.append(res)

    if len(filtered_faces) == 0:
        st.warning("No valid face detected. Please upload a clearer image.")
    else:
        st.subheader("Detected Faces (Grid View)")

        cols = st.columns(3)  # display cropped faces in grid
        col_idx = 0

        for res in filtered_faces:
            x, y, w, h = res['box']
            x, y = max(0, x), max(0, y)

            # Crop face
            face = img_rgb[y:y+h, x:x+w]

            # Preprocess for model
            face_resized = cv2.resize(face, (48,48))
            gray_face = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
            input_img = gray_face.astype('float32') / 255.0
            input_img = np.expand_dims(input_img, axis=0)   # batch dimension
            input_img = np.expand_dims(input_img, axis=-1)  # channel dimension

            # Predict emotion
            predictions = model.predict(input_img, verbose=0)
            emotion_idx = np.argmax(predictions)
            emotion_label = classes[emotion_idx]

            # Draw bounding box & label (without %)
            cv2.rectangle(img_rgb, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img_rgb, f"{emotion_label}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

            # Show in grid (without %)
            with cols[col_idx]:
                st.image(face, caption=f"{emotion_label}", use_container_width=True)
            col_idx = (col_idx + 1) % 3

        # Show final image with all detections
        st.subheader("Final Image with Bounding Boxes")
        st.image(img_rgb, caption="Detected Faces", use_container_width=True)
