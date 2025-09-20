import os
import subprocess
from train_model import train_model

MODEL_PATH = "models/emotion_model.h5"
DATA_DIR = "data/fer2013-images"

# Step 1: Train model if it does not exist
if not os.path.exists(MODEL_PATH):
    print("🚀 Training new model (first time only)...")
    train_model(
        base_dir=DATA_DIR,
        model_path=MODEL_PATH,
        epochs=100  # max epochs; early stopping will handle early exit
    )
else:
    print("✅ Model already exists. Skipping training.")

# Step 2: Launch Streamlit app
print("🌍 Launching Streamlit app...")
subprocess.run("streamlit run app.py", shell=True)
