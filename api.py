from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load model
model = tf.keras.models.load_model("models/best_model.keras")

# Class names
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# FastAPI app
app = FastAPI(title="Smart Waste Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def home():
    return {"message": "Smart Waste Classifier API is running ✅"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100
    return {
        "predicted_class": predicted_class,
        "confidence": f"{confidence:.2f}%",
        "all_probabilities": {
            CLASS_NAMES[i]: f"{predictions[0][i]*100:.2f}%"
            for i in range(len(CLASS_NAMES))
        }
    }