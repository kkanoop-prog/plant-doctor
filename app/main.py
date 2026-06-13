from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications.efficientnet import preprocess_input

app = FastAPI(title="PlantVillage Model API", version="1.0.0")

MODEL_PATH = "model/best_tomato_efficientnetb0.keras"
LABELS_PATH = "app/labels.txt"
IMG_SIZE = 224

# Load model once at startup
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array

@app.get("/")
def root():
    return {"message": "PlantVillage FastAPI service is running"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "num_classes": len(class_names)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_tensor = preprocess_image(contents)

        predictions = model.predict(input_tensor, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_label = class_names[predicted_index]
        confidence = float(predictions[predicted_index])

        top3_indices = np.argsort(predictions)[-3:][::-1]
        top3 = [
            {
                "label": class_names[int(i)],
                "confidence": float(predictions[int(i)])
            }
            for i in top3_indices
        ]

        return JSONResponse({
            "predicted_class": predicted_label,
            "confidence": confidence,
            "top_3_predictions": top3
        })

    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(ex)}")