from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
app = FastAPI(title="PlantVillage Model API", version="1.0.0")

MODEL_PATH = "model/best_tomato_efficientnetb0.keras"
LABELS_PATH = "app/labels.txt"
IMG_SIZE = 224

model = None
class_names = []
tf = None
preprocess_input = None
model_loaded = False

@app.on_event("startup")
def load_model():
    global model, class_names, tf, preprocess_input
    # Set environment flags before importing TensorFlow to avoid LLVM/oneDNN/XLA issues
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_cpu_global_jit=0')
    logging.getLogger('uvicorn').info('TensorFlow env flags set: TF_ENABLE_ONEDNN_OPTS=0')
    if tf is None:
        import tensorflow as _tf
        from tensorflow.keras.applications.efficientnet import preprocess_input as _preprocess_input
        tf = _tf
        preprocess_input = _preprocess_input

    if model is None:
        try:
            # Avoid loading optimizer state which can cause variable/compatibility issues
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception:
            import keras as _keras
            model = _keras.models.load_model(MODEL_PATH, compile=False)

        try:
            tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
        except Exception:
            pass

        global model_loaded
        model_loaded = True
        logging.getLogger('uvicorn').info('Model loaded successfully')
    
    if not class_names:
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
    return {"status": "ok", "model_loaded": model_loaded, "num_classes": len(class_names)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
