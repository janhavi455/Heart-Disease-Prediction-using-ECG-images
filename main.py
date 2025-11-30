# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image
import os

app = FastAPI(
    title="ECG Heart Disease Prediction API",
    description="Classify ECG images as Normal or Abnormal",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = "ecg_heart_disease_model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except:
    print("Warning: Model not found. Please train the model first.")
    model = None

# Class names for 2-class classification
CLASS_NAMES = ["normal", "abnormal"]

def preprocess_image(image_bytes):
    """Preprocess uploaded ECG image"""
    # Convert bytes to numpy array
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Resize and preprocess
    image_np = cv2.resize(image_np, (224, 224))
    image_np = image_np / 255.0
    image_np = np.expand_dims(image_np, axis=0)
    
    return image_np

@app.post("/predict")
async def predict_heart_disease(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Determine if heart disease is present
        has_heart_disease = predicted_class == "abnormal"
        
        return JSONResponse({
            "prediction": predicted_class,
            "has_heart_disease": has_heart_disease,
            "is_normal": not has_heart_disease,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "message": "Patient is normal" if not has_heart_disease else "Patient has heart disease"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {"message": "ECG Heart Disease Prediction API - 2 Class Classification"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Serve the frontend
@app.get("/frontend")
async def serve_frontend():
    return FileResponse('frontend/index.html')

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)