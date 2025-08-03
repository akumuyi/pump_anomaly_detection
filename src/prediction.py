from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
from tempfile import NamedTemporaryFile
from typing import List, Optional
import numpy as np
import pandas as pd
from preprocessing import AudioPreprocessor
from model import PumpAnomalyDetector
from security import verify_api_key
from storage import ModelStorage
from logging_config import api_logger
import config

app = FastAPI(
    title="Pump Anomaly Detection API",
    description="API for detecting anomalies in pump sounds using machine learning",
    version="1.0.0"
)

# Use ModelStorage for model persistence
storage = ModelStorage()

# Initialize preprocessor and model
preprocessor = AudioPreprocessor()
model = PumpAnomalyDetector()

@app.post("/predict/")
async def predict(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    api_logger.info(f"Prediction request received: {file.filename}")
    """
    Predict if a pump sound is normal or abnormal.
    
    Args:
        file: Audio file in .wav format
    
    Returns:
        Prediction results including class and probability
    """
    if not file.filename or not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    
    try:
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Preprocess audio
        features = preprocessor.preprocess_audio(temp_path)
        if features is None:
            raise HTTPException(status_code=400, detail="Failed to extract features from audio")
        
        # Scale features
        features_scaled = preprocessor.transform(features)
        
        # Make prediction
        probabilities = model.predict_proba(features_scaled)
        prediction = model.predict(features_scaled)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # For normal predictions (0), use the probability of class 0
        # For abnormal predictions (1), use the probability of class 1
        pred_class = int(prediction[0])
        confidence = float(probabilities[0][pred_class])
        
        return JSONResponse({
            "prediction": "abnormal" if pred_class == 1 else "normal",
            "probability": confidence,
            "confidence": confidence
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/")
async def retrain(files: List[UploadFile] = File(...), labels: Optional[List[int]] = None, api_key: str = Depends(verify_api_key)):
    api_logger.info(f"Retrain request received: {len(files)} files")
    """
    Retrain the model with new data.
    
    Args:
        files: List of audio files (.wav)
        labels: List of corresponding labels (0 for normal, 1 for abnormal)
    
    Returns:
        Retraining results and new model performance metrics
    """
    if not all(f.filename and f.filename.endswith('.wav') for f in files):
        raise HTTPException(status_code=400, detail="All files must be .wav format")
    
    if labels and len(files) != len(labels):
        raise HTTPException(status_code=400, detail="Number of files and labels must match")
    
    try:
        # Save files temporarily and extract features
        temp_files = []
        features_list = []
        
        for file in files:
            with NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_files.append(temp_file.name)
                
                # Extract features
                features = preprocessor.preprocess_audio(temp_file.name)
                if features is not None:
                    features_list.append(features)
        
        if not features_list:
            raise HTTPException(status_code=400, detail="Failed to extract features from files")
        
        # Combine features and scale
        X_new = np.vstack(features_list)
        X_new_scaled = preprocessor.transform(X_new)
        
        # Use provided labels or extract from filenames
        if labels:
            y_new = np.array(labels)
        else:
            # Extract labels from filenames and ensure we have both normal and abnormal samples
            y_new = np.array([1 if f.filename and 'abnormal' in f.filename.lower() else 0 for f in files])
            if len(np.unique(y_new)) < 2:
                raise HTTPException(
                    status_code=400, 
                    detail="Training data must include both normal and abnormal samples"
                )
        
        # Split data into train and validation sets
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_new_scaled, y_new, 
            test_size=0.2, 
            stratify=y_new,
            random_state=42
        )
        
        # Retrain model with validation set
        model.retrain(X_train, y_train, X_val, y_val, version="retrained")
        
        # Save model using storage
        storage.save_model(model.model, version="retrained")
        
        # Evaluate new model on full dataset
        eval_results = model.evaluate(X_new_scaled, y_new)
        
        # Clean up temp files
        for temp_file in temp_files:
            os.unlink(temp_file)
        
        return JSONResponse({
            "message": "Model successfully retrained",
            "performance": eval_results['classification_report']
        })
        # Removed duplicate return statement
    
    except Exception as e:
        # Clean up temp files in case of error
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info/")
async def get_model_info(api_key: str = Depends(verify_api_key)):
    api_logger.info("Model info requested")
    """Get information about the current model."""
    try:
        return JSONResponse({
            "model_type": "Random Forest Classifier",
            "model_path": model.model_dir,
            "scaler_path": preprocessor.model_dir,
            "is_trained": model.model is not None
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate/")
async def evaluate(api_key: str = Depends(verify_api_key)):
    api_logger.info("Model evaluation requested")
    """Get current model evaluation metrics."""
    try:
        if not hasattr(model, 'model') or model.model is None:
            raise HTTPException(status_code=400, detail="Model not trained")
            
        # Get latest test data from saved files
        data_dir = '../data'
        test_features = os.path.join(data_dir, 'test_features.csv')
        
        if not os.path.exists(test_features):
            raise HTTPException(status_code=404, detail="Test data not found")
            
        # Load test data
        test_df = pd.read_csv(test_features)
        X_test = test_df.drop(['file_path', 'label', 'augmented'], axis=1)
        y_test = test_df['label']
        
        # Scale features
        X_test_scaled = preprocessor.transform(X_test)
        
        # Get evaluation metrics
        eval_results = model.evaluate(X_test_scaled, y_test)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'classification_report': eval_results['classification_report'],
            'confusion_matrix': eval_results['confusion_matrix'].tolist(),
            'probabilities': eval_results['probabilities'].tolist() if isinstance(eval_results['probabilities'], np.ndarray) else eval_results['probabilities']
        }
        
        return JSONResponse(serializable_results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("prediction:app", host="0.0.0.0", port=8000, reload=True)