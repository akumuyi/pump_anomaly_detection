from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
from tempfile import NamedTemporaryFile
from typing import List, Optional
import numpy as np
import pandas as pd
from src.preprocessing import AudioPreprocessor
from src.model import PumpAnomalyDetector
from src.security import verify_api_key
from src.storage import ModelStorage
from src.logging_config import api_logger
from src import config

app = FastAPI(
    title="Pump Anomaly Detection API",
    description="API for detecting anomalies in pump sounds using machine learning",
    version="1.0.0"
)

# Use ModelStorage for model persistence
storage = ModelStorage()

# Initialize preprocessor and model
try:
    api_logger.info("Initializing preprocessor...")
    preprocessor = AudioPreprocessor()
    api_logger.info(f"Preprocessor initialized. Scaler fitted: {preprocessor.scaler is not None}")
    
    api_logger.info("Initializing model...")
    model = PumpAnomalyDetector()
    api_logger.info(f"Model initialized. Model loaded: {model.model is not None}")
    
    if model.model is None:
        api_logger.warning("No pre-trained model found. Model will need to be trained before use.")
    else:
        api_logger.info("Pre-trained model loaded successfully.")
        
except Exception as e:
    api_logger.error(f"Error during initialization: {str(e)}")
    raise

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
    
    # Check if model is loaded
    if not hasattr(model, 'model') or model.model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
    
    temp_path = None
    try:
        import time
        start_time = time.time()
        
        # Save uploaded file temporarily
        api_logger.info("Saving uploaded file...")
        with NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        save_time = time.time()
        api_logger.info(f"File saved in {save_time - start_time:.2f} seconds")
        
        # Preprocess audio
        api_logger.info("Extracting features...")
        features = preprocessor.preprocess_audio(temp_path)
        if features is None:
            raise HTTPException(status_code=400, detail="Failed to extract features from audio")
        
        feature_time = time.time()
        api_logger.info(f"Features extracted in {feature_time - save_time:.2f} seconds")
        api_logger.info(f"Feature shape: {features.shape}")
        
        # Scale features
        api_logger.info("Scaling features...")
        features_scaled = preprocessor.transform(features)
        
        scale_time = time.time()
        api_logger.info(f"Features scaled in {scale_time - feature_time:.2f} seconds")
        
        # Make prediction
        api_logger.info("Making prediction...")
        probabilities = model.predict_proba(features_scaled)
        prediction = model.predict(features_scaled)
        
        predict_time = time.time()
        api_logger.info(f"Prediction made in {predict_time - scale_time:.2f} seconds")
        
        # For normal predictions (0), use the probability of class 0
        # For abnormal predictions (1), use the probability of class 1
        pred_class = int(prediction[0])
        confidence = float(probabilities[0][pred_class])
        
        total_time = time.time()
        api_logger.info(f"Total prediction time: {total_time - start_time:.2f} seconds")
        
        return JSONResponse({
            "prediction": "abnormal" if pred_class == 1 else "normal",
            "probability": confidence,
            "confidence": confidence,
            "processing_time": round(total_time - start_time, 2)
        })
    
    except Exception as e:
        api_logger.error(f"Error during prediction: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                api_logger.info("Temporary file cleaned up")
            except Exception as e:
                api_logger.warning(f"Failed to clean up temp file: {e}")

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
        X_new_scaled = preprocessor.fit_transform(X_new)  # Use fit_transform for retraining
        
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

@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    try:
        model_status = "loaded" if hasattr(model, 'model') and model.model is not None else "not loaded"
        scaler_status = "fitted" if preprocessor.scaler is not None else "not fitted"
        
        return JSONResponse({
            "status": "healthy",
            "model_status": model_status,
            "scaler_status": scaler_status,
            "timestamp": str(pd.Timestamp.now())
        })
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": str(pd.Timestamp.now())
        })

@app.get("/evaluate/")
async def evaluate(api_key: str = Depends(verify_api_key)):
    api_logger.info("Model evaluation requested")
    """Get current model evaluation metrics."""
    try:
        if not hasattr(model, 'model') or model.model is None:
            raise HTTPException(status_code=400, detail="Model not trained")
            
        # Get latest test data from saved files
        data_dir = os.path.join(config.BASE_DIR, 'data')
        test_features = os.path.join(data_dir, 'test_features.csv')
        
        if not os.path.exists(test_features):
            api_logger.error(f"Test data file not found at: {test_features}")
            raise HTTPException(status_code=404, detail="Test data not found")
            
        # Load test data
        api_logger.info(f"Loading test data from: {test_features}")
        test_df = pd.read_csv(test_features)
        api_logger.info(f"Test data shape: {test_df.shape}")
        api_logger.info(f"Test data columns: {list(test_df.columns)}")
        
        # Check if required columns exist
        required_columns = ['file_path', 'label']
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        if missing_columns:
            api_logger.error(f"Missing required columns: {missing_columns}")
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")
        
        # Drop metadata columns
        drop_columns = ['file_path', 'label']
        if 'augmented' in test_df.columns:
            drop_columns.append('augmented')
            
        X_test = test_df.drop(drop_columns, axis=1)
        y_test = test_df['label']
        
        api_logger.info(f"Feature matrix shape: {X_test.shape}")
        api_logger.info(f"Labels shape: {y_test.shape}")
        
        # Check if scaler is fitted
        if preprocessor.scaler is None:
            api_logger.error("Preprocessor scaler is not fitted")
            raise HTTPException(status_code=400, detail="Preprocessor scaler is not fitted. Please retrain the model first.")
        
        # Scale features
        api_logger.info("Scaling features...")
        X_test_scaled = preprocessor.transform(X_test)
        api_logger.info(f"Scaled features shape: {X_test_scaled.shape}")
        
        # Get evaluation metrics
        api_logger.info("Evaluating model...")
        eval_results = model.evaluate(X_test_scaled, y_test)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'classification_report': eval_results['classification_report'],
            'confusion_matrix': eval_results['confusion_matrix'].tolist(),
            'probabilities': eval_results['probabilities'].tolist() if isinstance(eval_results['probabilities'], np.ndarray) else eval_results['probabilities']
        }
        
        api_logger.info("Model evaluation completed successfully")
        return JSONResponse(serializable_results)
        
    except HTTPException as he:
        api_logger.error(f"HTTP Exception during model evaluation: {he.detail}")
        raise he
    except FileNotFoundError as fe:
        api_logger.error(f"File not found during model evaluation: {str(fe)}")
        raise HTTPException(status_code=404, detail=f"File not found: {str(fe)}")
    except ValueError as ve:
        api_logger.error(f"Value error during model evaluation: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Value error: {str(ve)}")
    except Exception as e:
        api_logger.error(f"Unexpected error during model evaluation: {type(e).__name__}: {str(e)}")
        import traceback
        api_logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("prediction:app", host="0.0.0.0", port=8000, reload=True)