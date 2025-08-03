import os
import boto3
import joblib
from datetime import datetime
from . import config
from .logging_config import model_logger as logger

class ModelStorage:
    def __init__(self):
        self.storage_type = config.STORAGE_TYPE
        if self.storage_type == 's3':
            try:
                self.s3 = boto3.client(
                    's3',
                    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                    region_name=config.S3_REGION
                )
                self.bucket = config.S3_BUCKET
                logger.info(f"Initialized S3 storage with bucket: {self.bucket}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {str(e)}")
                raise
        
    def save_model(self, model, version=None):
        """Save model to storage (local or S3)."""
        if version:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'random_forest_model_v{version}_{timestamp}.pkl'
        else:
            filename = 'random_forest_model.pkl'
        
        local_path = os.path.join(config.MODEL_DIR, filename)
        
        try:
            # Save locally first
            logger.info(f"Saving model to local path: {local_path}")
            joblib.dump(model, local_path)
            
            # If using S3, upload to cloud
            if self.storage_type == 's3':
                try:
                    logger.info(f"Uploading model to S3: {filename}")
                    self.s3.upload_file(local_path, self.bucket, f'models/{filename}')
                    logger.info("Model uploaded to S3 successfully")
                    
                    # Delete local file after successful upload if it's not the latest model
                    if version and os.path.exists(local_path):
                        os.remove(local_path)
                        logger.info(f"Removed local version of model: {filename}")
                except Exception as e:
                    logger.error(f"Error uploading to S3: {str(e)}")
                    logger.info("Keeping local copy due to S3 upload failure")
            return local_path
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except:
                    pass
            raise
        
        return local_path
    
    def load_model(self, version=None):
        """Load model from storage."""
        try:
            if version:
                # List available models and find the matching version
                models = self.list_models()
                model_file = next((m for m in models if f'_v{version}_' in m), None)
                if not model_file:
                    msg = f"Model version {version} not found"
                    logger.error(msg)
                    raise ValueError(msg)
            else:
                model_file = 'random_forest_model.pkl'
            
            local_path = os.path.join(config.MODEL_DIR, model_file)
            logger.info(f"Attempting to load model: {model_file}")
            
            # If using S3 and file not found locally, download from cloud
            if self.storage_type == 's3' and not os.path.exists(local_path):
                try:
                    logger.info(f"Downloading model from S3: {model_file}")
                    self.s3.download_file(self.bucket, f'models/{model_file}', local_path)
                    logger.info("Model downloaded successfully")
                except Exception as e:
                    logger.error(f"Error downloading model from S3: {str(e)}")
                    raise
            
            # Load the model
            logger.info(f"Loading model from local path: {local_path}")
            model = joblib.load(local_path)
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def list_models(self):
        """List all available model versions."""
        if self.storage_type == 's3':
            try:
                response = self.s3.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix='models/'
                )
                return [obj['Key'].split('/')[-1] for obj in response.get('Contents', [])]
            except Exception as e:
                print(f"Error listing models from S3: {str(e)}")
                return []
        else:
            return [f for f in os.listdir(config.MODEL_DIR) 
                   if f.startswith('random_forest_model')]

    def delete_model(self, version):
        """Delete a specific model version."""
        models = self.list_models()
        model_file = next((m for m in models if f'_v{version}_' in m), None)
        
        if not model_file:
            raise ValueError(f"Model version {version} not found")
        
        local_path = os.path.join(config.MODEL_DIR, model_file)
        
        # Delete from S3 if using cloud storage
        if self.storage_type == 's3':
            try:
                self.s3.delete_object(
                    Bucket=self.bucket,
                    Key=f'models/{model_file}'
                )
            except Exception as e:
                print(f"Error deleting model from S3: {str(e)}")
        
        # Delete local file if it exists
        if os.path.exists(local_path):
            os.remove(local_path)
