import os
import boto3
import joblib
from datetime import datetime
from src import config

class ModelStorage:
    def __init__(self):
        self.storage_type = config.STORAGE_TYPE
        if self.storage_type == 's3':
            self.s3 = boto3.client('s3')
            self.bucket = config.S3_BUCKET
        
    def save_model(self, model, version=None):
        """Save model to storage (local or S3)."""
        if version:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'random_forest_model_v{version}_{timestamp}.pkl'
        else:
            filename = 'random_forest_model.pkl'
        
        local_path = os.path.join(config.MODEL_DIR, filename)
        
        # Save locally first
        joblib.dump(model, local_path)
        
        # If using S3, upload to cloud
        if self.storage_type == 's3':
            try:
                self.s3.upload_file(local_path, self.bucket, f'models/{filename}')
                # Delete local file after successful upload if it's not the latest model
                if version and os.path.exists(local_path):
                    os.remove(local_path)
            except Exception as e:
                print(f"Error uploading to S3: {str(e)}")
        
        return local_path
    
    def load_model(self, version=None):
        """Load model from storage."""
        if version:
            # List available models and find the matching version
            models = self.list_models()
            model_file = next((m for m in models if f'_v{version}_' in m), None)
            if not model_file:
                raise ValueError(f"Model version {version} not found")
        else:
            model_file = 'random_forest_model.pkl'
        
        local_path = os.path.join(config.MODEL_DIR, model_file)
        
        # If using S3 and file not found locally, download from cloud
        if self.storage_type == 's3' and not os.path.exists(local_path):
            try:
                self.s3.download_file(self.bucket, f'models/{model_file}', local_path)
            except Exception as e:
                raise Exception(f"Error downloading model from S3: {str(e)}")
        
        return joblib.load(local_path)
    
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
