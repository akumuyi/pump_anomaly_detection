import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

class PumpAnomalyDetector:
    def __init__(self, model_dir='../models'):
        """Initialize the model with path to model directory."""
        self.model_dir = model_dir
        self.model = None
        self.threshold = 0.5  # Default threshold for anomaly detection
        self._load_model()

    def _load_model(self):
        """Load the trained model if it exists."""
        model_path = os.path.join(self.model_dir, 'random_forest_model.pkl')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)

    def save_model(self, version=None):
        """Save the trained model."""
        if version:
            # Save versioned model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f'random_forest_model_v{version}_{timestamp}.pkl'
        else:
            # Save as latest model
            model_name = 'random_forest_model.pkl'
        
        model_path = os.path.join(self.model_dir, model_name)
        joblib.dump(self.model, model_path)
        return model_path

    def train(self, X_train, y_train, params=None):
        """Train the model with given parameters."""
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': 42
            }
        
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get probability predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        
        # Calculate various metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'probabilities': y_prob
        }

    def retrain(self, X_new, y_new, X_val=None, y_val=None, version=None):
        """
        Retrain the model with new data.
        
        Args:
            X_new: New training features
            y_new: New training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            version: Version number for the retrained model
        """
        # Save current model as backup
        if self.model is not None:
            backup_path = self.save_model(version='backup')
            print(f"Backup saved to {backup_path}")

        # Train model on new data
        self.train(X_new, y_new)

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            eval_results = self.evaluate(X_val, y_val)
            print("\nValidation Results:")
            print(f"Classification Report:\n{eval_results['classification_report']}")
        
        # Save retrained model
        model_path = self.save_model(version=version)
        print(f"Retrained model saved to {model_path}")
        
        return self.model

    def should_retrain(self, X_recent, y_recent, threshold=0.1):
        """
        Determine if model should be retrained based on recent performance.
        
        Args:
            X_recent: Features from recent data
            y_recent: True labels from recent data
            threshold: Performance degradation threshold to trigger retraining
        
        Returns:
            bool: True if retraining is recommended
        """
        if self.model is None:
            return True

        # Get current performance on recent data
        eval_results = self.evaluate(X_recent, y_recent)
        current_f1 = eval_results['classification_report']['weighted avg']['f1-score']

        # If F1 score drops below threshold, recommend retraining
        if current_f1 < (1 - threshold):
            return True
        
        return False