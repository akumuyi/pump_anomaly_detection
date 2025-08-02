import os
import librosa
import numpy as np
from scipy import signal
import random
from sklearn.preprocessing import StandardScaler
import joblib

def add_noise(y, noise_level=0.005):
    """Add random noise to the audio signal."""
    noise = np.random.normal(0, noise_level, len(y))
    return y + noise

def apply_time_masking(y, n_masks=2, mask_size=0.1):
    """Apply time masking augmentation."""
    y_masked = y.copy()
    len_y = len(y)
    mask_length = int(len_y * mask_size)
    for i in range(n_masks):
        start = random.randint(0, len_y - mask_length - 1)
        y_masked[start:start + mask_length] = 0
    return y_masked

def apply_reverb(y, sr, decay=2.0):
    """Apply reverb effect to the audio signal."""
    impulse_length = int(sr * decay)
    impulse_response = np.exp(-np.linspace(0, decay, impulse_length))
    impulse_response = impulse_response / np.sum(impulse_response)
    return signal.convolve(y, impulse_response, mode='same')

def augment_audio(y, sr):
    """Apply various augmentations to the audio signal."""
    augmentations = []
    
    # Basic augmentations
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    augmentations.append(y_trimmed)
    
    # Pitch shifts
    try:
        augmentations.append(librosa.effects.pitch_shift(y=y_trimmed, sr=sr, n_steps=1))
        augmentations.append(librosa.effects.pitch_shift(y=y_trimmed, sr=sr, n_steps=-1))
    except Exception as e:
        print(f"Error in pitch shifting: {e}")
    
    # Time stretching
    try:
        augmentations.append(librosa.effects.time_stretch(y=y_trimmed, rate=1.1))
        augmentations.append(librosa.effects.time_stretch(y=y_trimmed, rate=0.9))
    except Exception as e:
        print(f"Error in time stretching: {e}")
    
    # Noise overlay
    augmentations.append(add_noise(y_trimmed, noise_level=0.002))
    augmentations.append(add_noise(y_trimmed, noise_level=0.005))
    
    # Time masking
    augmentations.append(apply_time_masking(y_trimmed, n_masks=1, mask_size=0.05))
    
    # Reverb
    augmentations.append(apply_reverb(y_trimmed, sr, decay=0.5))
    
    return augmentations

def extract_features(audio_file, augment=False):
    """Extract audio features from a file."""
    try:
        y, sr = librosa.load(audio_file, sr=None)
        features_list = []
        
        # Extract features for original audio
        features = extract_single_features(y, sr, audio_file)
        if features:
            features_list.append(features)
        
        # Extract features for augmented versions if augment=True
        if augment:
            augmentations = augment_audio(y, sr)
            for aug_y in augmentations:
                aug_features = extract_single_features(aug_y, sr, audio_file, augmented=True)
                if aug_features:
                    features_list.append(aug_features)
        
        return features_list
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return []

def extract_single_features(y, sr, file_path, augmented=False):
    """Extract features from a single audio signal."""
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        
        feature_stats = {
            'file_path': file_path,
            'label': 1 if 'abnormal' in file_path else 0,
            'augmented': augmented
        }
        
        # Extract statistical features from MFCCs
        for i, mfcc in enumerate(mfccs):
            feature_stats[f'mfcc{i+1}_mean'] = np.mean(mfcc)
            feature_stats[f'mfcc{i+1}_std'] = np.std(mfcc)
            feature_stats[f'mfcc{i+1}_max'] = np.max(mfcc)
            feature_stats[f'mfcc{i+1}_min'] = np.min(mfcc)
        
        # Extract statistical features from other audio characteristics
        for name, feature in [
            ('spectral_centroid', spectral_centroid),
            ('spectral_bandwidth', spectral_bandwidth),
            ('spectral_rolloff', spectral_rolloff),
            ('zero_crossing_rate', zero_crossing_rate),
            ('rms', rms)
        ]:
            feature_stats[f'{name}_mean'] = np.mean(feature)
            feature_stats[f'{name}_std'] = np.std(feature)
            feature_stats[f'{name}_max'] = np.max(feature)
            feature_stats[f'{name}_min'] = np.min(feature)
        
        return feature_stats
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

class AudioPreprocessor:
    def __init__(self, model_dir='../models'):
        """Initialize the preprocessor with model directory path."""
        self.model_dir = model_dir
        self.scaler = None
        self._load_scaler()

    def _load_scaler(self):
        """Load the trained scaler."""
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = StandardScaler()

    def save_scaler(self):
        """Save the fitted scaler."""
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

    def preprocess_audio(self, audio_file, augment=False):
        """Preprocess a single audio file."""
        features_list = extract_features(audio_file, augment)
        if not features_list:
            return None
        
        # Convert features to array format
        feature_arrays = []
        for features in features_list:
            feature_dict = {k: v for k, v in features.items() 
                          if k not in ['file_path', 'label', 'augmented']}
            feature_arrays.append(list(feature_dict.values()))
        
        return np.array(feature_arrays)

    def fit_transform(self, X):
        """Fit the scaler and transform the data."""
        if self.scaler is None:
            self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.save_scaler()
        return X_scaled

    def transform(self, X):
        """Transform the data using the fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        return self.scaler.transform(X)