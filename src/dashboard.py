import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time
import requests
from functools import wraps
from requests.exceptions import RequestException, ConnectionError, Timeout
import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Add src directory to Python path as well
src_dir = str(Path(__file__).parent)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from model import PumpAnomalyDetector
from preprocessing import AudioPreprocessor
import config
import librosa
import librosa.display
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import boto3

# Initialize model and preprocessor
model = PumpAnomalyDetector()
preprocessor = AudioPreprocessor()

# FastAPI endpoint configuration
ENVIRONMENT = os.environ.get('ENVIRONMENT', config.ENVIRONMENT)
DEFAULT_API_URL = 'https://pump-anomaly-api.onrender.com' if ENVIRONMENT == 'production' else 'http://localhost:8000'
API_URL = os.environ.get('API_URL', config.API_URL or DEFAULT_API_URL)
API_KEY = os.environ.get('API_KEY', config.API_KEY)  # Get from config if not in env

# Setup API authentication headers - FastAPI expects the API key in either:
# 1. Header named 'X-API-Key' (custom header)
# 2. Query parameter named 'api_key'
# 3. Header named 'Authorization' with "Bearer {token}"
# We'll use all three for maximum compatibility
HEADERS = {}
if API_KEY:
    HEADERS["X-API-Key"] = API_KEY
    HEADERS["Authorization"] = f"Bearer {API_KEY}"

# API calling utility function with retries and error handling
def call_api(method, endpoint, files=None, json_data=None, max_retries=3, timeout=None):
    """
    Call API with retry logic and error handling.
    
    Args:
        method (str): 'get' or 'post'
        endpoint (str): API endpoint (without base URL)
        files (dict, optional): Files for POST request
        json_data (dict, optional): JSON data for POST request
        max_retries (int): Maximum number of retry attempts
        timeout (int, optional): Request timeout in seconds. If None, uses endpoint-appropriate defaults.
        
    Returns:
        dict or None: JSON response or None if request failed
    """
    # Ensure API key is sent both in headers and as a query parameter for maximum compatibility
    url = f"{API_URL}/{endpoint.lstrip('/')}"
    params = {}
    if API_KEY:
        params['api_key'] = API_KEY
    
    # Set appropriate timeouts based on endpoint if none specified
    if timeout is None:
        if 'predict' in endpoint:
            # Audio processing takes longer, especially for larger files
            timeout = 60  # 60 seconds for prediction endpoints
        elif 'retrain' in endpoint:
            # Training can take a very long time
            timeout = 300  # 5 minutes for retraining
        elif 'evaluate' in endpoint:
            # Model evaluation involves processing test datasets
            timeout = 60  # 60 seconds for evaluation
        else:
            # Default for simpler endpoints
            timeout = 15  # 15 seconds for other endpoints
    
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            if method.lower() == 'get':
                response = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
            elif method.lower() == 'post':
                response = requests.post(url, headers=HEADERS, params=params, files=files, json=json_data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except ConnectionError:
            if attempt < max_retries - 1:
                st.warning(f"⚠️ Connection failed (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                # Use longer retry delay for prediction endpoints
                if 'predict' in endpoint or 'retrain' in endpoint:
                    retry_delay *= 3  # More aggressive exponential backoff for compute-heavy operations
                else:
                    retry_delay *= 2  # Standard exponential backoff
            else:
                if ENVIRONMENT != "production":
                    st.error(f"🌐 Connection Error: Cannot reach API at {API_URL}. Is the server running?")
                else:
                    st.error(f"🌐 Connection Error: Cannot reach API service. Please try again later.")
                return None
                
        except Timeout:
            if attempt < max_retries - 1:
                # Add more information about timeouts for prediction endpoints
                if 'predict' in endpoint:
                    st.warning(f"⚠️ Prediction request timed out (attempt {attempt+1}/{max_retries}). Audio processing may take longer than expected. Retrying with longer timeout...")
                    # Increase the timeout for the next attempt
                    timeout *= 1.5
                else:
                    st.warning(f"⚠️ Request timed out (attempt {attempt+1}/{max_retries}). Retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                if 'predict' in endpoint:
                    st.error("⏱️ Prediction request timed out. The audio file may be too large or complex to process in the allowed time.")
                else:
                    st.error("⏱️ API Request timed out. The server might be experiencing heavy load.")
                return None
                
        except RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                try:
                    error_detail = e.response.json().get('detail', f"HTTP {status_code}")
                except:
                    error_detail = f"HTTP {status_code}"
                
                if status_code == 401:
                    st.error("🔐 Authentication Error: Invalid API key")
                elif status_code == 404:
                    st.error("🔍 API endpoint not found. Please check the API URL.")
                else:
                    st.error(f"🚨 API Error: {error_detail}")
            else:
                st.error(f"🔧 Request Error: {str(e)}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return None
        
        except Exception as e:
            st.error(f"🔧 Unexpected Error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return None
    
    return None

# API Health Check function
def check_api_health():
    """
    Check if the API is online and functioning properly.
    
    Returns:
        dict or None: API health status information
    """
    try:
        # Use a direct request with minimal timeout for health checks
        url = f"{API_URL}/health/"
        # Setup params for API key authentication
        params = {}
        if API_KEY:
            params['api_key'] = API_KEY
            
        # Short timeout for health checks since they should be fast
        response = requests.get(url, headers=HEADERS, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.Timeout:
        # Silently fail on timeout - we'll just show the API as offline
        return None
    except Exception:
        # Silently fail on other exceptions too
        return None

# Storage configuration
STORAGE_TYPE = os.environ.get('STORAGE_TYPE', 'local')
S3_BUCKET = os.environ.get('S3_BUCKET')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

s3_client = None
if STORAGE_TYPE == 's3' and S3_BUCKET and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    s3_client = boto3.client(
        's3',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

def fetch_s3_file(s3_path, local_path):
    if s3_client:
        try:
            s3_client.download_file(S3_BUCKET, s3_path, local_path)
            return local_path
        except Exception as e:
            st.warning(f"Could not fetch {s3_path} from S3: {str(e)}")
            return None
    return None

def get_model_uptime():
    """Calculate model uptime based on file modification time."""
    model_path = Path(project_root) / 'models' / 'random_forest_model.pkl'
    if model_path.exists():
        mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
        uptime = datetime.now() - mod_time
        return uptime
    return None

def plot_confusion_matrix(confusion_matrix, labels):
    """Create a confusion matrix plot using plotly."""
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=labels,
        y=labels,
        hoverongaps=False,
        colorscale='Blues',
        text=confusion_matrix,
        texttemplate="%{text:.3f}"))
    
    fig.update_layout(
        title='Confusion Matrix for Pump Anomaly Detection',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        title_x=0.5
    )
    return fig

def plot_feature_importance(feature_names, importance_values):
    """Create a feature importance plot using plotly."""
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance in Pump Anomaly Detection')
    fig.update_layout(
        xaxis_title='Importance Score',
        yaxis_title='Audio Feature',
        title_x=0.5
    )
    return fig

def plot_metrics_history():
    """Plot historical model performance metrics."""
    if STORAGE_TYPE == 's3':
        s3_metrics_path = 'metrics/metrics_history.csv'
        local_metrics_path = '/tmp/metrics_history.csv'
        metrics_file = fetch_s3_file(s3_metrics_path, local_metrics_path)
    else:
        metrics_file = str(Path(project_root) / 'data' / 'metrics_history.csv')
    if not metrics_file or not Path(metrics_file).exists():
        df = pd.DataFrame(columns=['timestamp', 'accuracy', 'precision', 'recall', 'f1'])
    else:
        df = pd.read_csv(metrics_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get latest metrics from API
    latest_eval = call_api('get', '/evaluate/', max_retries=2)  # Use fewer retries here as it's non-critical
    
    if latest_eval and 'classification_report' in latest_eval:
        weighted_avg = latest_eval['classification_report']['weighted avg']
        latest_metrics = {
            'timestamp': datetime.now(),
            'accuracy': latest_eval['classification_report']['accuracy'],
            'precision': weighted_avg['precision'],
            'recall': weighted_avg['recall'],
            'f1': weighted_avg['f1-score']
        }
        
        # Append new metrics if they're different from the last entry
        if len(df) == 0 or not df.iloc[-1][1:].equals(pd.Series(latest_metrics)[1:]):
            df = pd.concat([df, pd.DataFrame([latest_metrics])], ignore_index=True)
            # Save updated history
            df.to_csv(metrics_file, index=False)
    
    if len(df) > 0:
        # Create line plot
        fig = px.line(df, x='timestamp', y=['accuracy', 'precision', 'recall', 'f1'],
                      title='Model Performance History Over Time')
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Metric Value',
            title_x=0.5,
            yaxis_range=[0, 1]  # Set y-axis range from 0 to 1
        )
    else:
        # Create empty plot if no data
        fig = px.line(title='Model Performance History Over Time')
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Metric Value',
            title_x=0.5,
            yaxis_range=[0, 1]
        )
    
    return fig

def visualize_audio(audio_file, title_prefix=""):
    """Create waveform and spectrogram visualizations for an audio file."""
    y, sr = librosa.load(audio_file)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='green')
    ax1.set_title(f'{title_prefix} Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', x_axis='time', ax=ax2, cmap='viridis')
    ax2.set_title(f'{title_prefix} Log-Frequency Spectrogram')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    plt.tight_layout()
    return fig

# Streamlit app
st.set_page_config(page_title="Pump Anomaly Detection Dashboard", layout="wide")

st.title("Pump Anomaly Detection Dashboard")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Model Monitoring", "Predictions", "Training"])

# API configuration and status
st.sidebar.markdown("### API Configuration")
st.sidebar.text(f"Environment: {ENVIRONMENT}")
st.sidebar.text(f"API URL: {API_URL}")

# Show API key status but keep it secure
if API_KEY:
    masked_key = API_KEY[:4] + "*" * (len(API_KEY) - 4) if len(API_KEY) > 4 else "****"
    st.sidebar.text(f"API Key: {masked_key}")
    
    # Add an expandable debug section for advanced users/admins
    with st.sidebar.expander("🔧 Authentication Debug Info", expanded=False):
        st.markdown("**Headers sent to API:**")
        for key, value in HEADERS.items():
            if key.lower() == "authorization":
                st.code(f"{key}: Bearer ***")
            elif key.lower() == "x-api-key":
                st.code(f"{key}: ***")
            else:
                st.code(f"{key}: {value}")
        st.markdown("**Also sending API key as:**")
        st.code("api_key=*** (query parameter)")
else:
    st.sidebar.text("API Key: Not Set")
    st.sidebar.warning("⚠️ API Key not configured. API calls will fail.")

# Check API health
health_status = check_api_health()
if health_status:
    st.sidebar.markdown("🟢 **API Status: Online**")
    st.sidebar.markdown(f"🔄 Model: {health_status['model_status']}")
    st.sidebar.markdown(f"📊 Scaler: {health_status['scaler_status']}")
else:
    st.sidebar.markdown("🔴 **API Status: Offline**")
    st.sidebar.markdown("⚠️ Unable to connect to API service")
    if ENVIRONMENT != "production":
        st.sidebar.markdown("""
        💡 **Troubleshooting Tips:**
        1. Make sure the API server is running
        2. Check that the API URL is correct
        3. Verify network connectivity
        """)
    else:
        st.sidebar.markdown("The API service may be experiencing issues. Please try again later.")

if page == "Model Monitoring":
    st.header("Model Monitoring")
    
    # Model uptime
    col1, col2, col3 = st.columns(3)
    uptime = get_model_uptime()
    if uptime:
        col1.metric("Model Uptime", f"{uptime.days} days {uptime.seconds//3600} hours")
    
    # Get model info
    model_info = call_api('get', '/model-info/')
    if model_info:
        col2.metric("Model Type", model_info['model_type'])
        col3.metric("Model Status", "Active" if model_info['is_trained'] else "Not Trained")
    
    # Performance metrics over time
    st.subheader("Performance Metrics History")
    metrics_fig = plot_metrics_history()
    st.plotly_chart(metrics_fig, use_container_width=True)
    
    # Latest evaluation metrics
    st.subheader("Latest Model Evaluation")
    with st.spinner("Fetching evaluation metrics..."):
        latest_eval = call_api('get', '/evaluate/', timeout=60)  # 60 second timeout for evaluation
    
    try:
        if latest_eval:
            # Display classification report if available
            if 'classification_report' in latest_eval:
                metrics_df = pd.DataFrame(latest_eval['classification_report']).transpose()
                
                # Replace numeric labels with descriptive ones
                metrics_df.rename(index={
                    '0': 'Normal',
                    '1': 'Abnormal',
                    'accuracy': 'Accuracy',
                    'macro avg': 'Macro Avg',
                    'weighted avg': 'Weighted Avg'
                }, inplace=True)
                
                # Round numeric values to 3 decimal places
                numeric_columns = ['precision', 'recall', 'f1-score', 'support']
                metrics_df[numeric_columns] = metrics_df[numeric_columns].round(3)
                
                # Style the dataframe
                st.dataframe(
                    metrics_df,
                    use_container_width=True,
                    height=250
                )
            
            # Display confusion matrix if available
            if 'confusion_matrix' in latest_eval:
                st.subheader("Confusion Matrix")
                cm_fig = plot_confusion_matrix(
                    latest_eval['confusion_matrix'], 
                    ['Normal', 'Abnormal']
                )
                st.plotly_chart(cm_fig, use_container_width=True)
        else:
            st.info("📋 Model evaluation data could not be retrieved. This could be because the model is not trained yet, or there was an issue connecting to the API.")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Could not connect to the API: {str(e)}")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch model evaluation metrics: {str(e)}")
    
    # Feature Analysis Section
    st.header("Feature Analysis")
    if STORAGE_TYPE == 's3':
        s3_train_path = 'data/train_features_augmented.csv'
        local_train_path = '/tmp/train_features_augmented.csv'
        train_features = fetch_s3_file(s3_train_path, local_train_path)
    else:
        train_features = str(Path(project_root) / 'data' / 'train_features_augmented.csv')
    if train_features and Path(train_features).exists():
        df = pd.read_csv(train_features)
        df['Status'] = df['label'].map({0: 'Normal', 1: 'Abnormal'})
        
        # 1. MFCC Analysis
        st.subheader("1. Mel-Frequency Cepstral Coefficients (MFCCs)")
        st.write("""
        MFCCs capture the timbre of pump sounds, reflecting the shape of the audio spectrum. 
        Differences in MFCCs can indicate irregular vibrations or mechanical faults in pumps.
        """)
        
        # Select top 3 MFCCs based on feature importance (assuming available)
        mfcc_cols = [col for col in df.columns if 'mfcc' in col and 'mean' in col][:3]
        mfcc_data = df[mfcc_cols + ['Status']]
        
        # Create box plot for top 3 MFCCs
        fig_mfcc = go.Figure()
        for i, mfcc in enumerate(mfcc_cols, 1):
            for status in ['Normal', 'Abnormal']:
                fig_mfcc.add_trace(go.Box(
                    y=mfcc_data[mfcc_data['Status'] == status][mfcc],
                    name=f"{status} - MFCC {i}",
                    boxpoints='outliers',
                    line_color='green' if status == 'Normal' else 'red'
                ))
        
        fig_mfcc.update_layout(
            title="Distribution of Top 3 MFCCs by Pump Status",
            xaxis_title="MFCC Coefficient",
            yaxis_title="Value",
            showlegend=True,
            title_x=0.5
        )
        st.plotly_chart(fig_mfcc, use_container_width=True)
        
        # 2. Spectral Features Analysis
        st.subheader("2. Spectral Features")
        st.write("""
        Spectral features describe the frequency content of pump sounds:
        - **Frequency Center**: Indicates the average frequency of the sound, often higher in faulty pumps due to irregular vibrations.
        - **Frequency Spread**: Shows the range of frequencies, wider in anomalies due to erratic sounds.
        - **Energy Roll-off**: Marks the frequency below which most energy lies, shifting in abnormal cases.
        """)
        
        spectral_features = ['spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_rolloff_mean']
        spectral_data = df[spectral_features + ['Status']]
        
        # Create violin plots for spectral features
        fig_spectral = go.Figure()
        for feature in spectral_features:
            feature_name = feature.replace('_mean', '').replace('spectral_', '').title()
            feature_name = 'Frequency Center' if feature_name == 'Centroid' else \
                          'Frequency Spread' if feature_name == 'Bandwidth' else 'Energy Roll-off'
            for status in ['Normal', 'Abnormal']:
                fig_spectral.add_trace(go.Violin(
                    y=spectral_data[spectral_data['Status'] == status][feature],
                    name=f"{status} - {feature_name}",
                    box_visible=True,
                    meanline_visible=True,
                    line_color='green' if status == 'Normal' else 'red'
                ))
        
        fig_spectral.update_layout(
            title="Distribution of Spectral Features by Pump Status",
            yaxis_title="Value",
            showlegend=True,
            title_x=0.5
        )
        st.plotly_chart(fig_spectral, use_container_width=True)
        
        # 3. Time-Domain Features Analysis
        st.subheader("3. Time-Domain Features")
        st.write("""
        Time-domain features reflect the signal's amplitude and complexity:
        - **Signal Energy**: Measures the loudness of the pump sound, often higher in anomalies due to mechanical stress.
        - **Signal Complexity**: Indicates how often the signal changes, higher in faulty pumps due to irregular patterns.
        """)
        
        time_features = ['rms_mean', 'zero_crossing_rate_mean']
        time_data = df[time_features + ['Status']]
        
        # Create scatter plot
        fig_time = px.scatter(
            time_data,
            x='rms_mean',
            y='zero_crossing_rate_mean',
            color='Status',
            color_discrete_map={'Normal': 'green', 'Abnormal': 'red'},
            title="Signal Energy vs. Signal Complexity",
            labels={
            'rms_mean': 'Signal Energy',
            'zero_crossing_rate_mean': 'Signal Complexity'
            }
        )
        
        fig_time.update_layout(
            legend_title="Pump Status", 
            title_x=0.5
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
        # 4. Spectrogram Comparison
        st.subheader("4. Spectrogram Comparison")
        st.write("""
        Spectrograms visualize the frequency content of pump sounds over time. 
        Normal pumps show consistent patterns, while abnormal pumps exhibit irregular frequency spikes or disruptions.
        """)

        # Select one normal and one abnormal audio file for comparison
        if STORAGE_TYPE == 's3':
            s3_normal_audio = 'data/train/normal/normal_00000009.wav'
            s3_abnormal_audio = 'data/train/abnormal/abnormal_00000024.wav'
            local_normal_audio = '/tmp/normal_00000009.wav'
            local_abnormal_audio = '/tmp/abnormal_00000024.wav'
            normal_audio = fetch_s3_file(s3_normal_audio, local_normal_audio)
            abnormal_audio = fetch_s3_file(s3_abnormal_audio, local_abnormal_audio)
        else:
            normal_audio = str(Path(project_root) / 'data' / 'train' / 'normal' / 'normal_00000009.wav')
            abnormal_audio = str(Path(project_root) / 'data' / 'train' / 'abnormal' / 'abnormal_00000024.wav')

        if normal_audio and abnormal_audio and Path(normal_audio).exists() and Path(abnormal_audio).exists():
            col1, col2 = st.columns(2)
            with col1:
                st.write("Normal Pump Spectrogram")
                fig_normal = visualize_audio(normal_audio, "Normal Pump")
                st.pyplot(fig_normal)
            with col2:
                st.write("Abnormal Pump Spectrogram")
                fig_abnormal = visualize_audio(abnormal_audio, "Abnormal Pump")
                st.pyplot(fig_abnormal)
        else:
            st.warning("Sample audio files not found. Please ensure data paths are correct or S3 is configured.")
        
        # Feature Interpretation
        st.subheader("Feature Interpretation Summary")
        st.write("""
        The visualizations above reveals the below information about pump health:

        1. **MFCC Patterns**:
           - Normal pumps exhibit stable MFCC values, reflecting consistent sound timbre.
           - Abnormal pumps show wider MFCC distributions, indicating irregular vibrations or friction, such as bearing faults.

        2. **Spectral Characteristics**:
           - Normal pumps have a stable frequency center, suggesting steady operation.
           - Abnormal pumps show higher variability in frequency center and wider spread, often due to mechanical stress or misalignment.
           - Energy roll-off shifts higher in anomalies, indicating more high-frequency noise from faults.

        3. **Time-Domain Behavior**:
           - Normal pumps have lower signal energy and complexity, reflecting smooth operation.
           - Abnormal pumps exhibit higher energy (louder sounds) and complexity (erratic patterns), signaling mechanical issues.

        4. **Spectrogram Insights**:
           - Normal pump spectrograms show uniform frequency bands, indicating regular operation.
           - Abnormal pump spectrograms reveal irregular spikes or gaps, corresponding to faults like imbalance or contamination.

        These features collectively enable the model to distinguish normal from anomalous pump sounds, supporting predictive maintenance in industrial settings.
        """)
    else:
        st.warning("Training data not found. Please make sure the data files are available.")

elif page == "Predictions":
    st.header("Make Predictions")
    
    uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=['wav'])
    
    if uploaded_file:
        # Save temporary file
        temp_path = f"temp_{int(time.time())}.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # If using S3, upload temp file to S3 for API access (optional, if API expects S3 path)
        s3_temp_path = None
        if STORAGE_TYPE == 's3' and s3_client and S3_BUCKET:
            s3_temp_path = f"temp/{os.path.basename(temp_path)}"
            try:
                s3_client.upload_file(temp_path, S3_BUCKET, s3_temp_path)
            except Exception as e:
                st.warning(f"Could not upload temp file to S3: {str(e)}")
        
        # Display audio visualizations
        st.subheader("Audio Visualization")
        vis_fig = visualize_audio(temp_path, "Uploaded Audio")
        st.pyplot(vis_fig)
        
        # Make prediction
        try:
            if STORAGE_TYPE == 's3' and s3_temp_path:
                # If API supports S3 path, send S3 path instead of file
                json_data = {'s3_path': s3_temp_path}
                st.info("📊 Processing audio from S3... This may take up to 2 minutes for larger or complex audio files.")
                with st.spinner("Analyzing audio patterns... Please wait."):
                    prediction = call_api('post', '/predict/', json_data=json_data, timeout=120)
            else:
                with open(temp_path, 'rb') as file_handle:
                    files = {'file': ('audio.wav', file_handle.read(), 'audio/wav')}
                    st.info("📊 Processing audio... This may take up to 2 minutes for larger or complex audio files.")
                    with st.spinner("Analyzing audio patterns... Please wait."):
                        prediction = call_api('post', '/predict/', files=files, timeout=120)
            
            if prediction:
                # Display prediction
                col1, col2 = st.columns(2)
                col1.metric("Prediction", prediction['prediction'].upper())
                col2.metric("Confidence", f"{prediction['probability']:.2%}")
            else:
                st.error("Could not get a prediction from the API")
                
        except Exception as e:
            st.error(f"Error processing prediction: {str(e)}")
        finally:
            try:
                time.sleep(0.1)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                # Optionally remove temp file from S3
                if STORAGE_TYPE == 's3' and s3_client and s3_temp_path:
                    try:
                        s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_temp_path)
                    except Exception:
                        pass
            except Exception as e:
                st.warning(f"Could not remove temporary file: {str(e)}")

elif page == "Training":
    st.header("Model Training")
    
    # Training data upload
    st.subheader("Upload Training Data")
    
    # Create two separate upload sections for normal and abnormal files
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Normal Pump Sounds")
        normal_files = st.file_uploader(
            "Upload normal pump audio files (.wav)", 
            type=['wav'], 
            accept_multiple_files=True,
            key="normal_files"
        )
        if normal_files:
            st.write(f"Uploaded {len(normal_files)} normal files")
            # Display sample names
            with st.expander("View uploaded normal files"):
                for file in normal_files:
                    st.text(f"📂 {file.name}")
    
    with col2:
        st.markdown("### Abnormal Pump Sounds")
        abnormal_files = st.file_uploader(
            "Upload abnormal pump audio files (.wav)", 
            type=['wav'], 
            accept_multiple_files=True,
            key="abnormal_files"
        )
        if abnormal_files:
            st.write(f"Uploaded {len(abnormal_files)} abnormal files")
            # Display sample names
            with st.expander("View uploaded abnormal files"):
                for file in abnormal_files:
                    st.text(f"📂 {file.name}")
    
    # Show training button only if both types of files are uploaded
    if normal_files and abnormal_files:
        total_files = len(normal_files) + len(abnormal_files)
        st.write(f"Total files ready for training: {total_files}")
        
        # Add warning for large dataset uploads
        if total_files > 20:
            st.warning(f"⚠️ You've uploaded {total_files} files. Training with a large number of audio files may take several minutes. Please be patient during processing.")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            train_button = st.button("Start Training", use_container_width=True)
        with col2:
            cancel_button = st.button("Cancel", use_container_width=True)
        
        if train_button:
            try:
                progress_container = st.empty()
                status_container = st.empty()
                result_container = st.empty()
                
                with progress_container.container():
                    st.markdown("### Training Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                # Initialize progress
                total_steps = 5
                current_step = 0
                
                # Step 1: Prepare files
                status_text.markdown("🔄 Preparing training files...")
                progress_bar.progress(current_step / total_steps)
                
                files = []
                s3_paths = []
                # Save files locally or upload to S3
                for file in normal_files:
                    name = file.name if 'normal' in file.name.lower() else f'normal_{file.name}'
                    if STORAGE_TYPE == 's3' and s3_client and S3_BUCKET:
                        s3_key = f"training/normal/{name}"
                        try:
                            # Save to temp then upload to S3
                            temp_file = f"/tmp/{name}"
                            with open(temp_file, "wb") as f:
                                f.write(file.getbuffer())
                            s3_client.upload_file(temp_file, S3_BUCKET, s3_key)
                            s3_paths.append({'label': 0, 's3_path': s3_key})
                            os.remove(temp_file)
                        except Exception as e:
                            st.warning(f"Could not upload {name} to S3: {str(e)}")
                    else:
                        files.append(('files', (name, file.getvalue(), 'audio/wav')))
                
                for file in abnormal_files:
                    name = file.name if 'abnormal' in file.name.lower() else f'abnormal_{file.name}'
                    if STORAGE_TYPE == 's3' and s3_client and S3_BUCKET:
                        s3_key = f"training/abnormal/{name}"
                        try:
                            temp_file = f"/tmp/{name}"
                            with open(temp_file, "wb") as f:
                                f.write(file.getbuffer())
                            s3_client.upload_file(temp_file, S3_BUCKET, s3_key)
                            s3_paths.append({'label': 1, 's3_path': s3_key})
                            os.remove(temp_file)
                        except Exception as e:
                            st.warning(f"Could not upload {name} to S3: {str(e)}")
                    else:
                        files.append(('files', (name, file.getvalue(), 'audio/wav')))
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                # Step 2: Feature Extraction
                status_text.markdown("📊 Extracting audio features...")
                time.sleep(0.5)
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                # Step 3: Model Training
                status_text.markdown("🔄 Training model... This may take several minutes depending on dataset size.")
                st.info("⏳ Model training is in progress. Please do not close this window or navigate away. For large datasets, this operation can take up to 5 minutes.")
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                if STORAGE_TYPE == 's3' and s3_paths:
                    # Send S3 paths to API
                    json_data = {'s3_files': s3_paths}
                    result = call_api('post', '/retrain/', json_data=json_data, timeout=300)
                else:
                    result = call_api('post', '/retrain/', files=files, timeout=300)
                
                if result:
                    # Step 4: Model Evaluation
                    status_text.markdown("📈 Evaluating model performance...")
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    
                    if isinstance(result.get('performance', None), dict):
                        # Step 5: Finalizing
                        status_text.markdown("✅ Training completed successfully!")
                        current_step += 1
                        progress_bar.progress(1.0)
                        
                        with result_container.container():
                            st.success(result['message'])
                            st.subheader("New Model Performance")
                            metrics_df = pd.DataFrame(result['performance']).transpose()
                            
                            metrics_df.rename(index={
                                '0': 'Normal',
                                '1': 'Abnormal',
                                'accuracy': 'Accuracy',
                                'macro avg': 'Macro Avg',
                                'weighted avg': 'Weighted Avg'
                            }, inplace=True)
                        
                        numeric_columns = ['precision', 'recall', 'f1-score', 'support']
                        metrics_df[numeric_columns] = metrics_df[numeric_columns].round(3)
                        
                        st.dataframe(
                            metrics_df,
                            use_container_width=True,
                            height=250
                        )
                else:
                    status_text.markdown("❌ Training failed!")
                    progress_bar.progress(1.0)
                    with result_container.container():
                        st.error(f"Training failed: {result.get('detail', 'Unknown error')}")
                    
            except Exception as e:
                status_text.markdown("❌ Error occurred!")
                progress_bar.progress(1.0)
                with result_container.container():
                    st.error(f"Error during training: {str(e)}")

    # Display training history
    st.subheader("Training History")
    if STORAGE_TYPE == 's3' and s3_client and S3_BUCKET:
        # List model files in S3 bucket
        try:
            response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix='models/')
            models = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    filename = os.path.basename(obj['Key'])
                    if filename.startswith('random_forest_model') and filename.endswith('.pkl'):
                        models.append({'Key': obj['Key'], 'Filename': filename, 'LastModified': obj['LastModified']})
            if models:
                model_info = []
                for m in models:
                    try:
                        if '_v' in m['Filename']:
                            version = m['Filename'].split('_v')[-1].split('_')[0]
                            timestamp_str = m['Filename'].split('_')[-1].split('.')[0]
                        else:
                            version = '1'
                            timestamp_str = m['LastModified'].strftime('%Y%m%d_%H%M%S')
                        mod_time = m['LastModified']
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        except ValueError:
                            timestamp = mod_time
                        model_info.append({
                            'Version': version,
                            'Timestamp': timestamp,
                            'Filename': m['Filename']
                        })
                    except Exception as e:
                        st.warning(f"Error processing model file {m['Filename']}: {str(e)}")
                if model_info:
                    models_df = pd.DataFrame(model_info)
                    models_df = models_df.sort_values('Timestamp', ascending=False)
                    models_df['Timestamp'] = models_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    st.dataframe(
                        models_df[['Version', 'Timestamp', 'Filename']],
                        use_container_width=True
                    )
                else:
                    st.info("No valid model files found in S3")
            else:
                st.info("No training history available in S3")
        except Exception as e:
            st.warning(f"Could not list models in S3: {str(e)}")
    else:
        models_dir = Path(project_root) / 'models'
        if models_dir.exists():
            models = [f.name for f in models_dir.glob('random_forest_model*.pkl')]
            if models:
                model_info = []
                for m in models:
                    try:
                        if '_v' in m:
                            version = m.split('_v')[-1].split('_')[0]
                            timestamp_str = m.split('_')[-1].split('.')[0]
                        else:
                            version = '1'
                            timestamp_str = str(int(time.time()))
                        mod_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(models_dir, m)))
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        except ValueError:
                            timestamp = mod_time
                        model_info.append({
                            'Version': version,
                            'Timestamp': timestamp,
                            'Filename': m
                        })
                    except Exception as e:
                        st.warning(f"Error processing model file {m}: {str(e)}")
                if model_info:
                    models_df = pd.DataFrame(model_info)
                    models_df = models_df.sort_values('Timestamp', ascending=False)
                    models_df['Timestamp'] = models_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    st.dataframe(
                        models_df[['Version', 'Timestamp', 'Filename']],
                        use_container_width=True
                    )
                else:
                    st.info("No valid model files found")
            else:
                st.info("No training history available")