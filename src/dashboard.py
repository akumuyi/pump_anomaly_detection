import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time
import requests
import json
from model import PumpAnomalyDetector
from preprocessing import AudioPreprocessor
import librosa
import librosa.display
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import boto3

# Initialize model and preprocessor
model = PumpAnomalyDetector()
preprocessor = AudioPreprocessor()

# FastAPI endpoint
API_URL = os.environ.get('API_URL', 'http://localhost:8000')
API_KEY = os.environ.get('API_KEY', None)
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

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
    model_path = os.path.join('../models', 'random_forest_model.pkl')
    if os.path.exists(model_path):
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
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
        metrics_file = os.path.join('../data', 'metrics_history.csv')
    if not metrics_file or not os.path.exists(metrics_file):
        df = pd.DataFrame(columns=['timestamp', 'accuracy', 'precision', 'recall', 'f1'])
    else:
        df = pd.read_csv(metrics_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get latest metrics from API
    try:
        response = requests.get(f"{API_URL}/evaluate/", headers=HEADERS, timeout=15)
        if response.status_code == 200:
            latest_eval = response.json()
            if 'classification_report' in latest_eval:
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
    except Exception as e:
        st.warning(f"Could not update metrics history: {str(e)}")
    
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

if page == "Model Monitoring":
    st.header("Model Monitoring")
    
    # Model uptime
    col1, col2, col3 = st.columns(3)
    uptime = get_model_uptime()
    if uptime:
        col1.metric("Model Uptime", f"{uptime.days} days {uptime.seconds//3600} hours")
    
    # Get model info
    try:
        response = requests.get(f"{API_URL}/model-info/", headers=HEADERS, timeout=10)
        if response.status_code == 200:
            model_info = response.json()
            col2.metric("Model Type", model_info['model_type'])
            col3.metric("Model Status", "Active" if model_info['is_trained'] else "Not Trained")
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"Could not connect to the API: {str(e)}")
    
    # Performance metrics over time
    st.subheader("Performance Metrics History")
    metrics_fig = plot_metrics_history()
    st.plotly_chart(metrics_fig, use_container_width=True)
    
    # Latest evaluation metrics
    if hasattr(model, 'model') and model.model is not None:
        st.subheader("Latest Model Evaluation")
        try:
            response = requests.get(f"{API_URL}/evaluate/", headers=HEADERS, timeout=15)
            if response.status_code == 200:
                latest_eval = response.json()
                
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
                st.warning(f"API Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.warning(f"Could not fetch model evaluation metrics from API: {str(e)}")
    
    # Feature Analysis Section
    st.header("Feature Analysis")
    if STORAGE_TYPE == 's3':
        s3_train_path = 'data/train_features_augmented.csv'
        local_train_path = '/tmp/train_features_augmented.csv'
        train_features = fetch_s3_file(s3_train_path, local_train_path)
    else:
        data_dir = '../data'
        train_features = os.path.join(data_dir, 'train_features_augmented.csv')
    if train_features and os.path.exists(train_features):
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
        normal_audio = '../data/train/normal/normal_00000009.wav'
        abnormal_audio = '../data/train/abnormal/abnormal_00000024.wav'
        
        if os.path.exists(normal_audio) and os.path.exists(abnormal_audio):
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
            st.warning("Sample audio files not found. Please ensure data paths are correct.")
        
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
        
        # Display audio visualizations
        st.subheader("Audio Visualization")
        vis_fig = visualize_audio(temp_path, "Uploaded Audio")
        st.pyplot(vis_fig)
        
        # Make prediction
        try:
            with open(temp_path, 'rb') as file_handle:
                files = {'file': ('audio.wav', file_handle.read(), 'audio/wav')}
                response = requests.post(f"{API_URL}/predict/", files=files, headers=HEADERS, timeout=30)
            
            prediction = response.json()
            
            # Display prediction
            col1, col2 = st.columns(2)
            col1.metric("Prediction", prediction['prediction'].upper())
            col2.metric("Confidence", f"{prediction['probability']:.2%}")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
        finally:
            try:
                time.sleep(0.1)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
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
                for file in normal_files:
                    name = file.name if 'normal' in file.name.lower() else f'normal_{file.name}'
                    files.append(('files', (name, file.getvalue(), 'audio/wav')))
                
                for file in abnormal_files:
                    name = file.name if 'abnormal' in file.name.lower() else f'abnormal_{file.name}'
                    files.append(('files', (name, file.getvalue(), 'audio/wav')))
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                # Step 2: Feature Extraction
                status_text.markdown("📊 Extracting audio features...")
                time.sleep(0.5)  # Add small delay to show progress
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                # Step 3: Model Training
                status_text.markdown("🔄 Training model...")
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                response = requests.post(f"{API_URL}/retrain/", files=files, headers=HEADERS, timeout=120)
                result = response.json()
                
                if response.status_code == 200:
                    # Step 4: Model Evaluation
                    status_text.markdown("📈 Evaluating model performance...")
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    
                    if isinstance(result['performance'], dict):
                        # Step 5: Finalizing
                        status_text.markdown("✅ Training completed successfully!")
                        current_step += 1
                        progress_bar.progress(1.0)
                        
                        with result_container.container():
                            st.success(result['message'])
                            st.subheader("New Model Performance")
                            metrics_df = pd.DataFrame(result['performance']).transpose()
                            
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
    models_dir = '../models'
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.startswith('random_forest_model')]
        if models:
            model_info = []
            for m in models:
                try:
                    # Handle both old and new format filenames
                    if '_v' in m:
                        version = m.split('_v')[-1].split('_')[0]
                        timestamp_str = m.split('_')[-1].split('.')[0]
                    else:
                        version = '1'  # Default version for old format
                        timestamp_str = str(int(time.time()))  # Use current timestamp
                    
                    # Get file modification time as backup
                    mod_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(models_dir, m)))
                    
                    try:
                        # Try parsing the timestamp from filename
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    except ValueError:
                        # If parsing fails, use file modification time
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
                
                # Format timestamp for display
                models_df['Timestamp'] = models_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(
                    models_df[['Version', 'Timestamp', 'Filename']],
                    use_container_width=True
                )
            else:
                st.info("No valid model files found")
        else:
            st.info("No training history available")