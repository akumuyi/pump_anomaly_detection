# Pump Anomaly Detection

A machine learning system for detecting anomalies in industrial pump sounds using audio analysis and Random Forest classification. The system includes a FastAPI backend service for predictions and model management, and a Streamlit dashboard for visualization and monitoring.

## Features

- **Audio Analysis**: Process pump sound recordings to detect abnormal operations
- **Real-time Predictions**: Upload and analyze audio files through API or dashboard
- **Interactive Dashboard**: Visualize features, monitor model performance, and manage training
- **Cloud Storage**: AWS S3 integration for model versioning and persistence
- **Production-Ready**: Includes authentication, logging, and monitoring features

## Project Structure

```
pump_anomaly_detection/
├── data/
│   ├── test_features.csv
│   ├── train_features_augmented.csv
│   └── pump/
│       ├── id_00/
│       ├── id_02/
│       ├── id_04/
│       └── id_06/
├── models/
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── src/
│   ├── model.py            # Machine learning model implementation
│   ├── prediction.py       # FastAPI service endpoints
│   ├── preprocessing.py    # Audio feature extraction
│   ├── dashboard.py        # Streamlit dashboard
│   ├── config.py          # Configuration management
│   ├── security.py        # Authentication and authorization
│   ├── storage.py         # Model storage (local/S3)
│   └── logging_config.py   # Logging setup
├── logs/                   # Application logs
├── requirements.txt        # Project dependencies
└── render.yaml            # Render deployment configuration
```

## Technical Stack

- **Backend**: Python, FastAPI
- **Frontend**: Streamlit
- **ML Framework**: scikit-learn (Random Forest)
- **Audio Processing**: librosa
- **Storage**: AWS S3
- **Deployment**: Render
- **Authentication**: API Key
- **Visualization**: Plotly, Matplotlib

## Features Analysis

The system analyzes several audio features to detect pump anomalies:

1. **Mel-Frequency Cepstral Coefficients (MFCCs)**
   - Captures sound timbre and spectral characteristics
   - Helps identify irregular vibrations and mechanical faults

2. **Spectral Features**
   - Frequency Center: Indicates average frequency
   - Frequency Spread: Shows frequency distribution
   - Energy Roll-off: Measures frequency energy distribution

3. **Time-Domain Features**
   - Signal Energy: Measures sound intensity
   - Signal Complexity: Analyzes waveform patterns

## API Endpoints

### `/predict/`
- **Method**: POST
- **Purpose**: Make predictions on new audio files
- **Input**: .wav audio file
- **Output**: Prediction (normal/abnormal) with confidence score

### `/retrain/`
- **Method**: POST
- **Purpose**: Retrain model with new data
- **Input**: Multiple .wav files (normal and abnormal samples)
- **Output**: Training results and model performance metrics

### `/evaluate/`
- **Method**: GET
- **Purpose**: Get current model performance metrics
- **Output**: Classification report and confusion matrix

### `/model-info/`
- **Method**: GET
- **Purpose**: Get information about the current model
- **Output**: Model type, path, and training status

## Dashboard Sections

1. **Model Monitoring**
   - Model uptime and status
   - Performance metrics history
   - Latest evaluation metrics
   - Feature analysis visualizations

2. **Predictions**
   - Audio file upload
   - Waveform and spectrogram visualization
   - Real-time prediction results

3. **Training**
   - Separate upload for normal and abnormal samples
   - Training progress tracking
   - Model version history

## Setup and Deployment

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/akumuyi/pump_anomaly_detection.git
   cd pump_anomaly_detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export ENVIRONMENT=development
   export API_KEY=your_api_key
   export AWS_ACCESS_KEY_ID=your_aws_key
   export AWS_SECRET_ACCESS_KEY=your_aws_secret
   export S3_BUCKET=your_bucket_name
   ```

4. Run the services:
   ```bash
   # Start FastAPI server
   uvicorn src.prediction:app --reload

   # Start Streamlit dashboard
   streamlit run src/dashboard.py
   ```

### Production Deployment (Render)

1. Push code to GitHub

2. Connect repository to Render:
   - Create a new Blueprint
   - Select your repository
   - Render will automatically detect `render.yaml`

3. Set environment variables in Render dashboard:
   - `ENVIRONMENT=production`
   - `PYTHON_VERSION=3.13.5`
   - `API_KEY`
   - `SECRET_KEY`
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `S3_BUCKET`

## Security

- API key authentication for all endpoints
- Secure credential management
- JWT token support
- Environment-based configuration

## Monitoring and Logging

- Rotating file logs
- Separate loggers for API, model, and dashboard
- Performance metrics tracking
- Model version control

## Future Improvements

1. Real-time audio streaming analysis
2. Additional machine learning models
3. Automated model retraining triggers
4. Enhanced visualization options
5. Extended API functionality

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- **Your Name** - *Initial work* - [akumuyi](https://github.com/akumuyi)