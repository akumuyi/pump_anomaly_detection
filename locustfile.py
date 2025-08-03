import os
import time
import random
from pathlib import Path
import json
from locust import HttpUser, task, between, tag, events
from locust.exception import RescheduleTask
from locust.runners import MasterRunner
from locust.stats import RequestStats
from locust import events

# Import configuration if available
try:
    from src import config
    API_KEY = os.environ.get("API_KEY", config.API_KEY)
except ImportError:
    API_KEY = os.environ.get("API_KEY", "your-dev-api-key")

# Paths to test data
PROJECT_ROOT = Path(__file__).parent
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test"
NORMAL_AUDIO_DIR = TEST_DATA_DIR / "normal"
ABNORMAL_AUDIO_DIR = TEST_DATA_DIR / "abnormal"

# Global statistics
total_predictions = 0
normal_predictions = 0
abnormal_predictions = 0
avg_response_time = 0
peak_response_time = 0
min_response_time = float('inf')

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize test environment and print setup information."""
    print(f"Test setup initialized")
    print(f"API Key configured: {'Yes' if API_KEY else 'No'}")
    print(f"Test data directories:")
    print(f" - Normal audio: {NORMAL_AUDIO_DIR}")
    print(f" - Abnormal audio: {ABNORMAL_AUDIO_DIR}")
    
    if not NORMAL_AUDIO_DIR.exists() or not ABNORMAL_AUDIO_DIR.exists():
        print("WARNING: Test data directories not found. Tests may fail.")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log when test starts."""
    if isinstance(environment.runner, MasterRunner):
        print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print test summary when test finishes."""
    if isinstance(environment.runner, MasterRunner):
        print("\n--- Test Summary ---")
        print(f"Total predictions: {total_predictions}")
        normal_pct = (normal_predictions / total_predictions * 100) if total_predictions > 0 else 0
        abnormal_pct = (abnormal_predictions / total_predictions * 100) if total_predictions > 0 else 0
        print(f"Normal predictions: {normal_predictions} ({normal_pct:.1f}%)")
        print(f"Abnormal predictions: {abnormal_predictions} ({abnormal_pct:.1f}%)")
        print(f"Average response time: {avg_response_time:.2f} ms")
        print(f"Minimum response time: {min_response_time:.2f} ms")
        print(f"Peak response time: {peak_response_time:.2f} ms")

class PumpAnomalyUser(HttpUser):
    """Simulated user for load testing the pump anomaly detection API."""
    
    # Wait between 1 to 5 seconds between tasks
    wait_time = between(1, 5)
    
    def on_start(self):
        """Setup before starting tests."""
        # Cache available test files
        self.normal_files = list(NORMAL_AUDIO_DIR.glob("*.wav")) if NORMAL_AUDIO_DIR.exists() else []
        self.abnormal_files = list(ABNORMAL_AUDIO_DIR.glob("*.wav")) if ABNORMAL_AUDIO_DIR.exists() else []
        
        if not self.normal_files and not self.abnormal_files:
            print("ERROR: No test files found. Load test will fail.")
        else:
            print(f"Found {len(self.normal_files)} normal files and {len(self.abnormal_files)} abnormal files.")
            
        # Set default headers
        self.client.headers = {
            "X-API-Key": API_KEY,
            "Authorization": f"Bearer {API_KEY}"
        }
    
    @tag("health")
    @task(1)
    def check_health(self):
        """Check API health endpoint."""
        with self.client.get("/health/", name="Health Check", catch_response=True) as response:
            try:
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") == "healthy":
                        events.request.fire(
                            request_type="GET",
                            name="Health Check",
                            response_time=response.elapsed.total_seconds() * 1000,
                            response_length=len(response.content),
                            response=response,
                            context={},
                            exception=None
                        )
                    else:
                        events.request.fire(
                            request_type="GET",
                            name="Health Check",
                            response_time=response.elapsed.total_seconds() * 1000,
                            response_length=len(response.content),
                            response=response,
                            context={},
                            exception=Exception(f"API reported unhealthy status: {health_data}")
                        )
                else:
                    events.request.fire(
                        request_type="GET",
                        name="Health Check",
                        response_time=response.elapsed.total_seconds() * 1000,
                        response_length=len(response.content),
                        response=response,
                        context={},
                        exception=Exception(f"Health check failed: {response.status_code}")
                    )
            except Exception as e:
                events.request.fire(
                    request_type="GET",
                    name="Health Check",
                    response_time=0,
                    response_length=0,
                    response=response,
                    context={},
                    exception=e
                )
    
    @tag("model_info")
    @task(1)
    def get_model_info(self):
        """Get model information."""
        with self.client.get("/model-info/", name="Model Info", catch_response=True) as response:
            try:
                if response.status_code == 200:
                    model_info = response.json()
                    if model_info.get("is_trained"):
                        events.request.fire(
                            request_type="GET",
                            name="Model Info",
                            response_time=response.elapsed.total_seconds() * 1000,
                            response_length=len(response.content),
                            response=response,
                            context={},
                            exception=None
                        )
                    else:
                        events.request.fire(
                            request_type="GET",
                            name="Model Info",
                            response_time=response.elapsed.total_seconds() * 1000,
                            response_length=len(response.content),
                            response=response,
                            context={},
                            exception=Exception("Model not trained")
                        )
                else:
                    events.request.fire(
                        request_type="GET",
                        name="Model Info",
                        response_time=response.elapsed.total_seconds() * 1000,
                        response_length=len(response.content),
                        response=response,
                        context={},
                        exception=Exception(f"Could not get model info: {response.status_code}")
                    )
            except Exception as e:
                events.request.fire(
                    request_type="GET",
                    name="Model Info",
                    response_time=0,
                    response_length=0,
                    response=response,
                    context={},
                    exception=e
                )
    
    @tag("predict")
    @task(10)  # Higher weight means this task runs more frequently
    def predict_random_file(self):
        """Send random audio file for prediction."""
        global total_predictions, normal_predictions, abnormal_predictions, avg_response_time, peak_response_time, min_response_time
        
        # Skip if no files available
        if not self.normal_files and not self.abnormal_files:
            raise RescheduleTask("No audio files available for testing")
        
        # Randomly choose normal or abnormal file
        use_normal = random.choice([True, False])
        files_list = self.normal_files if use_normal and self.normal_files else self.abnormal_files
        
        if not files_list:
            files_list = self.normal_files if self.normal_files else self.abnormal_files
            
        if not files_list:
            raise RescheduleTask("No audio files available for testing")
        
        # Select random file
        audio_file = random.choice(files_list)
        expected_class = "normal" if "normal" in audio_file.name.lower() else "abnormal"
        
        start_time = time.time()
        
        # Send file for prediction
        try:
            with open(audio_file, "rb") as file:
                with self.client.post(
                    "/predict/",
                    files={"file": (audio_file.name, file, "audio/wav")},
                    params={"api_key": API_KEY},  # Also send as query param
                    name=f"Predict {expected_class}",
                    catch_response=True
                ) as response:
                    end_time = time.time()
                    response_time_ms = (end_time - start_time) * 1000
                    
                    if response.status_code == 200:
                        prediction = response.json()
                        predicted_class = prediction.get("prediction")
                        confidence = prediction.get("probability", 0)
                        processing_time = prediction.get("processing_time", 0)
                        
                        # Update global stats
                        total_predictions += 1
                        if predicted_class == "normal":
                            normal_predictions += 1
                        else:
                            abnormal_predictions += 1
                        
                        avg_response_time = ((avg_response_time * (total_predictions - 1)) + response_time_ms) / total_predictions
                        peak_response_time = max(peak_response_time, response_time_ms)
                        min_response_time = min(min_response_time, response_time_ms)
                        
                        # Log details for this request
                        self.environment.events.request.fire(
                            request_type="POST",
                            name=f"Prediction ({expected_class})",
                            response_time=response_time_ms,
                            response_length=len(response.content),
                            context={
                                "predicted_class": predicted_class,
                                "confidence": confidence,
                                "processing_time": processing_time
                            },
                            exception=None
                        )
                        
                        # Check if prediction matched expectation
                        if predicted_class == expected_class:
                            events.request.fire(
                                request_type="POST",
                                name=f"Prediction ({expected_class})",
                                response_time=response_time_ms,
                                response_length=len(response.content),
                                response=response,
                                context={
                                    "predicted_class": predicted_class,
                                    "confidence": confidence,
                                    "processing_time": processing_time
                                },
                                exception=None
                            )
                        else:
                            events.request.fire(
                                request_type="POST",
                                name=f"Prediction ({expected_class})",
                                response_time=response_time_ms,
                                response_length=len(response.content),
                                response=response,
                                context={
                                    "predicted_class": predicted_class,
                                    "confidence": confidence,
                                    "processing_time": processing_time
                                },
                                exception=Exception(f"Expected {expected_class}, but got {predicted_class} with confidence {confidence:.2%}")
                            )
                    else:
                        events.request.fire(
                            request_type="POST",
                            name=f"Prediction ({expected_class})",
                            response_time=response_time_ms,
                            response_length=len(response.content),
                            response=response,
                            context={},
                            exception=Exception(f"Prediction request failed: {response.status_code}")
                        )
        except Exception as e:
            self.environment.events.request.fire(
                request_type="POST",
                name=f"Prediction ({expected_class})",
                response_time=0,
                response_length=0,
                context={},
                exception=e
            )
    
    @tag("evaluate")
    @task(2)
    def evaluate_model(self):
        """Request model evaluation."""
        with self.client.get("/evaluate/", name="Model Evaluation", catch_response=True) as response:
            try:
                if response.status_code == 200:
                    eval_data = response.json()
                    if "classification_report" in eval_data:
                        accuracy = eval_data["classification_report"].get("accuracy", 0)
                        events.request.fire(
                            request_type="GET",
                            name="Model Evaluation",
                            response_time=response.elapsed.total_seconds() * 1000,
                            response_length=len(response.content),
                            response=response,
                            context={"accuracy": accuracy},
                            exception=None
                        )
                    else:
                        events.request.fire(
                            request_type="GET",
                            name="Model Evaluation",
                            response_time=response.elapsed.total_seconds() * 1000,
                            response_length=len(response.content),
                            response=response,
                            context={},
                            exception=Exception("Incomplete evaluation data")
                        )
                elif response.status_code == 400 and "Model not trained" in response.text:
                    # Skip if model is not trained, but mark as success since this is an expected state
                    events.request.fire(
                        request_type="GET",
                        name="Model Evaluation (Not Trained)",
                        response_time=response.elapsed.total_seconds() * 1000,
                        response_length=len(response.content),
                        response=response,
                        context={},
                        exception=None
                    )
                else:
                    events.request.fire(
                        request_type="GET",
                        name="Model Evaluation",
                        response_time=response.elapsed.total_seconds() * 1000,
                        response_length=len(response.content),
                        response=response,
                        context={},
                        exception=Exception(f"Evaluation request failed: {response.status_code}")
                    )
            except Exception as e:
                events.request.fire(
                    request_type="GET",
                    name="Model Evaluation",
                    response_time=0,
                    response_length=0,
                    response=response,
                    context={},
                    exception=e
                )
                
if __name__ == "__main__":
    # Print usage instructions when run directly
    print("Locust Load Testing Script for Pump Anomaly Detection API")
    print("=========================================================")
    print("To run this script with Locust web UI:")
    print("    locust -f locustfile.py")
    print("\nTo run with command line options:")
    print("    locust -f locustfile.py --headless -u 10 -r 2 -t 60s -H http://localhost:8000")
    print("\nCommand explanation:")
    print("    -u 10     : 10 simulated users")
    print("    -r 2      : Spawn 2 users per second")
    print("    -t 60s    : Run for 60 seconds")
    print("    -H [url]  : Host URL of your API")
