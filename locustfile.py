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

# Load test configuration
from datetime import datetime

# Configuration
PRODUCTION_HOST = "https://pump-anomaly-api.onrender.com"  # Production API endpoint
DEFAULT_HOST = PRODUCTION_HOST  # Default to production server

# Endpoint-specific timeouts (adjusted for production latency)
DEFAULT_TIMEOUT = {
    "predict": 180,    # 180s timeout for predictions (increased for production)
    "evaluate": 90,    # 90s for model evaluation (increased for production)
    "retrain": 300,    # 300s for model retraining
    "default": 15      # 15s for other endpoints (increased for production latency)
}

# Import configuration if available
try:
    from src import config
    API_KEY = os.environ.get("API_KEY", config.API_KEY)
    # Always use production host unless explicitly overridden
    API_HOST = os.environ.get("API_HOST", PRODUCTION_HOST)
except ImportError:
    API_KEY = os.environ.get("API_KEY", "your-dev-api-key")
    API_HOST = os.environ.get("API_HOST", PRODUCTION_HOST)

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
test_start_time = None
success_count = 0
failure_count = 0
timeouts = 0

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize test environment and print setup information."""
    print("\n=== Pump Anomaly Detection Load Test ===")
    print(f"Configuration:")
    print(f"- API Host: {API_HOST}")
    print(f"- API Key configured: {'Yes' if API_KEY else 'No'}")
    print(f"- Test data paths:")
    print(f"  - Normal audio: {NORMAL_AUDIO_DIR}")
    print(f"  - Abnormal audio: {ABNORMAL_AUDIO_DIR}")
    print("\nEndpoint Timeouts:")
    print(f"- Prediction: {DEFAULT_TIMEOUT['predict']}s")
    print(f"- Evaluation: {DEFAULT_TIMEOUT['evaluate']}s")
    print(f"- Other endpoints: {DEFAULT_TIMEOUT['default']}s")
    
    if not NORMAL_AUDIO_DIR.exists() or not ABNORMAL_AUDIO_DIR.exists():
        print("\n⚠️  WARNING: Test data directories not found!")
        print("Please ensure the test data is properly set up before running the tests.")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log when test starts."""
    global test_start_time
    if isinstance(environment.runner, MasterRunner):
        test_start_time = datetime.now()
        print(f"\n🚀 Test started at: {test_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Ramping up simulated users...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print detailed test summary when test finishes."""
    if isinstance(environment.runner, MasterRunner):
        test_duration = datetime.now() - test_start_time if test_start_time else None
        
        print("\n=== Test Summary ===")
        if test_duration:
            print(f"Duration: {test_duration}")
        
        print("\nRequest Statistics:")
        print(f"Total Requests: {success_count + failure_count}")
        success_rate = (success_count / (success_count + failure_count) * 100) if (success_count + failure_count) > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Timeouts: {timeouts}")
        
        print("\nPrediction Statistics:")
        print(f"Total predictions: {total_predictions}")
        if total_predictions > 0:
            normal_pct = (normal_predictions / total_predictions * 100)
            abnormal_pct = (abnormal_predictions / total_predictions * 100)
            print(f"- Normal: {normal_predictions} ({normal_pct:.1f}%)")
            print(f"- Abnormal: {abnormal_predictions} ({abnormal_pct:.1f}%)")
        
        print("\nResponse Times (ms):")
        print(f"- Average: {avg_response_time:.2f}")
        print(f"- Minimum: {min_response_time:.2f}")
        print(f"- Peak: {peak_response_time:.2f}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"load_test_results_{timestamp}.txt"
        try:
            with open(results_file, "w") as f:
                f.write("=== Pump Anomaly Detection Load Test Results ===\n")
                f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if test_duration:
                    f.write(f"Duration: {test_duration}\n")
                f.write(f"Success Rate: {success_rate:.1f}%\n")
                f.write(f"Total Predictions: {total_predictions}\n")
                f.write(f"Average Response Time: {avg_response_time:.2f} ms\n")
            print(f"\nDetailed results saved to: {results_file}")
        except Exception as e:
            print(f"\nWarning: Could not save test results to file: {str(e)}")

class PumpAnomalyUser(HttpUser):
    """Simulated user for load testing the pump anomaly detection API."""
    
    # Wait time between tasks based on the endpoint's typical processing time
    wait_time = between(1, 5)  # Default wait time
    def on_start(self):
        """Setup before starting tests."""
        # Cache available test files
        self.normal_files = list(NORMAL_AUDIO_DIR.glob("*.wav")) if NORMAL_AUDIO_DIR.exists() else []
        self.abnormal_files = list(ABNORMAL_AUDIO_DIR.glob("*.wav")) if ABNORMAL_AUDIO_DIR.exists() else []
        self.last_task = None  # Track last executed task
        
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
    @task(5)  # Higher weight for health checks as they're lightweight
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
            except Exception as e:
                events.request.fire(
                    request_type="GET",
                    name="Health Check",
                    response_time=response.elapsed.total_seconds() * 1000,
                    response_length=len(response.content),
                    response=response,
                    context={},
                    exception=e
                )
    
    @tag("model_info")
    @task(3)  # Medium weight for model info
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
    @task(8)  # Reduced weight for predictions due to their high resource usage
    def predict_random_file(self):
        """Send random audio file for prediction."""
        global total_predictions, normal_predictions, abnormal_predictions, avg_response_time, peak_response_time, min_response_time
        
        # Skip if no files available
        if not self.normal_files and not self.abnormal_files:
            raise RescheduleTask("No audio files available for testing")
        
        # Weighted random selection to ensure 70% normal, 30% abnormal distribution
        use_normal = random.random() < 0.7
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
        start_time = time.time()
        
        with self.client.get(
            "/evaluate/", 
            name="Model Evaluation",
            catch_response=True
        ) as response:
            try:
                if response.status_code == 200:
                    eval_data = response.json()
                    if "classification_report" in eval_data:
                        # Extract key metrics
                        report = eval_data["classification_report"]
                        accuracy = report.get("accuracy", 0)
                        normal_f1 = report.get("0", {}).get("f1-score", 0)
                        abnormal_f1 = report.get("1", {}).get("f1-score", 0)
                        
                        events.request.fire(
                            request_type="GET",
                            name="Model Evaluation",
                            response_time=(time.time() - start_time) * 1000,
                            response_length=len(response.content),
                            response=response,
                            context={
                                "accuracy": accuracy,
                                "normal_f1": normal_f1,
                                "abnormal_f1": abnormal_f1
                            },
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
    print(f"    locust -f locustfile.py --headless -u 10 -r 2 -t 60s -H {PRODUCTION_HOST}")
    print("\nProduction load test (recommended):")
    print(f"    locust -f locustfile.py -H {PRODUCTION_HOST}")
    print("\nLocal development test:")
    print("    locust -f locustfile.py -H http://localhost:8000")
    print("\nCommand explanation:")
    print("    -u 10     : 10 simulated users")
    print("    -r 2      : Spawn 2 users per second")
    print("    -t 60s    : Run for 60 seconds")
    print("    -H [url]  : Host URL of your API (defaults to production if not specified)")
    print("\nNote: When testing production, consider:")
    print("1. Start with fewer users (-u 5) and slower ramp-up (-r 1)")
    print("2. Monitor the API's response times and adjust accordingly")
    print("3. Be mindful of the API's rate limits and resource constraints")
