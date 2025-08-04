# Pump Anomaly Detection Load Testing

This directory contains a Locust load testing script to evaluate the performance and reliability of the Pump Anomaly Detection API under high load.

## Prerequisites

- Locust installed (`pip install locust`)
- Running instance of the Pump Anomaly Detection API
- Test audio files in the data/test directory

## Running the Load Tests

### With Web UI (recommended for interactive testing)

```bash
locust -f locustfile.py -H http://localhost:8000
```

Then open your browser at http://localhost:8089 to access the Locust web interface. From there, you can:
- Set the number of users to simulate
- Set the spawn rate (users per second)
- Start/stop the test
- Monitor real-time statistics
- View charts of response times and requests per second

### Headless Mode (for CI/CD or automated testing)

```bash
locust -f locustfile.py --headless -u 10 -r 2 -t 5m -H http://localhost:8000
```

Parameters:
- `-u 10`: Simulate 10 concurrent users
- `-r 2`: Spawn rate of 2 users per second
- `-t 5m`: Run test for 5 minutes
- `-H http://localhost:8000`: Host URL of your API

## Test Scenarios

The load test includes several simulated user behaviors:

1. **Health Check** (weight: 1)
   - Calls the `/health/` endpoint to verify API availability
   
2. **Model Info** (weight: 1)
   - Calls the `/model-info/` endpoint to get model status

3. **Predictions** (weight: 10)
   - Sends random audio files (normal or abnormal) to the `/predict/` endpoint
   - Verifies if predictions match expected class
   - Tracks prediction statistics and response times

4. **Model Evaluation** (weight: 2)
   - Calls the `/evaluate/` endpoint to test model evaluation performance

## API Authentication

The test automatically uses the API key from your configuration or environment variables. Make sure your API key is properly set before running tests.

## Output Statistics

The test will output various statistics including:
- Request counts and failure rates
- Response time percentiles (min, median, max, 95th percentile)
- Total predictions made
- Breakdown of normal vs. abnormal predictions
- Average, minimum, and peak response times

## Customizing Tests

You can modify the test behavior by:
- Adjusting task weights in the `@task()` decorators
- Changing the wait time between requests (`wait_time = between(min, max)`)
- Adding new API endpoints to test
- Modifying assertion conditions

## Test Specific Endpoints

To test only specific endpoints, use the `--tags` parameter:

```bash
locust -f locustfile.py --tags predict -H http://localhost:8000
```

Available tags:
- `health`: Health check endpoint
- `model_info`: Model information endpoint
- `predict`: Prediction endpoint
- `evaluate`: Model evaluation endpoint
