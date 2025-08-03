import logging
from logging.handlers import RotatingFileHandler
import os
from . import config

def setup_logger(name):
    """Set up logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(config.LOG_LEVEL))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler
    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Create logger instances
api_logger = setup_logger('pump_anomaly.api')
model_logger = setup_logger('pump_anomaly.model')
dashboard_logger = setup_logger('pump_anomaly.dashboard')
