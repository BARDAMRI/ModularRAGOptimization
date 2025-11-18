import logging
import os
from datetime import datetime

# Get absolute root path of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create logs directory if it doesn't exist
logs_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Generate log file name based on current datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"{timestamp}.log"
log_file_path = os.path.join(logs_dir, log_filename)

# Configure logger
logger = logging.getLogger("ModularRAGOptimization")
logger.setLevel(logging.INFO)

# File handler for logging to a file
file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)

# (Optional) Stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(file_formatter)
logger.addHandler(stream_handler)

logger.addHandler(file_handler)
