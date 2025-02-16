# utils/logger.py
import logging
from config import LOG_FILE

# Set up basic configuration for logging.
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_info(message):
    """Log an informational message."""
    logging.info(message)

def log_error(message):
    """Log an error message."""
    logging.error(message)