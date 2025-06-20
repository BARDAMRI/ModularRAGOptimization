import logging

# Configure logger
logger = logging.getLogger("ModularRAGOptimization")
logger.setLevel(logging.INFO)

# Stream handler for console output
stream_handler = logging.StreamHandler()
stream_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(stream_formatter)

# File handler for logging to a file
file_handler = logging.FileHandler("logger.log")
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
