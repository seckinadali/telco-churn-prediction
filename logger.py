"""
Logging setup for the telco churn prediction project.
"""

import logging
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).resolve().parent / "logs"
if not logs_dir.exists():
    logs_dir.mkdir(parents=True)

# More detailed logs in file
file_handler = logging.FileHandler(logs_dir / "pipeline.log")
file_handler.setLevel(logging.DEBUG)

# Less verbose in console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[file_handler, console_handler]
)