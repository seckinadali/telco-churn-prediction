"""
Logging setup for the telco churn prediction project.
"""

import logging
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).resolve().parent / "logs"
if not logs_dir.exists():
    logs_dir.mkdir(parents=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(logs_dir / "pipeline.log"),
        logging.StreamHandler()
    ]
)