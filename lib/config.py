# lib/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Directory where cached JSON files will be stored.
CACHE_DIR = os.getenv("CACHE_DIR", "cache")

# A Path object for the cache directory.
CACHE_PATH = Path(CACHE_DIR)

def ensure_cache_directory():
    """Ensure the cache directory exists."""
    CACHE_PATH.mkdir(parents=True, exist_ok=True)