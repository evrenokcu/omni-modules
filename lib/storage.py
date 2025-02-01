# lib/storage.py

import json
from pathlib import Path
from .config import CACHE_DIR

class JSONPriceStorage:
    def __init__(self, cache_path: Path = Path(CACHE_DIR) / "llm_prices_cache.json"):
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def load_prices(self) -> dict:
        try:
            with open(self.cache_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_prices(self, prices: dict):
        with open(self.cache_path, "w") as f:
            json.dump(prices, f, indent=2)