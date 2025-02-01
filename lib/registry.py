# lib/registry.py

import json
from pathlib import Path
from typing import List, Protocol

from .models import LLMModel, LLMProvider
from .config import CACHE_DIR

# Define a Protocol for the LLM Registry.
class LLMRegistryProtocol(Protocol):
    def get_all_models(self) -> List[LLMModel]:
        ...

    def persist(self) -> None:
        ...

# JSON-based implementation of the LLM Registry.
class JSONLLMRegistry(LLMRegistryProtocol):
    DEFAULT_MODELS = [
        LLMModel(provider=LLMProvider.ANTHROPIC, api_model_name="claude-3-haiku-20240307"),
        LLMModel(provider=LLMProvider.ANTHROPIC, api_model_name="claude-3-sonnet-20240307"),
        LLMModel(provider=LLMProvider.ANTHROPIC, api_model_name="claude-3-opus-20240307"),
        LLMModel(provider=LLMProvider.OPENAI, api_model_name="gpt-3.5-turbo"),
        LLMModel(provider=LLMProvider.OPENAI, api_model_name="gpt-4-turbo-preview"),
    ]

    def __init__(self, json_path: Path = Path(CACHE_DIR) / "llm_registry.json"):
        self.json_path = json_path
        self.models = self._load_models()
        self.initialize()

    def _load_models(self) -> List[LLMModel]:
        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)
                return [LLMModel.from_dict(model) for model in data]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def persist(self):
        # Ensure the parent directory exists.
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, "w") as f:
            json.dump([model.to_dict() for model in self.models], f, indent=2)

    def initialize(self):
        if not self.models:
            print("Initializing model registry with default models...")
            self.models = self.DEFAULT_MODELS.copy()
            self.persist()

    def get_all_models(self) -> List[LLMModel]:
        return self.models