# lib/registry.py

import json
from pathlib import Path
from typing import List, Protocol

from .models import LlmModel, LlmModelConfig, LlmName
from .config import CACHE_DIR

# Define a Protocol for the LLM Registry.
class LLMRegistryProtocol(Protocol):
    def get_all_models(self) -> List[LlmModelConfig]:
        ...

    def persist(self) -> None:
        ...
 
    # "ChatGPT": ChatOpenAI(model_name="gpt-4"),
    # #"ChatGPT": ChatGoogleGenerativeAI(model="models/gemini-1.5-flash"),
    #  "Claude": ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=os.getenv("ANTHROPIC_API_KEY")),
    # #"Claude": ChatGoogleGenerativeAI(model="models/gemini-1.5-flash"),
    # "Gemini": ChatGoogleGenerativeAI(model="models/gemini-exp-1121"),

# JSON-based implementation of the LLM Registry.
class JSONLLMRegistry(LLMRegistryProtocol):
    # Generate default LlmModelConfig instances.
    DEFAULT_MODEL_CONFIGS: List[LlmModelConfig] = [
        # CLAUDE models: one enabled, one disabled.
        LlmModelConfig(
            model=LlmModel(
                llm_name=LlmName.Claude,
                model_name="claude-3-haiku-20240307"
            ),
            enabled=False,
            color="#00FF00",  # lime green
            initial_char="C"
        ),
        LlmModelConfig(
            model=LlmModel(
                llm_name=LlmName.Claude,
                model_name="claude-3-5-sonnet-20240620"
            ),
            enabled=True,
            color="#008000",  # dark green
            initial_char="C"
        ),
        # OPENAI models: one enabled, one disabled.
        LlmModelConfig(
            model=LlmModel(
                llm_name=LlmName.ChatGPT,
                model_name="gpt-3.5-turbo"
            ),
            enabled=True,
            color="#0000FF",  # blue
            initial_char="O"
        ),
        LlmModelConfig(
            model=LlmModel(
                llm_name=LlmName.ChatGPT,
                model_name="gpt-4-turbo-preview"
            ),
            enabled=False,
            color="#000080",  # navy
            initial_char="O"
        ),
        LlmModelConfig(
            model=LlmModel(
                llm_name=LlmName.Gemini,
                model_name="gemini-2.0-flash-exp"
            ),
            enabled=True,
            color="#000090",  # navy
            initial_char="O"
        ),
    ]

    def __init__(self, json_path: Path = Path(CACHE_DIR) / "llm_registry.json"):
        self.json_path = json_path
        self.models = self._load_models()
        self.initialize()

    def _load_models(self) -> List[LlmModelConfig]:
        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)
                return [LlmModelConfig.from_dict(model) for model in data]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def persist(self) -> None:
        # Ensure the parent directory exists.
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, "w") as f:
            json.dump([model.to_dict() for model in self.models], f, indent=2)

    def initialize(self) -> None:
        if not self.models:
            print("Initializing model registry with default models...")
            self.models = self.DEFAULT_MODEL_CONFIGS.copy()
            self.persist()

    def get_all_models(self) -> List[LlmModelConfig]:
        return self.models