import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Protocol
from dataclasses import dataclass
from enum import Enum, auto
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


# Define Enum for Providers
class LLMProvider(Enum):
    ANTHROPIC = auto()
    OPENAI = auto()


# Define Data Classes
@dataclass
class LLMModel:
    provider: LLMProvider
    api_model_name: str

    def to_dict(self) -> Dict:
        return {"provider": self.provider.name, "api_model_name": self.api_model_name}

    @classmethod
    def from_dict(cls, data: Dict) -> "LLMModel":
        return cls(provider=LLMProvider[data["provider"]], api_model_name=data["api_model_name"])


@dataclass
class PriceResponse:
    model: LLMModel
    input_price: float
    output_price: float
    currency: str = "USD"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "provider": self.model.provider.name,
            "api_model_name": self.model.api_model_name,
            "pricing": {
                "input_price": self.input_price,
                "output_price": self.output_price,
                "currency": self.currency
            },
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict, model: LLMModel) -> "PriceResponse":
        pricing = data["pricing"]
        return cls(
            model=model,
            input_price=pricing["input_price"],
            output_price=pricing["output_price"],
            currency=pricing["currency"],
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )


# Define Protocol for LLM Model Registry
class LLMRegistryProtocol(Protocol):
    def get_all_models(self) -> List[LLMModel]:
        ...

    def persist(self) -> None:
        ...


# JSON-based Implementation of LLMRegistry
class JSONLLMRegistry(LLMRegistryProtocol):
    DEFAULT_MODELS = [
        LLMModel(provider=LLMProvider.ANTHROPIC, api_model_name="claude-3-haiku-20240307"),
        LLMModel(provider=LLMProvider.ANTHROPIC, api_model_name="claude-3-sonnet-20240307"),
        LLMModel(provider=LLMProvider.ANTHROPIC, api_model_name="claude-3-opus-20240307"),
        LLMModel(provider=LLMProvider.OPENAI, api_model_name="gpt-3.5-turbo"),
        LLMModel(provider=LLMProvider.OPENAI, api_model_name="gpt-4-turbo-preview"),
    ]

    def __init__(self, json_path: Path = Path("cache/llm_registry.json")):
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
        with open(self.json_path, "w") as f:
            json.dump([model.to_dict() for model in self.models], f, indent=2)

    def initialize(self):
        if not self.models:
            print("Initializing model registry with default models...")
            self.models = self.DEFAULT_MODELS.copy()
            self.persist()

    def get_all_models(self) -> List[LLMModel]:
        return self.models


# JSON-based Implementation of Price Storage
class JSONPriceStorage:
    def __init__(self, cache_path: Path = Path("cache/llm_prices_cache.json")):
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(exist_ok=True)

    def load_prices(self) -> Dict[str, Dict]:
        try:
            with open(self.cache_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_prices(self, prices: Dict[str, Dict]):
        with open(self.cache_path, "w") as f:
            json.dump(prices, f, indent=2)


# Price Manager Class
class LLMPriceManager:
    def __init__(self, registry: LLMRegistryProtocol, storage: JSONPriceStorage):
        self.registry = registry
        self.storage = storage
        self._load_environment()
        self._initialize_llm_clients()

    def _load_environment(self):
        load_dotenv()
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.anthropic_key:
            raise ValueError("Anthropic API key must be set in environment variables")

    def _initialize_llm_clients(self):
        self.price_fetcher = ChatAnthropic(
            model="claude-3-haiku-20240307",
            api_key=self.anthropic_key,
            temperature=0
        )
        self.prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an AI pricing expert. Return only a JSON object with the exact structure:
{{
    "models": [
        {{
            "api_model_name": "model_name",
            "pricing": {{
                "input_price": number,
                "output_price": number,
                "currency": "USD"
            }}
        }},
        ...
    ]
}}
Use actual current prices per 1K tokens."""
    ),
    ("human", "Provide the latest input and output token prices for these models: {model_list}")
])

    def fetch_all_prices(self):
        """Fetch the latest prices for all registered LLMs in a single query"""
        models = self.registry.get_all_models()
        model_names = [model.api_model_name for model in models]

        if not model_names:
            print("No models found in the registry.")
            return

        print("\nFetching LLM pricing for all models in one request...")

        try:
            chain = self.prompt | self.price_fetcher
            response = chain.invoke({"model_list": ", ".join(model_names)})
            data = json.loads(response.content)

            if "models" not in data:
                raise ValueError("Invalid response format: missing 'models' field")

            cache = self.storage.load_prices()

            for model_data in data["models"]:
                model_name = model_data["api_model_name"]
                model = next((m for m in models if m.api_model_name == model_name), None)

                if model:
                    price_response = PriceResponse(
                        model=model,
                        input_price=model_data["pricing"]["input_price"],
                        output_price=model_data["pricing"]["output_price"],
                        currency=model_data["pricing"]["currency"]
                    )
                    cache[f"{model.provider.name}:{model.api_model_name}"] = price_response.to_dict()
            
            self.storage.save_prices(cache)
            print("Price registry updated successfully.")

        except Exception as e:
            print(f"Error fetching prices: {str(e)}")


# Main Execution
def main():
    registry = JSONLLMRegistry()
    storage = JSONPriceStorage()
    price_manager = LLMPriceManager(registry, storage)

    # Fetch and update all model prices in a single request
    price_manager.fetch_all_prices()


if __name__ == "__main__":
    main()