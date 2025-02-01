import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Protocol
from dataclasses import dataclass
from enum import Enum, auto
from dotenv import load_dotenv

# Load environment variables early so they can be used in default parameters.
load_dotenv()
CACHE_DIR = os.getenv("CACHE_DIR", "cache")

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# ===============================
# === Original Logic (Unchanged)
# ===============================

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


# JSON-based Implementation of Price Storage
class JSONPriceStorage:
    def __init__(self, cache_path: Path = Path(CACHE_DIR) / "llm_prices_cache.json"):
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

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


# Main Execution (Original Logic)
def main():
    registry = JSONLLMRegistry()
    storage = JSONPriceStorage()
    price_manager = LLMPriceManager(registry, storage)

    # Fetch and update all model prices in a single request
    price_manager.fetch_all_prices()


# ===============================
# === API Endpoints with Quart
# ===============================

from quart import Quart, request, jsonify
from quart_cors import cors
import asyncio
import threading
import time
import schedule
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

# Initialize Quart and enable CORS
app = Quart(__name__)
app = cors(app, allow_origin="*")

# Global objects for registry, storage, and price manager.
# (They will be initialized in the startup event.)
registry = None
storage = None
price_manager = None

@app.before_serving
async def startup():
    global registry, storage, price_manager
    # Initialize the registry (which will check and create the JSON repo file if needed)
    registry = JSONLLMRegistry()
    storage = JSONPriceStorage()
    price_manager = LLMPriceManager(registry, storage)
    
    # If the prices file is empty, fetch the prices immediately.
    if not storage.load_prices():
        price_manager.fetch_all_prices()
    
    # Schedule price fetch at 01:30 AM daily.
    schedule.every().day.at("01:30").do(price_manager.fetch_all_prices)

    # Run the scheduler in a background thread.
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every 60 seconds.
    threading.Thread(target=run_scheduler, daemon=True).start()


# Endpoint to get all registered LLMs.
@app.route("/llms", methods=["GET"])
async def get_llms():
    models = registry.get_all_models()
    return jsonify([model.to_dict() for model in models])


# Pydantic model for adding a new LLM.
class LLMModelInput(BaseModel):
    provider: str = Field(..., description="LLM provider, e.g. ANTHROPIC or OPENAI")
    api_model_name: str = Field(..., description="The API model name")


# Endpoint to add a new LLM model.
@app.route("/llms", methods=["POST"])
async def add_llm():
    data = await request.get_json()
    try:
        llm_input = LLMModelInput(**data)
        try:
            provider_enum = LLMProvider[llm_input.provider.upper()]
        except KeyError:
            return jsonify({"error": "Invalid provider"}), 400
        new_model = LLMModel(provider=provider_enum, api_model_name=llm_input.api_model_name)
        registry.models.append(new_model)
        registry.persist()
        return jsonify(new_model.to_dict()), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Endpoint to get current prices.
@app.route("/prices", methods=["GET"])
async def get_prices():
    prices = storage.load_prices()
    return jsonify(prices)


# Endpoint to force a refresh of prices.
@app.route("/prices/refresh", methods=["POST"])
async def refresh_prices():
    price_manager.fetch_all_prices()
    prices = storage.load_prices()
    return jsonify(prices)


# SSE endpoint to stream price updates.
@app.route("/prices/stream")
async def stream_prices():
    async def event_generator():
        while True:
            prices = storage.load_prices()
            yield {
                "event": "price_update",
                "data": json.dumps(prices)
            }
            await asyncio.sleep(10)
    return EventSourceResponse(event_generator())


# ===============================
# === Application Entry Point
# ===============================

if __name__ == "__main__":
    # Ensure the cache directory exists.
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure llm_registry.json exists. If not, create an empty JSON array.
    registry_file = cache_dir / "llm_registry.json"
    if not registry_file.exists():
        registry_file.write_text("[]")
    
    # Ensure llm_prices_cache.json exists. If not, create an empty JSON object.
    prices_file = cache_dir / "llm_prices_cache.json"
    if not prices_file.exists():
        prices_file.write_text("{}")
    
    # Always run the API server without expecting any command-line arguments.
    import uvicorn
    port = int(os.getenv("PORT", 8081))
    #print(f"Starting server on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)