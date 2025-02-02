# services/price_service/app.py

import os
import json
import time
import threading
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from quart import Quart, request, jsonify
from quart_cors import cors
import schedule
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

# Load environment variables.
load_dotenv()

# Ensure CACHE_DIR exists.
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Make sure the project root is in sys.path so we can import from lib.
import sys
from os.path import abspath, join, dirname
project_root = abspath(join(dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our library modules.
from lib.registry import JSONLLMRegistry
from lib.storage import JSONPriceStorage
from lib.price_manager import LLMPriceManager
from lib.models import LlmModel, LlmName, LlmModelConfig

# Initialize Quart and enable CORS.
app = Quart(__name__)
app = cors(app, allow_origin="*")

# Global objects for the registry, storage, and price manager.
registry = None
storage = None
price_manager = None

@app.before_serving
async def startup():
    global registry, storage, price_manager
    # Initialize the registry (this creates the JSON file if needed)
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

# class LLMModelInput(BaseModel):
#     model: LlmModel = Field(..., description="A nested object containing 'llm_name' and 'model_name'")
#     enabled: bool = Field(..., description="Whether the model should be enabled")
#     color: str = Field(..., description="Hex color code for the model (e.g. '#FFFFFF')")
#     initial_char: str = Field(..., description="A single character representing the model (e.g. 'A')")

# # Endpoint to add a new LLM model.
# @app.route("/llms", methods=["POST"])
# async def add_llm():
#     data = await request.get_json()
#     try:
#         llm_input = LLMModelInput(**data)
#         new_model = LlmModelConfig(
#             model=llm_input.model,
#             enabled=llm_input.enabled,
#             color=llm_input.color,
#             initial_char=llm_input.initial_char
#         )
#         registry.models.append(new_model)
#         registry.persist()
#         return jsonify(new_model.to_dict()), 201
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

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
# In services/price_service/app.py

@app.route("/hello", methods=["GET"])
async def get_llms():
    # Return hardcoded models
    hardcoded_models = [
        {
            "llm_name": "ChatGPT",
            "model_name": "gpt-3.5-turbo",
            "id": "ChatGPT:gpt-3.5-turbo"
        },
        {
            "llm_name": "Claude",
            "model_name": "claude-3",
            "id": "Claude:claude-3"
        }
    ]
    return jsonify(hardcoded_models)

@app.route("/prices/aggregated", methods=["GET"])
async def get_aggregated_prices():
    try:
        aggregated = price_manager.get_combined_enabled_prices()
        # Using model_dump() in Pydantic v2:
        return aggregated.model_dump_json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure that the required JSON files exist.
    registry_file = Path(CACHE_DIR) / "llm_registry.json"
    if not registry_file.exists():
        registry_file.write_text("[]")
    
    prices_file = Path(CACHE_DIR) / "llm_prices_cache.json"
    if not prices_file.exists():
        prices_file.write_text("{}")
    
    # Start the API server with uvicorn.
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)