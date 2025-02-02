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

# Create Quart app instance
app = Quart(__name__)
app = cors(app)


@app.route("/llms", methods=["GET"])
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

# class LLMModelInput(BaseModel):
#     model: LlmModel = Field(..., description="A nested object containing 'llm_name' and 'model_name'")
#     enabled: bool = Field(..., description="Whether the model should be enabled")
#     color: str = Field(..., description="Hex color code for the model (e.g. '#FFFFFF')")
#     initial_char: str = Field(..., description="A single character representing the model (e.g. 'A')")

#



if __name__ == "__main__":
    # Ensure that the required JSON files exist.
    # //registry_file = Path(CACHE_DIR) / "llm_registry.json"
    # if not registry_file.exists():
    #     registry_file.write_text("[]")
    
    # prices_file = Path(CACHE_DIR) / "llm_prices_cache.json"
    # if not prices_file.exists():
    #     prices_file.write_text("{}")
    
    # Start the API server with uvicorn.
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)