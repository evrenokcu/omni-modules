    # lib/price_manager.py

import json
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from .models import PriceResponse
from .storage import JSONPriceStorage

class LLMPriceManager:
    def __init__(self, registry, storage: JSONPriceStorage):
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
{
    "models": [
        {
            "api_model_name": "model_name",
            "pricing": {
                "input_price": number,
                "output_price": number,
                "currency": "USD"
            }
        },
        ...
    ]
}
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