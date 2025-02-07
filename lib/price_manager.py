    # lib/price_manager.py

import json
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from .models import AggregatedPrice, AggregatedPriceResponse, ModelPrice, PriceResponse
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
{{
    "models": [
        {{
            "model_name": "model_name",
            "pricing": {{
                "input_price": number,
                "output_price": number,
                "currency": "USD"
            }}
        }},
        ...
    ]
}}
Use actual current prices per 1M tokens."""
    ),
    ("human", "Provide the latest input and output token prices for these models: {model_list}")
])


    def fetch_all_prices(self):
        """Fetch the latest prices for all registered LLMs in a single query"""
        models = self.registry.get_all_models()
        # Extract model names from the nested LlmModel instances.
        model_names = [model.model.model_name for model in models]

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
                model_name = model_data["model_name"]
                # Look up using the nested model attribute.
                model_config = next((m for m in models if m.model.model_name == model_name), None)

                if model_config:
                    # Create a ModelPrice instance.
                    model_price = ModelPrice(
                        input_price=model_data["pricing"]["input_price"],
                        output_price=model_data["pricing"]["output_price"],
                        currency=model_data["pricing"]["currency"]
                    )
                    # Create a PriceResponse using the nested LlmModel and new ModelPrice.
                    price_response = PriceResponse(
                        model=model_config.model,
                        pricing=model_price
                    )
                    key = f"{model_config.model.llm_name.name}:{model_config.model.model_name}"
                    cache[key] = price_response.to_dict()

            self.storage.save_prices(cache)
            print("Price registry updated successfully.")

        except Exception as e:
            print(f"Error fetching prices: {str(e)}")


    def get_combined_enabled_prices(self):
        """
        Filters the registry for enabled LLM model configurations,
        retrieves their pricing information from storage, and returns
        an aggregated response entity composed of AggregatedPrice items.
        """
        # Get all registered models.
        all_models = self.registry.get_all_models()
        # Filter only enabled configurations.
        enabled_configs = [cfg for cfg in all_models if cfg.enabled]

        # Load stored price information.
        cache = self.storage.load_prices()
        aggregated_list = []

        # For each enabled configuration, build an AggregatedPrice.
        for cfg in enabled_configs:
            key = f"{cfg.model.id}"
            if key in cache:
                # Create a PriceResponse from the stored cache.
                price_resp = PriceResponse.from_dict(cache[key], cfg.model)
            else:
                price_resp = None

            aggregated_list.append(
                AggregatedPrice(
                    config=cfg,
                    price=price_resp.pricing
                )
            )

        print(f"Aggregated {len(aggregated_list)} enabled model prices.")

        # Return the aggregated response.
        return AggregatedPriceResponse(responses=aggregated_list)