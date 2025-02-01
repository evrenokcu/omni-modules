# lib/models.py

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Dict

# Define Enum for Providers
class LLMProvider(Enum):
    ANTHROPIC = auto()
    OPENAI = auto()

# Data class for an LLM model.
@dataclass
class LLMModel:
    provider: LLMProvider
    api_model_name: str

    def to_dict(self) -> Dict:
        return {"provider": self.provider.name, "api_model_name": self.api_model_name}

    @classmethod
    def from_dict(cls, data: Dict) -> "LLMModel":
        return cls(provider=LLMProvider[data["provider"]], api_model_name=data["api_model_name"])

# Data class for storing pricing response.
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