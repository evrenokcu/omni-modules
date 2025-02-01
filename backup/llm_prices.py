import os
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Optional, Union, List
from dataclasses import dataclass
from enum import Enum, auto
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# Define enums and data classes first
class LLMProvider(Enum):
    ANTHROPIC = auto()
    OPENAI = auto()

@dataclass
class LLMModel:
    name: str
    provider: LLMProvider
    api_model_name: str
    description: str = ""
    
    def __str__(self) -> str:
        return f"{self.name} ({self.provider.name})"

@dataclass
class PriceResponse:
    """Standardized price response structure"""
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
            "model_name": self.model.name,
            "provider": self.model.provider.name,
            "pricing": {
                "input_price": self.input_price,
                "output_price": self.output_price,
                "currency": self.currency
            },
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict, model: LLMModel) -> 'PriceResponse':
        pricing = data["pricing"]
        return cls(
            model=model,
            input_price=pricing["input_price"],
            output_price=pricing["output_price"],
            currency=pricing["currency"],
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )

class LLMRegistry:
    """Registry of supported LLM models"""
    
    MODELS = [
        LLMModel(
            name="Claude 3 Haiku",
            provider=LLMProvider.ANTHROPIC,
            api_model_name="claude-3-haiku-20240307",
            description="Fast and efficient model for everyday tasks"
        ),
        LLMModel(
            name="Claude 3 Sonnet",
            provider=LLMProvider.ANTHROPIC,
            api_model_name="claude-3-sonnet-20240307",
            description="Balanced model for complex tasks"
        ),
        LLMModel(
            name="Claude 3 Opus",
            provider=LLMProvider.ANTHROPIC,
            api_model_name="claude-3-opus-20240307",
            description="Most capable model for advanced tasks"
        ),
        LLMModel(
            name="GPT-3.5 Turbo",
            provider=LLMProvider.OPENAI,
            api_model_name="gpt-3.5-turbo",
            description="Efficient model for general tasks"
        ),
        LLMModel(
            name="GPT-4 Turbo",
            provider=LLMProvider.OPENAI,
            api_model_name="gpt-4-turbo-preview",
            description="Advanced model for complex reasoning"
        )
    ]
    
    @classmethod
    def get_model(cls, name: str) -> Optional[LLMModel]:
        """Get model by name"""
        return next((model for model in cls.MODELS if model.name.lower() == name.lower()), None)
    
    @classmethod
    def get_all_models(cls) -> List[LLMModel]:
        """Get all registered models"""
        return cls.MODELS

class LLMPriceManager:
    def __init__(self, fetch_immediately: bool = True):
        self.fetch_immediately = fetch_immediately
        self._load_environment()
        self._setup_cache_directory()
        self._initialize_llm_clients()
        
    def _load_environment(self):
        """Load environment variables from .env file"""
        env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
        load_dotenv(dotenv_path=env_path)
        
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not (self.openai_key and self.anthropic_key):
            raise ValueError("API keys for OpenAI and Anthropic must be set in environment variables")

    def _setup_cache_directory(self):
        """Create cache directory if it doesn't exist"""
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.price_cache_path = self.cache_dir / "llm_prices_cache.json"

    def _initialize_llm_clients(self):
        """Initialize LLM clients for price fetching"""
        # Set temperature to 0 to ensure exact JSON responses
        self.price_fetcher = ChatAnthropic(
            model="claude-3-haiku-20240307",
            api_key=self.anthropic_key,
            temperature=0
        )
        
        # Fixed prompt template with properly escaped JSON structure
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in AI model pricing. Return only a JSON object with this exact structure:
{
    "pricing": {
        "input_price": number,
        "output_price": number,
        "currency": "USD"
    }
}
Use actual current prices per 1K tokens."""),
            ("human", "What are the current input and output token prices for {model_name}?")
        ])


    def _parse_llm_response(self, response: str, model: LLMModel) -> PriceResponse:
        """Parse LLM response into structured format"""
        try:
            # Extract JSON from response
            json_str = response.split("```json")[-1].split("```")[0] if "```" in response else response
            json_str = json_str.strip()
            
            # Additional cleanup
            json_str = json_str.replace('\n', '').replace('\\n', '')
            
            data = json.loads(json_str)
            
            # Validate required fields
            if "pricing" not in data:
                raise ValueError("Missing 'pricing' field in response")
                
            pricing = data["pricing"]
            required_fields = ["input_price", "output_price", "currency"]
            if not all(field in pricing for field in required_fields):
                raise ValueError(f"Missing required pricing fields: {required_fields}")
            
            return PriceResponse(
                model=model,
                input_price=float(pricing["input_price"]),
                output_price=float(pricing["output_price"]),
                currency=pricing["currency"]
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Invalid price response format: {str(e)}\nResponse: {response}")

    def _fetch_price_from_llm(self, model: LLMModel) -> PriceResponse:
        """Fetch price information for a specific model from LLM"""
        try:
            chain = self.prompt | self.price_fetcher
            response = chain.invoke({"model_name": model.name})
            return self._parse_llm_response(response.content, model)
        except Exception as e:
            raise ValueError(f"Error fetching price for {model.name}: {str(e)}")

    def _load_cache(self) -> Dict:
        """Load prices from cache file"""
        try:
            with open(self.price_cache_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_cache(self, prices: Dict):
        """Save prices to cache file"""
        with open(self.price_cache_path, 'w') as f:
            json.dump(prices, f, indent=2)

    def get_model_price(self, model_name: str) -> PriceResponse:
        """Get price information for a specific model"""
        model = LLMRegistry.get_model(model_name)
        if not model:
            raise ValueError(f"Unknown model: {model_name}")
        
        cache = self._load_cache()
        
        # Check if we need to fetch new prices
        if (self.fetch_immediately or 
            model.name not in cache or 
            (datetime.now() - datetime.fromisoformat(cache[model.name]["timestamp"])).days >= 1):
            
            try:
                price_response = self._fetch_price_from_llm(model)
                cache[model.name] = price_response.to_dict()
                self._save_cache(cache)
                return price_response
            except Exception as e:
                if model.name in cache:
                    # Return cached data if available, even if expired
                    return PriceResponse.from_dict(cache[model.name], model)
                raise ValueError(f"Could not fetch prices for {model.name}: {str(e)}")
        
        return PriceResponse.from_dict(cache[model.name], model)

    def fetch_all_prices(self) -> Dict[str, PriceResponse]:
        """Fetch prices for all registered models"""
        results = {}
        for model in LLMRegistry.get_all_models():
            try:
                results[model.name] = self.get_model_price(model.name)
            except Exception as e:
                print(f"Error fetching prices for {model.name}: {str(e)}")
        return results

def is_running_in_container() -> bool:
    """Detect if the script is running inside a Docker container."""
    try:
        with open("/proc/self/cgroup", "r") as f:
            return any("docker" in line or "containerd" in line for line in f)
    except FileNotFoundError:
        return False

def main():
    container_status = "inside" if is_running_in_container() else "outside"
    print(f"The app is running {container_status} a container.")

    try:
        price_manager = LLMPriceManager(fetch_immediately=True)
        
        # Get price for a specific model with better error handling
        print("\nFetching GPT-4 Turbo pricing...")
        try:
            gpt4_price = price_manager.get_model_price("GPT-4 Turbo")
            print("GPT-4 Turbo Pricing:")
            print(json.dumps(gpt4_price.to_dict(), indent=2))
        except Exception as e:
            print(f"Error fetching GPT-4 Turbo pricing: {str(e)}")
        
        # Fetch all prices with better error handling
        print("\nFetching all model prices...")
        all_prices = price_manager.fetch_all_prices()
        
        if all_prices:
            print("\nSuccessfully retrieved prices:")
            for model_name, price_response in all_prices.items():
                print(f"\n{model_name}:")
                print(json.dumps(price_response.to_dict(), indent=2))
        else:
            print("\nNo prices were retrieved successfully.")
            
    except Exception as e:
        print(f"Critical error in price manager: {str(e)}")
        raise

if __name__ == "__main__":
    main()