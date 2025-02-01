import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import json
def is_running_in_container() -> bool:
    """
    Detects if the script is running inside a Docker container.
    """
    try:
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                if "docker" in line or "containerd" in line:
                    return True
    except FileNotFoundError:
        return False
    return False

if is_running_in_container():
    print("The app is running inside a container.")
else:
    print("The app is not running inside a container.")
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
    load_dotenv(dotenv_path=env_path)
    os.environ["PORT"] = "8000"

def fetch_llm_prices():
    
    # Requires setting environment variables for API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not (openai_key and anthropic_key):
        raise ValueError("API keys for OpenAI and Anthropic must be set")
    
    # Prompt to get pricing information
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in AI model pricing. Provide current pricing details for major LLM providers in JSON format."),
        ("human", "What are the current input and output token prices for Claude 3 Haiku, GPT-3.5 Turbo, GPT-4 Turbo, and other major LLM models?")
    ])
    
    # Use Claude for fetching pricing (most reliable)
    llm = ChatAnthropic(model='claude-3-haiku-20240307', api_key=anthropic_key)
    
    chain = prompt | llm
    response = chain.invoke({})
    
    # Extract JSON from response (you might need to clean the response)
    try:
        prices = json.loads(response.content)
        return json.dumps(prices, indent=2)
    except json.JSONDecodeError:
        # Fallback error handling
        return json.dumps({
            "error": "Could not parse LLM pricing",
            "raw_response": str(response.content)
        })

if __name__ == "__main__":
    print(fetch_llm_prices())