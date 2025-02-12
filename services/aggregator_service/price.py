from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import google.generativeai as genai
import os

# Set up your Google API key
# //os.getenv("ANTHROPIC_API_KEY")?>os.environ["GOOGLE_API_KEY"] = "your-api-key-here"  # Replace with your actual API key

# Initialize Gemini Pro
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

def get_llm_prices(query: str) -> str:
    """Fetches LLM prices from the internet using Gemini Pro."""
    try:
        # Gemini Pro is capable of accessing the internet directly, so a simple query is often enough
        return llm.invoke(query).content
    except Exception as e:
        return f"Error fetching LLM prices: {e}"

# Define tools for the agent. In this case, we only need one.
tools = [
    Tool(
        name="LLMPrices",
        func=get_llm_prices,
        description="Useful for getting the latest LLM pricing information. Input should be a search query related to LLM prices.",
    )
]

# Initialize the agent. We use the ZERO_SHOT_REACT_DESCRIPTION agent type which is suitable for this task.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # Set to True for detailed output of the agent's reasoning
)

# Example usage:
try:
    result = agent.run("What are the current pricing options for Gemini Pro and other competitive LLMs?")
    print(result)
    
    result = agent.run("How much does it cost to use GPT-4 for fine-tuning?")
    print(result)
    
    result = agent.run("Compare the pricing of Claude Instant and GPT-3.5-turbo.")
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")