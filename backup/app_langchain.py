import os
from dotenv import load_dotenv
from quart import Quart, request, jsonify
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from datetime import datetime
from quart_cors import cors
from pydantic import BaseModel
import time
import asyncio 
from typing import List

# Load environment variables
#load_dotenv()

# Suppress gRPC warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

def is_running_in_container() -> bool:
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

# Initialize Quart
app = Quart(__name__)
app = cors(app, allow_origin="*")
app.debug = True

class SingleLlmRequest(BaseModel):
    llm_name: str
    prompt: str
class LlmRequest(BaseModel):
    prompt: str
class LlmResult(BaseModel):
    llm_name: str
    response: str

class LlmResponse(BaseModel):
    llm_name: str
    response: str
    timestamp: str
    status: str
    duration: float = None

class LlmResponseList(BaseModel):
    responses: list[LlmResponse]

class LlmResultList(BaseModel):
    responses: list[LlmResult]

# Initialize LLM clients
llms = {
    "ChatGPT": OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    # "Claude": Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
    "Claude": Gemini(api_key=os.getenv("GOOGLE_API_KEY")),
    "Gemini": Gemini(api_key=os.getenv("GOOGLE_API_KEY")),
    # "Groq": Groq(model="llama3-70b-8192", api_key=os.getenv("xai-fwJaNgak7lu7IZOZVrlTeqtn8WtJ2zV47VLvoK6fedx3b6VZnu9vUCKzz3i3zRisqCN0W2yTtzFtHfrb")),
}

async def process_llm(request: SingleLlmRequest) -> LlmResponse:
    start_time = time.time()
    llm_client = llms[request.llm_name]
    response = await llm_client.acomplete(request.prompt)
    response_text = response.text if hasattr(response, 'text') else str(response)
    end_time = time.time()
    return LlmResponse(
        llm_name=request.llm_name,
        response=response_text,
        timestamp=datetime.now().isoformat(),
        status="completed",
        duration=end_time - start_time
    )
async def process_llm_list(llm_request: LlmRequest, llm_names: List[str]) -> LlmResponseList:
    """
    Process the specified LLMs in parallel based on the given LlmRequest and return an LlmResponseList.
    """
    tasks = [
        process_llm(SingleLlmRequest(llm_name=llm_name, prompt=llm_request.prompt))
        for llm_name in llm_names
    ]
    results = await asyncio.gather(*tasks)
    return LlmResponseList(responses=results)

def generate_prompt_from_result_list(llm_result_list: LlmResultList, llm_request: LlmRequest) -> LlmRequest:
    """
    Generate a combined string from the responses in LlmResultList and 
    create a new LlmRequest with the updated prompt.
    """
    combined_responses = "\n\n".join(result.response for result in llm_result_list.responses)
    updated_prompt = f"{llm_request.prompt}\n\n{combined_responses}"
    return LlmRequest(prompt=updated_prompt)

async def process_llm_result_list(llm_result_list: LlmResultList, request: LlmRequest) -> LlmResponseList:
    """
    Process the specified LLMs in parallel based on the given LlmRequest and return an LlmResponseList.
    """
    # Generate a new LlmRequest with an updated prompt
    llm_request = generate_prompt_from_result_list(llm_result_list, request)

    # Call process_llm_list with the updated LlmRequest and all LLM names
    return await process_llm_list(llm_request, list(llms.keys()))

async def process_llm_result_list_on_llm(llm_result_list: LlmResultList, request: SingleLlmRequest) -> LlmResponseList:
    """
    Process the specified LLMs in parallel based on the given LlmRequest and return an LlmResponseList.
    """
    # Generate a new LlmRequest with an updated prompt
    llm_request = generate_prompt_from_result_list(llm_result_list, LlmRequest(prompt=request.prompt))

    # Call process_llm_list with the updated LlmRequest and all LLM names
    return await process_llm(SingleLlmRequest(llm_name= request.llm_name,prompt=llm_request.prompt))

async def process_summarize(responses: LlmResultList) -> LlmResponseList:
    prompt = os.getenv("MERGE_PROMPT", "Summarize these responses.")
    summarize_request = SingleLlmRequest(llm_name="ChatGPT", prompt=prompt)
    return await process_llm_result_list_on_llm(responses, summarize_request)
async def process_refine(responses:LlmResultList)->LlmResponseList:
    prompt=os.getenv("EVALUATION_PROMPT")
    llm_response_list = await process_llm_result_list(responses, LlmRequest(prompt=prompt))
    return llm_response_list

@app.post("/refine")
async def refine():
    data = await request.get_json()
    llm_request = LlmResultList(**data)
    llm_response_list = await process_refine(llm_request)

    # Return the LlmResponseList as a JSON response
    return jsonify(llm_response_list.dict())

@app.post("/summarize")
async def summarize():
    data = await request.get_json()
    llm_request = LlmResultList(**data)
    llm_response_list = await process_summarize(llm_request)

    # Return the LlmResponseList as a JSON response
    return jsonify(llm_response_list.dict())

async def process_aggregate(llm_request:LlmRequest, llms:List[str])->LlmResponseList:
    return await process_llm_list(llm_request, llms)

@app.post("/aggregate")
async def aggregate():
    """
    Query all LLMs in parallel and return a list of responses.
    """
    data = await request.get_json()
    llm_request = LlmRequest(**data)

    # Use the helper method to process the LLMs
    llm_response_list = await process_aggregate(llm_request, llms.keys())

    # Return the LlmResponseList as a JSON response
    return jsonify(llm_response_list.dict())

@app.post("/flow")
async def flow():
    """
    Execute a flow of aggregate -> refine -> summarize and return the final result as JSON.
    """
    try:
        # Parse the input data into an LlmRequest
        data = await request.get_json()
        llm_request = LlmRequest(**data)

        # Step 1: Aggregate
        llm_names = list(llms.keys())
        aggregated_response = await process_aggregate(llm_request, llm_names)

        # Convert the aggregated response to LlmResultList for refine step
        aggregated_results = LlmResultList(responses=[
            LlmResult(llm_name=response.llm_name, response=response.response)
            for response in aggregated_response.responses
        ])

        # Step 2: Refine
        refined_response = await process_refine(aggregated_results)

        # Convert the refined response to LlmResultList for summarize step
        refined_results = LlmResultList(responses=[
            LlmResult(llm_name=response.llm_name, response=response.response)
            for response in refined_response.responses
        ])

        # Step 3: Summarize
        summarize_request = SingleLlmRequest(llm_name="ChatGPT", prompt="Summarize these results.")
        summarized_response = await process_llm_result_list_on_llm(refined_results, summarize_request)

        # Return the final summarized response as JSON
        return jsonify(summarized_response.dict())

    except Exception as e:
        return jsonify({
            "error": f"Error in flow execution: {str(e)}"
        }), 500

@app.post("/")
async def query_llm():
    """
    Query the specified LLM with a given prompt and return the response as a string.
    """
    data = await request.json
    llm_name = data.get("llm_name")
    prompt = data.get("prompt")

    # Check if the specified LLM is supported
    if llm_name not in llms:
        return jsonify({
            "error": f"LLM '{llm_name}' not supported. Available: {list(llms.keys())}"
        }), 400

    # Get the corresponding LLM client
    llm_client = llms[llm_name]

    # Query the LLM asynchronously
    try:
        response = await llm_client.acomplete(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)
        return jsonify({
            "llm": llm_name,
            "response": response_text,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
    except Exception as e:
        return jsonify({
            "error": f"Error querying {llm_name}: {str(e)}"
        }), 500
    
@app.post("/llm")
async def llm_call():
    """
    Query the specified LLM with a given prompt and return the response as a structured object.
    """
    data = await request.get_json()
    try:
        # Validate and parse the incoming JSON using SingleLlmRequest
        llm_request = SingleLlmRequest(**data)

        # Check if the specified LLM is supported
        if llm_request.llm_name not in llms:
            return jsonify({
                "error": f"LLM '{llm_request.llm_name}' not supported. Available: {list(llms.keys())}"
            }), 400

        # Use the llm helper function to process the request
        llm_response = await process_llm(llm_request)
        return jsonify(llm_response.dict())

    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500

if __name__ == "__main__":
    # Usage
    

    port = int(os.getenv("PORT", 8080))
    #print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)