from dataclasses import dataclass
import os
#from dotenv import load_dotenv
from quart import Quart, request, jsonify
#from llama_index.llms.openai import OpenAI
# from llama_index.llms.anthropic import Anthropic
# from llama_index.llms.gemini import Gemini
# from llama_index.llms.groq import Groq
from datetime import datetime
from quart_cors import cors
from pydantic import BaseModel
import time
import asyncio 
from typing import Dict, List
### start of langchain
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
import os
#from langchain_community.callbacks import get_anthropic_callback

from langchain_openai import ChatOpenAI, OpenAI
from langchain_anthropic import Anthropic, AnthropicLLM, ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


#env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./.env"))


#load_dotenv(dotenv_path=env_path)
# print(os.getenv("ANTHROPIC_API_KEY"))
# #llm2 = OpenAI(model_name="gpt-3.5-turbo-instruct")
# ##llm = ChatGoogleGenerativeAI(model='claude-3-opus-20240229')
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")


# with get_openai_callback() as cb:
#     result = llm.invoke("Tell me a joke")
#     print(result)
#     print("---")
# print()

# print(f"Total Tokens: {cb.total_tokens}")
# print(f"Prompt Tokens: {cb.prompt_tokens}")
# print(f"Completion Tokens: {cb.completion_tokens}")
# print(f"Total Cost (USD): ${cb.total_cost}")
# ##end of langchain
# #print(llm.invoke("Tell me a joke"))

# Load environment variables
#load_dotenv()

# Suppress gRPC warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

# Load environment variables.
load_dotenv()

def is_running_in_container() -> bool:
    try:
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                if "docker" in line or "containerd" in line:
                    return True
    except FileNotFoundError:
        return False
    return False

import sys
from os.path import abspath, join, dirname
project_root = abspath(join(dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lib.models import LlmModel, LlmName

# if is_running_in_container():
#         print("The app is running inside a container.")
        
# else:
#         print("The app is not running inside a container.")
#         env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
#         load_dotenv(dotenv_path=env_path)
#         os.environ["PORT"] = "8000"

# Initialize Quart
app = Quart(__name__)
app = cors(app, allow_origin="*")
app.debug = True

class SingleLlmRequest(BaseModel):
    llm: "LlmModel"  # changed from string field llm_name to an Llm object
    prompt: str

class LlmRequest(BaseModel):
    prompt: str

class LlmResult(BaseModel):
    llm: "LlmModel"  # changed from string field llm_name to an Llm object
    response: str

class LlmResponse(BaseModel):
    llm: "LlmModel"  # changed from string field llm_name to an Llm object
    response: str
    timestamp: str
    status: str
    duration: float 
    token_count: int = 0  # Default value for token count
    price: float = 0.0

class LlmResponseList(BaseModel):
    responses: list[LlmResponse]

class LlmResultList(BaseModel):
    responses: list[LlmResult]

# ----------------------------
# New code: Llm object with enum for llm_name and a computed id field
from enum import Enum
from pydantic import computed_field

# class LlmName(str, Enum):
#     CHATGPT = "ChatGPT"
#     CLAUDE = "Claude"
#     GEMINI = "Gemini"

# class LlmModel(BaseModel):
#     llm_name: LlmName
#     model_name: str

#     @computed_field
#     @property
#     def id(self) -> str:
#         return f"{self.llm_name.value}_{self.model_name}"

#     class Config:
#         frozen = True  # if you really need immutability
       
#     def to_dict(self) -> Dict:
#         return {"llm_name": self.llm_name.name, "model_name": self.model_name}

#     @classmethod
#     def from_dict(cls, data: Dict) -> "LlmModel":
#         return cls(llm_name=LlmName[data["llm_name"]], model_name=data["model_name"])
# ----------------------------



def create_chat_client(llm: LlmModel):
    if llm.llm_name == LlmName.OPENAI:
        return ChatOpenAI(model_name=llm.model_name)
    elif llm.llm_name == LlmName.CLAUDE:
        return ChatAnthropic(model=llm.model_name, api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif llm.llm_name == LlmName.GEMINI:
        return ChatGoogleGenerativeAI(model=llm.model_name)
    else:
        raise ValueError(f"Unsupported LLM type: {llm.llm_name}")

# Create a list of Llm objects
llm_list = [
    LlmModel(llm_name=LlmName.OPENAI, model_name="gpt-4"),
    LlmModel(llm_name=LlmName.CLAUDE, model_name="claude-3-5-sonnet-20240620"),
    LlmModel(llm_name=LlmName.GEMINI, model_name="gemini-exp-1121"),
]

# Build the llms dictionary from the list of Llm objects using the factory method.
# The keys are Llm objects and the values are the corresponding chat client instances.
llms = { llm: create_chat_client(llm) for llm in llm_list }

async def process_llm(request: SingleLlmRequest) -> LlmResponse:
    start_time = time.time()
    llm_client = llms[request.llm]
    response = await llm_client.ainvoke(request.prompt)
    token_count = response.usage_metadata.get("total_tokens", 0)  # 0 as a fallback value
    
    response_text = response.content if hasattr(response, 'content') else str(response)
    end_time = time.time()
    return LlmResponse(
        llm=request.llm,
        response=response_text,
        timestamp=datetime.now().isoformat(),
        status="completed",
        duration=end_time - start_time,
        token_count=token_count,  # Default value for token count
        price=0.0
    )

async def process_llm_list(llm_request: LlmRequest, llm_objs: List[LlmModel]) -> LlmResponseList:
    """
    Process the specified LLMs in parallel based on the given LlmRequest and return an LlmResponseList.
    """
    tasks = [
        process_llm(SingleLlmRequest(llm=llm_obj, prompt=llm_request.prompt))
        for llm_obj in llm_objs
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

    # Call process_llm_list with the updated LlmRequest and all LLM objects
    return await process_llm_list(llm_request, list(llms.keys()))

async def process_llm_result_list_on_llm(llm_result_list: LlmResultList, request: SingleLlmRequest) -> LlmResponseList:
    """
    Process the specified LLMs in parallel based on the given LlmRequest and return an LlmResponseList.
    """
    # Generate a new LlmRequest with an updated prompt
    llm_request = generate_prompt_from_result_list(llm_result_list, LlmRequest(prompt=request.prompt))

    # Call process_llm with the updated LlmRequest and the Llm from request
    return await process_llm(SingleLlmRequest(llm=request.llm, prompt=llm_request.prompt))

async def process_summarize(responses: LlmResultList) -> LlmResponseList:
    prompt = os.getenv("MERGE_PROMPT", "Summarize these responses.")
    summarize_request = SingleLlmRequest(
        llm=LlmModel(llm_name=LlmName.GEMINI, model_name="models/gemini-exp-1121"),
        prompt=prompt
    )
    return await process_llm_result_list_on_llm(responses, summarize_request)

async def process_refine(responses: LlmResultList) -> LlmResponseList:
    prompt = os.getenv("EVALUATION_PROMPT")
    llm_response_list = await process_llm_result_list(responses, LlmRequest(prompt=prompt))
    return llm_response_list

@app.post("/refine")
async def refine():
    data = await request.get_json()
    llm_request = LlmResultList(**data)
    llm_response_list = await process_refine(llm_request)

    # Return the LlmResponseList as a JSON response
    return jsonify(llm_response_list.model_dump())

@app.post("/summarize")
async def summarize():
    data = await request.get_json()
    llm_request = LlmResultList(**data)
    llm_response_list = await process_summarize(llm_request)

    # Return the LlmResponseList as a JSON response
    return jsonify(llm_response_list.model_dump())

async def process_aggregate(llm_request: LlmRequest, llm_objs: List[LlmModel]) -> LlmResponseList:
    return await process_llm_list(llm_request, llm_objs)

@app.post("/aggregate")
async def aggregate():
    """
    Query all LLMs in parallel and return a list of responses.
    """
    data = await request.get_json()
    llm_request = LlmRequest(**data)

    # Use the helper method to process the LLMs
    llm_response_list = await process_aggregate(llm_request, list(llms.keys()))

    # Return the LlmResponseList as a JSON response
    return jsonify(llm_response_list.model_dump())

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
        llm_objs = list(llms.keys())
        aggregated_response = await process_aggregate(llm_request, llm_objs)

        # Convert the aggregated response to LlmResultList for refine step
        aggregated_results = LlmResultList(responses=[
            LlmResult(llm=response.llm, response=response.response)
            for response in aggregated_response.responses
        ])

        # Step 2: Refine
        refined_response = await process_refine(aggregated_results)

        # Convert the refined response to LlmResultList for summarize step
        refined_results = LlmResultList(responses=[
            LlmResult(llm=response.llm, response=response.response)
            for response in refined_response.responses
        ])

        # Step 3: Summarize
        summarize_request = SingleLlmRequest(
            llm=LlmModel(llm_name=LlmName.GEMINI, model_name="models/gemini-exp-1121"),
            prompt="Summarize these results."
        )
        summarized_response = await process_llm_result_list_on_llm(refined_results, summarize_request)

        # Return the final summarized response as JSON
        return jsonify(summarized_response.model_dump())

    except Exception as e:
        return jsonify({
            "error": f"Error in flow execution: {str(e)}"
        }), 500

@app.post("/")
async def query_llm():   
    #return await llm_call()
    """
    Query the specified LLM with a given prompt and return the response as a structured object.
    """
    data = await request.get_json()
    try:
        # Validate and parse the incoming JSON using SingleLlmRequest
        llm_request = SingleLlmRequest(**data)

        # Check if the specified LLM is supported
        if llm_request.llm not in llms:
            return jsonify({
                "error": f"LLM '{llm_request.llm}' not supported. Available: {list(llms.keys())}"
            }), 400

        # Use the llm helper function to process the request
        llm_response = await process_llm(llm_request)
        return jsonify(llm_response.model_dump())

    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}"
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
        if llm_request.llm not in llms:
            return jsonify({
                "error": f"LLM '{llm_request.llm}' not supported. Available: {list(llms.keys())}"
            }), 400

        # Use the llm helper function to process the request
        llm_response = await process_llm(llm_request)
        return jsonify(llm_response.model_dump())

    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500

if __name__ == "__main__":
    # Usage
    port = int(os.getenv("PORT", 8080))
    #print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)