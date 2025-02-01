import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from datetime import datetime
from openai import OpenAI
import requests
import json
import os

# Load environment variables
# env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./.env"))
# load_dotenv(dotenv_path=env_path)

# Suppress gRPC warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

# Initialize Flask
app = Flask(__name__)
app.debug = True

# Initialize LLM clients
# llms = {
#     "ChatGPT": OpenAI(),
#     "Claude": Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
#     "Gemini": Gemini(api_key=os.getenv("GOOGLE_API_KEY")),
# }

@app.get("/hello")
def hello():
    return {"message": "Hello from FastAPI!"}

# @app.post("/echo")
# async def echo(data: EchoData):
#     return {"you_sent": data.dict()}

# @app.route("/", methods=["POST"])
# def query_llm():
#     """
#     Query the specified LLM with a given prompt and return the response as a string.
#     """
#     data = request.json
#     llm_name = data.get("llm_name")
#     prompt = data.get("prompt")

#     # Check if the specified LLM is supported
#     if llm_name not in llms:
#         return jsonify({
#             "error": f"LLM '{llm_name}' not supported. Available: {list(llms.keys())}"
#         }), 400
    
#     # Get the corresponding LLM client
#     llm_client = llms[llm_name]

#     # Query the LLM
#     try:
#         response = llm_client.complete(prompt)
#         response_text = response.text if hasattr(response, 'text') else str(response)
#         return jsonify({
#             "llm": llm_name,
#             "response": response_text,
#             "timestamp": datetime.now().isoformat(),
#             "status": "completed"
#         })
#     except Exception as e:
#         return jsonify({
#             "error": f"Error querying {llm_name}: {str(e)}"
#         }), 500
@app.route("/direct_openai", methods=["POST"])
def call_openai():
    try:
        client = OpenAI()

        # chat_completion = client.chat.completions.create(
        #     messages=[
        #     {
        #         "role": "user",
        #         "content": "Say this is a test",
        #     }
        #   ],model="gpt-4o",)
        
        response = client.chat.completions.create(
             model="gpt-3.5-turbo",  # You can change this to another model if desired
             messages=[
                 {"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": "why is the sky blue?"}
             ],
             max_tokens=150,  # Set the maximum number of tokens in the output
             temperature=0.7  # Adjust the creativity level of the response
         )
        return  response.choices[0].message.content

        #return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route("/test_connectivity", methods=["GET"])
def test_connectivity():
    try:
        response = requests.get("https://www.google.com")
        return {"status": response.status_code, "message": "Internet connectivity is working"}
    except Exception as e:
        return {"error": f"Connectivity issue: {str(e)}"}, 500
    
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
