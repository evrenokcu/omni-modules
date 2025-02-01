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
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./.env"))
load_dotenv(dotenv_path=env_path)

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Why is the sky blue?"
        }
    ]
)

print(completion.choices[0].message)