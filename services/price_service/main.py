import os
from dotenv import load_dotenv
from quart import Quart, jsonify
from quart_cors import cors

# Load environment variables from a .env file if available
load_dotenv()

app = Quart(__name__)
app = cors(app)  # Enable CORS on all routes

@app.route("/dummy", methods=["GET"])
async def get_dummy():
    dummy_data = {
        "id": 1,
        "name": "Sample Object",
        "description": "This is a dummy JSON object returned by the API."
    }
    return jsonify(dummy_data)

if __name__ == "__main__":
    import uvicorn
    # Fetch the port from the environment variable, defaulting to 8000 if not defined
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")