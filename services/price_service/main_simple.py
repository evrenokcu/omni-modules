import os
import datetime
from dotenv import load_dotenv
from quart import Quart, jsonify, send_from_directory
from quart_cors import cors

# Load environment variables from a .env file if available
load_dotenv()

app = Quart(__name__, static_folder="static")
app = cors(app)

@app.route("/dummy", methods=["GET"])
async def get_dummy():
    dummy_data = {
        "id": 1,
        "name": "Sample Object",
        "description": "This is a dummy JSON object returned by the API."
    }
    return jsonify(dummy_data)

# Route to serve the favicon if available
@app.route("/favicon.ico")
async def favicon():
    return await send_from_directory(app.static_folder, "favicon.ico")

# New endpoint to write a file to the /price volume mount
@app.route("/write", methods=["GET"])
async def write_file():
    volume_path = "/price"
    # Ensure the /price directory exists (it will be mounted in production)
    os.makedirs(volume_path, exist_ok=True)
    
    # Define the file name and content
    filename = os.path.join(volume_path, "dummy.txt")
    content = f"File written at {datetime.datetime.now()}\n"
    
    try:
        # Write content to the file
        with open(filename, "w") as f:
            f.write(content)
        return jsonify({
            "status": "success",
            "filename": filename,
            "content": content
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    import uvicorn
    # Fetch the port from the environment variable, defaulting to 8000 if not defined
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")