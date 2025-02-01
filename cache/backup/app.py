import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Pydantic model (optional) for the POST request data
class EchoData(BaseModel):
    message: str

@app.get("/hello")
def hello():
    return {"message": "Hello from FastAPI!"}

@app.post("/echo")
async def echo(data: EchoData):
    return {"you_sent": data.dict()}

if __name__ == "__main__":
    # Cloud Run sets the PORT environment variable; default to 8080 if not set
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
