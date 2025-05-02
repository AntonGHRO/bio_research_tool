# backend/app/main.py
from fastapi import FastAPI
from pydantic import BaseModel

class Message(BaseModel):
    message: str

app = FastAPI()

@app.get("/hello", response_model=Message)
async def hello():
    return {"message": "Hello from Python!"}
