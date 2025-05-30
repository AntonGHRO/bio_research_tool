# backend/app/main.py
from fastapi import FastAPI
from pydantic import BaseModel

from backend.app.graph import get_subgraph


class Message(BaseModel):
    message: str

app = FastAPI()

@app.get("/hello", response_model=Message)
async def hello():
    return {"message": "Hello from Python!"}

@app.get("/subgraph")
def subgraph():
    return get_subgraph(G)
