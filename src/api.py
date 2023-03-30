from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gpt import generate_solution, generate_index

class GptIndexer(BaseModel):
    name: str
    input: Optional[str] = None
    documents: Optional[List[str]] = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"successful": True}

@app.post("/index")
def create_index(payload: GptIndexer):
    generate_index(payload)
    return {"successful": True}

@app.post("/prompt")
def create_solution(payload: GptIndexer):
    solution = generate_solution(payload)
    return {"solution": solution}