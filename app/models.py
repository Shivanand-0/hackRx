# app/models.py
from pydantic import BaseModel
from typing import List

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]