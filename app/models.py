# app/models.py
from pydantic import BaseModel
from typing import List

class HackRxRequest(BaseModel):
    documents: str  # A single document URL as a string
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]