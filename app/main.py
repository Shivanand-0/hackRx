# app/main.py
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Header, HTTPException, Depends
from app.models import HackRxRequest, HackRxResponse
from app.config import settings
from app.services.rag_service import RAGService
import uuid

app = FastAPI(title="HackRx LLM Query System")

async def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {settings.BEARER_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

rag_service = RAGService()

@app.post("/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: HackRxRequest):
    namespace = str(uuid.uuid4())
    rag_service.process_and_index_document(doc_url=request.documents, namespace=namespace)
    answers = [rag_service.get_answer(question=q, namespace=namespace) for q in request.questions]
    try:
        rag_service.pinecone_index.delete(delete_all=True, namespace=namespace)
        print(f"Cleaned up namespace: {namespace}")
    except Exception as e:
        print(f"Could not clean up namespace {namespace}: {e}")
    return HackRxResponse(answers=answers)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "System is running!"}