# app/main.py
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Header, HTTPException, Depends
from app.models import HackRxRequest, HackRxResponse
from app.config import settings
from app.services.rag_service import RAGService
import uuid
import asyncio

app = FastAPI(title="HackRx LLM Query System")

async def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {settings.BEARER_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

rag_service = RAGService()
limiter = asyncio.Semaphore(3)

async def get_answer_with_limit(question: str, namespace: str):
    async with limiter:
        return await rag_service.get_answer(question=question, namespace=namespace)

@app.post("/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: HackRxRequest):
    # Create a unique, temporary namespace for this single request
    namespace = str(uuid.uuid4())
    
    # --- STAGE 1: ON-THE-FLY INDEXING ---
    success = await rag_service.process_and_index_document(doc_url=request.documents, namespace=namespace)
    
    # --- STAGE 2: REAL-TIME ANSWERING ---
    if success:
        tasks = [get_answer_with_limit(question=q, namespace=namespace) for q in request.questions]
        answers = await asyncio.gather(*tasks)
    else:
        # If document processing failed, return an error for all questions
        error_answer = '{"answer": "Failed to process the document provided in the URL.", "rationale": "The document could not be downloaded or parsed correctly."}'
        answers = [error_answer for _ in request.questions]

    # --- STAGE 3: CLEANUP ---
    try:
        rag_service.pinecone_index.delete(delete_all=True, namespace=namespace)
        print(f"Cleaned up temporary namespace: {namespace}")
    except Exception as e:
        print(f"Could not clean up namespace {namespace}: {e}")
        
    return HackRxResponse(answers=answers)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "System is running!"}