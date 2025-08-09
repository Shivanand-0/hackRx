# app/services/rag_service.py
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from app.config import settings
from app.services.document_service import download_and_parse_document, chunk_text

class RAGService:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.llm = genai.GenerativeModel(settings.GENERATIVE_MODEL)
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME, dimension=768, metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        self.pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

    async def process_and_index_document(self, doc_url: str, namespace: str):
        print(f"Starting on-the-fly processing for: {doc_url}")
        doc_name = doc_url.split('/')[-1].split('?')[0]
        text = download_and_parse_document(doc_url)
        if not text:
            print(f"Could not parse document: {doc_name}")
            return False
            
        chunks = chunk_text(text)
        if not chunks:
            print("Document chunking resulted in no chunks.")
            return False
            
        print(f"Creating {len(chunks)} embeddings via batch API call...")
        result = genai.embed_content(model="models/embedding-001", content=chunks, task_type="retrieval_document")
        embeddings = result['embedding']
        vectors_to_upsert = [(f"{doc_name}-chunk-{i}", emb, {"text": chunk}) for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
        
        self.pinecone_index.upsert(vectors=vectors_to_upsert, namespace=namespace)
        print(f"Finished indexing document into temporary namespace: {namespace}")
        return True

    # In app/services/rag_service.py

    async def get_answer(self, question: str, namespace: str) -> str:
        # HyDE Part 1: Generate a hypothetical answer (remains the same)
        hypothetical_answer_prompt = f'Generate a short, hypothetical paragraph that directly answers the user question, written in the style of an insurance policy document. QUESTION: "{question}"'
        try:
            hyde_response = self.llm.generate_content(hypothetical_answer_prompt)
            hypothetical_answer = hyde_response.text
        except Exception:
            hypothetical_answer = question

        # HyDE Part 2: Embed and search (remains the same)
        result = genai.embed_content(model="models/embedding-001", content=hypothetical_answer, task_type="retrieval_query")
        query_embedding = result['embedding']
        results = self.pinecone_index.query(vector=query_embedding, top_k=5, include_metadata=True, namespace=namespace)
        context = "\n\n".join([res['metadata']['text'] for res in results['matches']])

        if not context.strip():
            return "Could not retrieve relevant context from the document to answer this question."
            
        # --- THIS PROMPT IS THE ONLY CHANGE ---
        # It now asks for a direct string answer.
        prompt = f"""
        You are a meticulous insurance policy analyst. Based ONLY on the provided context, provide a precise and direct answer to the question.

        CONTEXT:
        ---
        {context}
        ---
        
        INSTRUCTIONS:
        1. Your answer must be a single, concise paragraph or sentence that directly answers the question.
        2. Do not include any preamble like "The answer is..." or "Based on the context...".
        3. If the information is 100% absent from the context, respond with "The policy does not specify this detail."

        QUESTION: {question}
        """
        try:
            response = await self.llm.generate_content_async(prompt)
            # We no longer need to parse JSON, just return the text.
            return response.text.strip()
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return "Error: The generative model failed to produce an answer."