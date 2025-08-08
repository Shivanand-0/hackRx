# app/services/rag_service.py

# We no longer import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from app.config import settings
from app.services.document_service import download_and_parse_document, chunk_text
import time

class RAGService:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.llm = genai.GenerativeModel(settings.GENERATIVE_MODEL)
        
        # We no longer load the embedding model here, saving memory.
        
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # The embedding dimension for Google's model is 768.
        pinecone_dimension = 768 

        if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=pinecone_dimension, # Use the new dimension
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-west-2')
            )
        self.pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

    def process_and_index_document(self, doc_url: str, namespace: str):
        print(f"Starting document processing for: {doc_url}")
        doc_name = doc_url.split('/')[-1].split('?')[0]
        text = download_and_parse_document(doc_url)
        if not text:
            print(f"Could not parse document: {doc_name}")
            return
            
        chunks = chunk_text(text)
        if chunks:
            print(f"Creating {len(chunks)} embeddings via API...")
            # Use Google's API for embeddings
            # We add a small delay between calls to respect API rate limits
            embeddings = []
            for chunk in chunks:
                embeddings.append(genai.embed_content(model="models/embedding-001", content=chunk)['embedding'])
                time.sleep(1) # Add a 1-second delay
            
            vectors_to_upsert = [(f"{doc_name}-chunk-{i}", emb, {"text": chunk}) for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
            self.pinecone_index.upsert(vectors=vectors_to_upsert, namespace=namespace)
        print(f"Finished indexing document: {doc_name}")

    def get_answer(self, question: str, namespace: str) -> str:
        # Use Google's API for the query embedding as well
        query_embedding = genai.embed_content(model="models/embedding-001", content=question)['embedding']

        results = self.pinecone_index.query(vector=query_embedding, top_k=5, include_metadata=True, namespace=namespace)
        context = "\n\n".join([res['metadata']['text'] for res in results['matches']])
        prompt = f"""
        You are a meticulous insurance policy analyst. Your task is to provide a precise and direct answer to the user's question based ONLY on the provided context from the policy document.
        CONTEXT:
        ---
        {context}
        ---
        INSTRUCTIONS:
        1. Scrutinize the context to find the exact information that answers the question.
        2. Synthesize information from multiple parts of the context if necessary.
        3. Your answer must be a direct response to the question, not a summary of the context.
        4. If the context explicitly contains the answer, you MUST provide it. Do not apologize or claim the information is missing if it's present.
        5. If, and only if, the context definitively does NOT contain the information to answer the question, state that the policy does not specify the detail.
        
        QUESTION: {question}
        ---
        Your entire output must be a single, clean JSON object with two keys: "answer" and "rationale".
        The "answer" should be the direct answer extracted from the context.
        The "rationale" should be a brief explanation of which part of the context justifies your answer.
        """
        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip().replace("```json", "").replace("```", "")
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return '{"answer": "Error: Could not retrieve an answer.", "rationale": "The generative model failed."}'