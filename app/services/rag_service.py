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
        pinecone_dimension = 768 
        if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME, dimension=pinecone_dimension, metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        self.pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

    # In app/services/rag_service.py

    async def process_and_index_document(self, doc_url: str, namespace: str):
        print(f"Starting document processing for: {doc_url}")
        doc_name = doc_url.split('/')[-1].split('?')[0]
        text = download_and_parse_document(doc_url)
        if not text:
            print(f"Could not parse document: {doc_name}")
            return
            
        chunks = chunk_text(text)
        
        # --- THIS IS THE NEW LINE ---
        # Filter out any empty strings from the list before sending to the API
        chunks = [chunk for chunk in chunks if chunk.strip()]
        # --- END OF NEW LINE ---

        if chunks:
            print(f"Creating {len(chunks)} embeddings via a single batch API call...")
            # Use a single batch call for embeddings - MUCH FASTER
            result = genai.embed_content(model="models/embedding-001", content=chunks, task_type="retrieval_document")
            embeddings = result['embedding']

            vectors_to_upsert = [(f"{doc_name}-chunk-{i}", emb, {"text": chunk}) for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
            self.pinecone_index.upsert(vectors=vectors_to_upsert, namespace=namespace)
        print(f"Finished indexing document: {doc_name}")

    # In app/services/rag_service.py, replace the get_answer function

    async def get_answer(self, question: str, namespace: str) -> str:
        # HyDE Part 1: Generate a hypothetical answer
        hypothetical_answer_prompt = f"""
        Please generate a short, hypothetical paragraph that directly answers the following user question. 
        This paragraph should be written in the style of an insurance policy document.
        Do not say "this is a hypothetical answer". Just generate the text.
        
        QUESTION: "{question}"
        """
        try:
            # Use the synchronous call for this quick, single generation
            hyde_response = self.llm.generate_content(hypothetical_answer_prompt)
            hypothetical_answer = hyde_response.text
        except Exception as e:
            print(f"HyDE generation failed: {e}")
            # Fallback to the original question if HyDE fails
            hypothetical_answer = question

        # HyDE Part 2: Embed the hypothetical answer and use it to search
        result = genai.embed_content(model="models/embedding-001", content=hypothetical_answer, task_type="retrieval_query")
        query_embedding = result['embedding']

        # With a better query vector, we can be more precise and retrieve fewer documents.
        # This also makes the final prompt smaller and faster for the LLM.
        results = self.pinecone_index.query(vector=query_embedding, top_k=5, include_metadata=True, namespace=namespace)
        context = "\n\n".join([res['metadata']['text'] for res in results['matches']])

        # Final, hyper-aggressive prompt (this remains the same)
        prompt = f"""
        You are a world-class, meticulous insurance policy analyst. Your job performance is critically evaluated based on your ability to provide a precise and direct answer to the user's question based ONLY on the provided context from a policy document.

        CONTEXT:
        ---
        {context}
        ---

        INSTRUCTIONS:
        1. Your primary directive is to find the answer within the context. Scrutinize every word.
        2. Synthesize information from multiple parts of the context if necessary to form a complete answer.
        3. Your answer MUST be a direct response to the question, not a summary of the context.
        4. If the information is present, you MUST extract it. Do not apologize or claim the information is missing if the answer can be inferred or directly found.
        5. Only if you have exhaustively searched the context and are 100% certain the information is absent, you must state that the policy does not specify the detail. Do not use this as an easy way out.
        
        QUESTION: {question}
        ---

        Your entire output must be a single, clean JSON object with two keys: "answer" and "rationale".
        The "answer" should be the direct answer extracted from the context.
        The "rationale" should be a brief explanation of which part of the context justifies your answer, or why the information is definitively absent.
        """
        try:
            # Use the async call for the final, more complex generation
            response = await self.llm.generate_content_async(prompt)
            return response.text.strip().replace("```json", "").replace("```", "")
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return '{"answer": "Error: Could not retrieve an answer.", "rationale": "The generative model failed."}'