# app/services/document_service.py
import requests
import io
from pypdf import PdfReader
from docx import Document
from typing import List
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def download_and_parse_document(url: str) -> str:
    """Downloads a document from a URL and extracts its text."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')
        file_stream = io.BytesIO(response.content)

        if "pdf" in content_type or url.endswith(".pdf"):
            reader = PdfReader(file_stream)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif "openxmlformats-officedocument.wordprocessingml.document" in content_type or url.endswith(".docx"):
            doc = Document(file_stream)
            text = "\n\n".join([para.text for para in doc.paragraphs if para.text])
        else:
            text = response.text
        return re.sub(r'\s{2,}', ' ', text) # Clean up extra whitespace
    except requests.exceptions.RequestException as e:
        print(f"Error downloading document from {url}: {e}")
        return ""

# --- REPLACED WITH A ROBUST LIBRARY IMPLEMENTATION ---
def chunk_text(text: str) -> List[str]:
    """Splits text into chunks using a robust, recursive method."""
    if not text:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a chunk size that is safely under the 40KB metadata limit
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)