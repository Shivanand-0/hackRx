# app/services/document_service.py
import requests
import io
from pypdf import PdfReader
from docx import Document
from typing import List

def download_and_parse_document(url: str) -> str:
    """Downloads a document from a URL and extracts its text."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        content_type = response.headers.get('content-type', '')
        file_stream = io.BytesIO(response.content)

        if "pdf" in content_type or url.endswith(".pdf"):
            reader = PdfReader(file_stream)
            text = "".join(page.extract_text() for page in reader.pages)
        elif "openxmlformats-officedocument.wordprocessingml.document" in content_type or url.endswith(".docx"):
            doc = Document(file_stream)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            # Fallback for plain text or other types
            text = response.text
        
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading document from {url}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """Splits text into smaller, overlapping chunks."""
    if not text:
        return []
        
    words = text.split()
    if not words:
        return []

    chunks = []
    current_pos = 0
    while current_pos < len(words):
        start = current_pos
        end = current_pos + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        current_pos += chunk_size - chunk_overlap
        if current_pos >= len(words):
            break
            
    return chunks