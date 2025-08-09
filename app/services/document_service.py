# app/services/document_service.py
import requests
import io
from pypdf import PdfReader
from docx import Document
from typing import List
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def download_and_parse_document(url: str) -> str:
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
        else: # Add other formats like docx if needed
            text = response.text
        return re.sub(r'\s{2,}', ' ', text)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading document from {url}: {e}")
        return ""

def chunk_text(text: str) -> List[str]:
    if not text: return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)