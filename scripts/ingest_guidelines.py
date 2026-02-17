"""
Ingest RAG knowledge document and prepare it for embedding + retrieval.
"""
from pathlib import Path
from typing import List

from src.config import PATHS


GUIDELINES_DIR = PATHS.project_root / "data" / "raw" / "guidelines"
RAG_DOC_PATH = GUIDELINES_DIR / "RAG KNOWLEDGE DOCUMENT.txt"


def load_rag_knowledge_document() -> str:
    """
    Load the RAG knowledge document text.
    """
    if not RAG_DOC_PATH.exists():
        raise FileNotFoundError(
            f"RAG knowledge document not found at {RAG_DOC_PATH}. "
            "Please ensure 'RAG KNOWLEDGE DOCUMENT.txt' exists in data/raw/guidelines/"
        )
    
    with RAG_DOC_PATH.open("r", encoding="utf-8") as f:
        text = f.read()
    
    return text


def chunk_guidelines_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split guidelines text into overlapping chunks for embedding.
    
    Args:
        text: Full guidelines text
        chunk_size: Target characters per chunk
        overlap: Characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    words = text.split()
    
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_words + [word]
            current_length = sum(len(w) + 1 for w in current_chunk)
        else:
            current_chunk.append(word)
            current_length += word_length
    
    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def ingest_rag_guidelines() -> List[str]:
    """
    Main function to ingest and chunk the RAG knowledge document.
    
    Returns:
        List of text chunks ready for embedding
    """
    print(f"Loading RAG knowledge document from {RAG_DOC_PATH}...")
    text = load_rag_knowledge_document()
    
    print(f"Document length: {len(text)} characters")
    
    print("Chunking text for embedding...")
    chunks = chunk_guidelines_text(text, chunk_size=500, overlap=50)
    
    print(f"Created {len(chunks)} text chunks")
    
    return chunks


if __name__ == "__main__":
    chunks = ingest_rag_guidelines()
    print(f"\n✅ Successfully ingested {len(chunks)} guideline chunks")
    print("\nFirst chunk preview:")
    print("-" * 60)
    print(chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0])
    print("-" * 60)
