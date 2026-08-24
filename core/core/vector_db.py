"""
Vector Database Module for NCET GenAI Multimodal RAG.
Handles document splitting, embedding generation via HuggingFace BGE models,
and persistence/querying with ChromaDB.
"""

import os
from typing import Any, Dict, List, Optional
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Instantiates the local HuggingFace embedding model (BAAI/bge-small-en-v1.5)
    with normalized embeddings for accurate cosine similarity search.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def index_documents_to_chroma(
    extracted_docs: List[Dict[str, Any]],
    persist_dir: str = CHROMA_PERSIST_DIR,
    chunk_size: int = 750,
    chunk_overlap: int = 120,
) -> Chroma:
    """
    Splits extracted raw text documents into overlapping semantic chunks
    and indexes them into the local Chroma vector database.

    Args:
        extracted_docs (List[Dict[str, Any]]): List of document dicts with keys:
            - 'content': Raw extracted text content
            - 'source': Source file name or URL
            - 'title': Document or video title
            - 'type': Ingestion source type (pdf, docx, pptx, web, youtube)
        persist_dir (str): Path to directory where ChromaDB stores vectors.
        chunk_size (int): Max character length per chunk.
        chunk_overlap (int): Overlap characters between consecutive chunks.

    Returns:
        Chroma: Initialized and populated LangChain Chroma vectorstore instance.
    """
    if not extracted_docs:
        raise ValueError("Cannot index an empty list of documents.")

    # Ensure target persistence directory exists
    os.makedirs(persist_dir, exist_ok=True)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for doc in extracted_docs:
        content = doc.get("content", "")
        if not content or not content.strip():
            continue

        texts.append(content)
        metadatas.append(
            {
                "source": doc.get("source", "Unknown"),
                "title": doc.get("title", "Untitled"),
                "type": doc.get("type", "generic"),
            }
        )

    if not texts:
        raise ValueError("All provided documents contained empty content.")

    # Split documents into chunk objects with attached metadata
    chunked_docs = text_splitter.create_documents(texts=texts, metadatas=metadatas)

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    return vectorstore


def load_existing_chroma_db(
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> Optional[Chroma]:
    """
    Loads an existing persisted Chroma database from disk if available.

    Args:
        persist_dir (str): Directory where vectors were previously saved.

    Returns:
        Optional[Chroma]: Loaded vectorstore, or None if directory is empty/missing.
    """
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        return None

    try:
        embeddings = get_embeddings()
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
    except Exception as e:
        print(f"[Warning in vector_db]: Failed loading existing Chroma DB: {e}")
        return None
