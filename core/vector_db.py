"""
Vector Database Module for PragyanAI GenAI Multimodal RAG.
Provides BGE embeddings generation, persistent ChromaDB indexing, 
fast in-memory FAISS vector indexing, MMR/similarity retrieval, and collection management.
"""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Instantiates the local HuggingFace embedding model (BAAI/bge-small-en-v1.5)
    with normalized embeddings for cosine similarity calculations.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def prepare_document_chunks(
    extracted_docs: List[Dict[str, Any]],
    chunk_size: int = 750,
    chunk_overlap: int = 120,
) -> List[Document]:
    """
    Splits a list of document dictionaries into structured LangChain Document chunks.
    """
    if not extracted_docs:
        raise ValueError("Cannot process an empty list of documents.")

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

        texts.append(content.strip())
        metadatas.append(
            {
                "source": doc.get("source", "Unknown"),
                "title": doc.get("title", "Untitled"),
                "type": doc.get("type", "generic"),
            }
        )

    if not texts:
        raise ValueError("All provided documents contained empty content.")

    return text_splitter.create_documents(texts=texts, metadatas=metadatas)


# =====================================================================
# 1. ChromaDB (Persistent Disk Storage)
# =====================================================================

def index_documents_to_chroma(
    extracted_docs: List[Dict[str, Any]],
    persist_dir: str = CHROMA_PERSIST_DIR,
    chunk_size: int = 750,
    chunk_overlap: int = 120,
) -> Chroma:
    """
    Splits extracted raw text documents into chunks and initializes/replaces 
    the local persistent Chroma vector database.
    """
    os.makedirs(persist_dir, exist_ok=True)
    chunked_docs = prepare_document_chunks(extracted_docs, chunk_size, chunk_overlap)
    embeddings = get_embeddings()

    return Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )


def append_documents_to_chroma(
    extracted_docs: List[Dict[str, Any]],
    persist_dir: str = CHROMA_PERSIST_DIR,
    chunk_size: int = 750,
    chunk_overlap: int = 120,
) -> Chroma:
    """
    Appends new documents incrementally into an existing ChromaDB instance.
    """
    existing_db = load_existing_chroma_db(persist_dir)
    chunked_docs = prepare_document_chunks(extracted_docs, chunk_size, chunk_overlap)

    if existing_db is None:
        return index_documents_to_chroma(extracted_docs, persist_dir, chunk_size, chunk_overlap)

    existing_db.add_documents(chunked_docs)
    return existing_db


def load_existing_chroma_db(
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> Optional[Chroma]:
    """
    Loads an existing persisted Chroma database from disk if present.
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


def clear_chroma_db(persist_dir: str = CHROMA_PERSIST_DIR) -> bool:
    """
    Deletes the persisted ChromaDB directory from disk for clean resets.
    """
    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir)
            os.makedirs(persist_dir, exist_ok=True)
            return True
        except Exception as e:
            print(f"[Error in vector_db]: Could not delete Chroma DB directory: {e}")
            return False
    return True


# =====================================================================
# 2. FAISS (Ultra-Fast In-Memory Storage)
# =====================================================================

def index_documents_to_faiss(
    extracted_docs: List[Dict[str, Any]],
    chunk_size: int = 750,
    chunk_overlap: int = 120,
) -> FAISS:
    """
    Creates an in-memory FAISS vector index for real-time sub-millisecond retrieval.
    """
    chunked_docs = prepare_document_chunks(extracted_docs, chunk_size, chunk_overlap)
    embeddings = get_embeddings()
    return FAISS.from_documents(documents=chunked_docs, embedding=embeddings)


def append_documents_to_faiss(
    existing_faiss: Optional[FAISS],
    extracted_docs: List[Dict[str, Any]],
    chunk_size: int = 750,
    chunk_overlap: int = 120,
) -> FAISS:
    """
    Appends new document chunks into an existing in-memory FAISS index.
    """
    chunked_docs = prepare_document_chunks(extracted_docs, chunk_size, chunk_overlap)
    if existing_faiss is None:
        return index_documents_to_faiss(extracted_docs, chunk_size, chunk_overlap)

    existing_faiss.add_documents(chunked_docs)
    return existing_faiss


# =====================================================================
# 3. Unified Search & Retriever Utilities
# =====================================================================

def get_custom_retriever(
    vectorstore: Union[Chroma, FAISS],
    search_type: str = "mmr",
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.7,
):
    """
    Constructs a LangChain retriever with configurable search types.
    
    Args:
        vectorstore: Chroma or FAISS vectorstore instance.
        search_type (str): 'similarity', 'mmr' (Maximal Marginal Relevance), or 'similarity_score_threshold'.
        k (int): Number of top context chunks to return.
        fetch_k (int): Candidates to fetch before MMR reranking (used if search_type='mmr').
        lambda_mult (float): Diversity vs relevance weighting for MMR (0 to 1).
    """
    if search_type == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )
    elif search_type == "similarity_score_threshold":
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": 0.3},
        )
    else:
        return vectorstore.as_retriever(search_kwargs={"k": k})


def search_with_relevance_scores(
    vectorstore: Union[Chroma, FAISS],
    query: str,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Searches vector database and returns matched documents alongside similarity distance scores.
    """
    return vectorstore.similarity_search_with_score(query, k=k)
