"""
Configuration module for NCET GenAI Multimodal RAG.
Reads environment variables, manages Streamlit secrets fallback, and maps multilingual codes.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_secret(key: str, default: str = "") -> str:
    """
    Safely retrieves a configuration key prioritizing Streamlit secrets,
    falling back to os.environ or a default value.
    """
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

# Groq LLM & Whisper Configurations
GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
GROQ_MODEL = get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_WHISPER_MODEL = "whisper-large-v3"

# Vector Database & Embeddings
CHROMA_PERSIST_DIR = get_secret("CHROMA_PERSIST_DIR", "./data/chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Propagate to environment for SDKs that require os.environ
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Multilingual Language Mappings
LANGUAGE_CODES = {
    "English": "en",
    "Hindi (हिंदी)": "hi",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Telugu (తెలుగు)": "te",
    "Tamil (தமிழ்)": "ta",
    "Marathi (मराठी)": "mr",
    "Bengali (বাংলা)": "bn",
    "German (Deutsch)": "de",
    "Spanish (Español)": "es"
}
