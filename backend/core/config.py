"""
EvidenceGuard Core Configuration
Central settings management with Pydantic v2 settings.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Application Info
    APP_NAME: str = "EvidenceGuard RAG"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Base Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "data" / "chroma_db"
    EXPERIMENTS_DIR: Path = BASE_DIR / "experiments"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # LLM Settings
    DEFAULT_LLM_PROVIDER: str = Field(default="mock", description="mock, local_hf, openai, gemini, ollama")
    LOCAL_MODEL_NAME_OR_PATH: str = "Qwen/Qwen2.5-3B-Instruct"
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_API_KEY: str = Field(default="", description="Gemini API key")
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    # Embedding & Reranker Models
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DEVICE: str = "cuda"  # auto-detected in code if torch.cuda.is_available()
    
    # Ingestion & Chunking
    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 64
    DEFAULT_CHUNKING_STRATEGY: str = "recursive"  # recursive, fixed, semantic, parent_child
    
    # Retrieval Hyperparameters
    DENSE_TOP_K: int = 15
    SPARSE_TOP_K: int = 15
    RRF_K: int = 60
    RERANK_TOP_K: int = 5
    
    # Epistemic & Verification Thresholds
    SUFFICIENCY_THRESHOLD: float = 0.65
    ABSTENTION_THRESHOLD: float = 0.35
    CONTRADICTION_THRESHOLD: float = 0.40
    ENTAILMENT_THRESHOLD: float = 0.50
    MAX_QUERY_RETRY_COUNT: int = 2
    
    # Telemetry
    ENABLE_LANGSMITH: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "evidenceguard-rag"
    ENABLE_MLFLOW: bool = True
    MLFLOW_TRACKING_URI: str = "file:./experiments/mlruns"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(settings.EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
