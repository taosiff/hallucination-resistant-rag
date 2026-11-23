"""
EvidenceGuard Production REST API
FastAPI application providing health, document management, query execution,
inspection, experiment reporting, and model telemetry endpoints.
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch

from backend.core.config import settings
from backend.core.models import (
    Document,
    DocumentMetadata,
    Chunk,
    ScoredPassage,
    QueryRequest,
    QueryResponse,
    EpistemicDecision
)
from backend.ingestion.loaders import DocumentLoader, sanitize_text, compute_hash
from backend.ingestion.chunker import ChunkingEngine
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.evidence.classifier import EvidenceSufficiencyClassifier
from backend.generation.baseline_rag import BaselineRAG
from backend.graph.workflow import EvidenceGuardRAG

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Hallucination-Resistant Retrieval-Augmented Generation REST API"
)

# Enable CORS for React/Vite development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton instances
hybrid_retriever = HybridRetriever()
baseline_rag = BaselineRAG(retriever=hybrid_retriever)
evidenceguard_rag = EvidenceGuardRAG()
classifier = EvidenceSufficiencyClassifier()

# In-memory document registry for listing
DOCUMENT_REGISTRY: Dict[str, Dict[str, Any]] = {}

@app.get("/health")
def health_check():
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_alloc = torch.cuda.memory_allocated(0) / (1024**2) if cuda_avail else 0.0
    vram_res = torch.cuda.memory_reserved(0) / (1024**2) if cuda_avail else 0.0
    
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "device": device_name,
        "cuda_available": cuda_avail,
        "vram_allocated_mb": round(vram_alloc, 2),
        "vram_reserved_mb": round(vram_res, 2),
        "indexed_chunks": hybrid_retriever.vector_store.count(),
        "total_documents": len(DOCUMENT_REGISTRY)
    }

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    date: Optional[str] = Form(None),
    chunk_size: int = Form(settings.DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(settings.DEFAULT_CHUNK_OVERLAP),
    strategy: str = Form("recursive")
):
    upload_dir = settings.DATA_DIR / "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    temp_path = upload_dir / file.filename
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        doc = DocumentLoader.load_from_file(
            file_path=str(temp_path),
            metadata={"title": title or file.filename, "date": date}
        )
        chunks = ChunkingEngine.chunk_document(
            doc,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        hybrid_retriever.index_chunks(chunks)
        
        DOCUMENT_REGISTRY[doc.id] = {
            "doc_id": doc.id,
            "title": doc.metadata.title,
            "source": file.filename,
            "file_type": doc.metadata.file_type,
            "date": doc.metadata.date,
            "chunks_count": len(chunks),
            "timestamp": time.time()
        }
        
        return {
            "message": "Document successfully ingested and indexed.",
            "doc_id": doc.id,
            "title": doc.metadata.title,
            "chunks_indexed": len(chunks),
            "strategy": strategy
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process document: {str(e)}")

@app.get("/documents")
def list_documents():
    return {
        "documents": list(DOCUMENT_REGISTRY.values()),
        "total_chunks": hybrid_retriever.vector_store.count()
    }

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if doc_id in DOCUMENT_REGISTRY:
        hybrid_retriever.vector_store.delete_by_doc_id(doc_id)
        del DOCUMENT_REGISTRY[doc_id]
        return {"message": f"Document {doc_id} deleted successfully."}
    raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

@app.post("/query", response_model=QueryResponse)
def execute_query(request: QueryRequest):
    if request.mode.lower() == "baseline":
        return baseline_rag.query(request)
    else:
        return evidenceguard_rag.query(request)

@app.post("/retrieve")
def retrieve_only(query: str, top_k: int = 5, mode: str = "hybrid_rerank"):
    passages = hybrid_retriever.retrieve(query=query, mode=mode, top_k=top_k)
    assessment = classifier.predict(query, passages)
    return {
        "query": query,
        "mode": mode,
        "retrieved_count": len(passages),
        "evidence_assessment": assessment,
        "passages": passages
    }

@app.post("/query/evaluate")
def evaluate_query(query: str, ground_truth: Optional[str] = None):
    # Run both baseline and evidenceguard for direct side-by-side comparison
    req_base = QueryRequest(query=query, mode="baseline")
    req_eg = QueryRequest(query=query, mode="evidenceguard")
    
    resp_base = baseline_rag.query(req_base)
    resp_eg = evidenceguard_rag.query(req_eg)
    
    return {
        "query": query,
        "ground_truth": ground_truth,
        "baseline": resp_base,
        "evidenceguard": resp_eg,
        "comparison": {
            "abstention_contrast": f"Baseline: {resp_base.decision.value} vs EvidenceGuard: {resp_eg.decision.value}",
            "confidence_contrast": f"Baseline: {resp_base.confidence:.2f} vs EvidenceGuard: {resp_eg.confidence:.2f}",
            "citations_contrast": f"Baseline cites: {len(resp_base.citations)} vs EvidenceGuard verified cites: {len(resp_eg.citations)}"
        }
    }

@app.get("/experiments")
def list_experiments():
    exp_files = [
        {"id": "abstention", "title": "Evidence Sufficiency & Calibrated Abstention Report", "path": "experiments/abstention/CLASSIFIER_EXPERIMENT_REPORT.md"},
        {"id": "retrieval", "title": "Advanced Retrieval & Hybrid Search Experiments", "path": "experiments/retrieval/RETRIEVAL_EXPERIMENT_REPORT.md"},
        {"id": "finetuning", "title": "RAFT Fine-Tuning on Noisy Retrieval Contexts", "path": "experiments/finetuning/FINETUNING_EXPERIMENT_REPORT.md"},
        {"id": "ablations", "title": "Master Benchmark Evaluation & Component Ablation Report", "path": "experiments/ablations/BENCHMARK_AND_ABLATION_REPORT.md"}
    ]
    return {"experiments": exp_files}

@app.get("/experiments/{exp_id}")
def get_experiment_report(exp_id: str):
    file_map = {
        "abstention": "experiments/abstention/CLASSIFIER_EXPERIMENT_REPORT.md",
        "retrieval": "experiments/retrieval/RETRIEVAL_EXPERIMENT_REPORT.md",
        "finetuning": "experiments/finetuning/FINETUNING_EXPERIMENT_REPORT.md",
        "ablations": "experiments/ablations/BENCHMARK_AND_ABLATION_REPORT.md"
    }
    if exp_id not in file_map:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found.")
    
    file_path = BASE_DIR / file_map[exp_id]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Experiment report file not found on disk.")
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"id": exp_id, "content": content}

@app.get("/models")
def get_model_telemetry():
    return {
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "reranker_model": settings.RERANKER_MODEL_NAME,
        "classifier_model": classifier.model_type,
        "llm_provider": settings.DEFAULT_LLM_PROVIDER,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__
    }

# Mount frontend static files if built
frontend_dist = BASE_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
