# EvidenceGuard: REST API Documentation

The EvidenceGuard REST API is built on FastAPI and exposes endpoints for health, document ingestion, query execution, retrieval inspection, experiment reporting, and model telemetry.

**Base URL:** `http://localhost:8000`

---

## 1. Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System status, GPU device, VRAM allocation, total indexed chunks |
| `POST` | `/documents/upload` | Upload & index document (PDF, TXT, DOCX, CSV, MD) |
| `GET` | `/documents` | List all indexed documents in ChromaDB |
| `DELETE` | `/documents/{id}` | Delete document and its vector chunks |
| `POST` | `/query` | Execute RAG query in `evidenceguard` or `baseline` mode |
| `POST` | `/query/evaluate` | Compare Baseline vs EvidenceGuard side-by-side |
| `POST` | `/retrieve` | Inspect retrieved & reranked passages without generation |
| `GET` | `/experiments` | List all available empirical experiment reports |
| `GET` | `/experiments/{id}` | Retrieve markdown content of specific experiment report |
| `GET` | `/models` | Retrieve model versions, parameters, and telemetry |

---

## 2. API Contracts & Examples

### 2.1 Health Check
**Request:**
```bash
curl -X GET http://localhost:8000/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "app_name": "EvidenceGuard RAG",
  "version": "1.0.0",
  "device": "NVIDIA GeForce RTX 4080 SUPER",
  "cuda_available": true,
  "vram_allocated_mb": 450.2,
  "vram_reserved_mb": 1024.0,
  "indexed_chunks": 140,
  "total_documents": 8
}
```

---

### 2.2 Execute Query (`POST /query`)
**Request:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the primary mechanism of action of metformin in treating type 2 diabetes?",
    "mode": "evidenceguard",
    "top_k": 5
  }'
```

**Response (200 OK):**
```json
{
  "query": "What is the primary mechanism of action of metformin in treating type 2 diabetes?",
  "mode": "evidenceguard",
  "answer": "Metformin lowers blood glucose primarily by decreasing hepatic glucose production (gluconeogenesis) and improving insulin sensitivity via AMPK activation [doc-A-1].",
  "decision": "SUFFICIENT",
  "confidence": 0.88,
  "citations": [
    {
      "citation_id": "cite_p1",
      "chunk_id": "p1",
      "source_title": "Clinical Pharmacology of Metformin",
      "snippet": "Metformin exerts its primary antihyperglycemic effect through...",
      "verified": true
    }
  ],
  "retrieved_passages": [
    {
      "id": "p1",
      "doc_id": "doc-A-1",
      "title": "Clinical Pharmacology of Metformin",
      "text": "Metformin exerts its primary antihyperglycemic effect through...",
      "dense_score": 0.82,
      "rerank_score": 0.88
    }
  ],
  "conflicts": null,
  "claim_verifications": [
    {
      "claim": "Metformin lowers blood glucose primarily by decreasing hepatic glucose production...",
      "supported": true,
      "citation_id": "p1",
      "entailment_score": 0.95
    }
  ],
  "abstention_reason": null,
  "execution_trace": [
    {"node": "analyze_query", "intent": "explanatory"},
    {"node": "retrieve_documents", "candidates_retrieved": 15},
    {"node": "rerank_documents", "top_rerank_score": 0.88},
    {"node": "assess_evidence", "decision": "SUFFICIENT", "confidence": 0.88},
    {"node": "generate_answer", "decision_context": "SUFFICIENT"},
    {"node": "verify_claims", "claims_extracted": 1, "supported_claims": 1},
    {"node": "validate_citations", "valid_citations_retained": 1},
    {"node": "finalize_response", "final_decision": "SUFFICIENT"}
  ],
  "latency_ms": 52.4
}
```

---

### 2.3 Upload Document (`POST /documents/upload`)
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@sample_policy.pdf" \
  -F "title=Company Security Policy" \
  -F "strategy=recursive" \
  -F "chunk_size=512" \
  -F "chunk_overlap=64"
```
