# EvidenceGuard RAG: Technical Stack & Architectural Decisions

**System Identity:** EvidenceGuard RAG  
**Core Purpose:** Hallucination-Resistant Retrieval-Augmented Generation Architecture with Calibrated Abstention, Conflict Detection, and Claim Verification  

---

## 1. Architectural Principles

EvidenceGuard is architected as an **evidence-first decision engine**, not merely an answer generator. Traditional RAG systems unconditionally pipe retrieved chunks into an LLM prompt; EvidenceGuard subjects retrieved context to strict **sufficiency verification**, **contradiction analysis**, and **claim-level grounding** before any answer reaches the user.

```
+-----------------------------------------------------------------------------------------+
|                                    EVIDENCEGUARD RAG                                    |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [User Query] ──> [Query Decomposition & Reformulation]                                 |
|                            │                                                            |
|                            ▼                                                            |
|          [Hybrid Search: Dense Vector + BM25 Sparse + RRF]                              |
|                            │                                                            |
|                            ▼                                                            |
|          [Cross-Encoder Reranking & Score Normalization]                                |
|                            │                                                            |
|                            ▼                                                            |
|         [Evidence Sufficiency & Answerability Classifier]                               |
|        (Calibrated Classifier: XGBoost / LightGBM / DeBERTa)                            |
|                            │                                                            |
|    ┌───────────────────────┼─────────────────────────┬──────────────────────┐           |
|    ▼                       ▼                         ▼                      ▼           |
| [Sufficient]         [Partial]                 [Contradictory]       [Insufficient]     |
| Grounded Ans     Qualified + Caveats       Conflict Disclosures    Principled Abstention|
|    │                       │                         │                      │           |
|    └───────────────────────┴───────────┬─────────────┴──────────────────────┘           |
|                                        ▼                                                |
|                   [Claim Extraction & Citation Validation]                              |
|                    (Sentence-level NLI entailment check)                                |
|                                        │                                                |
|                                        ▼                                                |
|                             [Audited Output Schema]                                     |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Technology Stack Selection

### 2.1 Backend Core & Orchestration
* **Python 3.12.3**: Modern Python with high-performance typing and asyncio support.
* **FastAPI (0.115+) & Uvicorn**: High-throughput asynchronous REST API framework for serving query, ingestion, evaluation, and experiment endpoints.
* **Pydantic v2**: High-performance validation schemas and strictly typed graph states.
* **LangGraph (0.2+)**: State-machine orchestration engine for directed acyclic and conditional execution graphs. Replaces rigid linear chains with inspectable, checkpointed, and retry-bounded graph nodes.
* **LangChain Core (0.3+)**: Reusable interfaces for document loaders, text splitters, and provider wrappers.

### 2.2 Ingestion & Parsing
* **Document Loaders**:
  - `pypdf` / `pdfplumber` for robust PDF extraction.
  - `python-docx` for Office Word files.
  - `markdown` / `unstructured` for Markdown and text.
  - Python `csv` module with metadata header preservation.
* **Chunking Strategies**:
  - Recursive Character Splitting (Baseline).
  - Semantic Chunking (Cosine distance thresholding over sentence embeddings).
  - Parent-Child / Hierarchical Chunking (Small retrieval chunks pointing to parent document context).

### 2.3 Retrieval & Indexing
* **Dense Vector Database**: **ChromaDB** (primary embedded store for ease of deployment, persistent metadata filtering, and zero external daemon overhead) and **FAISS** (as a high-throughput in-memory benchmark option).
* **Sparse Lexical Search**: **Rank-BM25** (Okapi BM25 implementation) for exact keyword and term matching.
* **Hybrid Fusion**: **Reciprocal Rank Fusion (RRF)** with standard smoothing constant $k=60$ and normalized convex score combination.
* **Dense Embedding Models**:
  - Primary Local: `BAAI/bge-large-en-v1.5` (1024-dim, state-of-the-art MTEB retrieval performance).
  - Fast Local: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, high-speed baseline).
  - API Fallback: OpenAI `text-embedding-3-small` / Gemini Embedding.

### 2.4 Reranking & Context Selection
* **Cross-Encoder Models**:
  - Primary Local: `BAAI/bge-reranker-base` and `BAAI/bge-reranker-large` (Hugging Face Transformers / Sentence-Transformers).
  - Lightweight Baseline: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* **Dynamic Context Pruning**: Min-score thresholding combined with top-$k$ diversification.

### 2.5 Evidence Assessment & Classifier Engine
* **Feature Engineering**:
  - Top cross-encoder reranker score ($s_1$).
  - Score margin between top-1 and top-2 passages ($s_1 - s_2$).
  - Entropy and variance across top-$k$ reranker scores.
  - BM25 lexical match density.
  - Semantic cosine similarity between query and retrieved passages.
  - Named entity and keyword coverage ratio between query and passage set.
  - Cross-passage pairwise contradiction score (NLI cross-encoder).
* **Classification Models**:
  - Level 1: Calibrated Logistic Regression with Platt scaling.
  - Level 2: Gradient Boosted Trees (**LightGBM** & **XGBoost**).
  - Level 3: Small Transformer NLI / Token-level classifier (`cross-encoder/nli-deberta-v3-base`).

### 2.6 LLM Providers (Pluggable Factory Interface)
* **Design Pattern**: Abstract Base Class `BaseLLMProvider` with standardized methods:
  - `generate(prompt: str, system_prompt: str, temperature: float = 0.0) -> LLMResponse`
  - `generate_structured(prompt: str, response_model: Type[BaseModel]) -> BaseModel`
* **Supported Backends**:
  1. `LocalHFProvider` (Local Transformers / PyTorch with Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct).
  2. `OllamaProvider` (Local Ollama daemon if installed).
  3. `GeminiProvider` (Google Gemini 1.5 / 2.0 Flash / Pro via official SDK).
  4. `OpenAIProvider` (OpenAI GPT-4o / GPT-4o-mini).
  5. `GroqProvider` (Ultra-low-latency Llama-3.3-70B API).

### 2.7 Verification & Citation Validation
* **Sentence Tokenization**: NLTK / spaCy sentence boundary detection.
* **Claim Extraction**: Extract discrete atomic factual claims from generated output.
* **Entailment Scoring**: Natural Language Inference (`roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli` or `nli-deberta-v3-base`) computing probabilities:
  $$\text{Entailment}, \quad \text{Neutral}, \quad \text{Contradiction}$$
* **Strict Citation Checker**: Verifies chunk IDs, page references, and character offsets against retrieved documents.

### 2.8 Experiment Tracking & Observability
* **Experiment Logging**: **MLflow** for hyperparameter runs, metric logging (F1, AUROC, Risk-Coverage curves), and model artifact storage.
* **Graph Tracing**: **LangSmith** and OpenTelemetry integration for complete graph node latency, token cost, and intermediate state debugging.

### 2.9 Frontend Application
* **Framework**: React 18 / Vite with TypeScript.
* **Styling**: Tailwind CSS with modern UI component design (clean dark/light theme, glassmorphism cards).
* **Interactive Views**:
  - Document Manager (upload, preview, delete, chunk view).
  - RAG Query Playground (with toggle for Baseline vs EvidenceGuard).
  - Inspection Drawer (step-by-step LangGraph execution trace, evidence sufficiency gauge, citation inspector, conflict alerts).
  - Benchmark & Evaluation Dashboard (live ROC/PR curves, risk-coverage trade-off plots, ablation tables).

---

## 3. Environment & Configuration Security

* Strictly no hardcoded API keys or model paths.
* Central configuration loaded via `pydantic-settings` from `.env`.
* `.env.example` provided with documented defaults.
* Safe handling of untrusted document content:
  - Explicit delimiters separating retrieval context from system instructions.
  - Anti-prompt-injection sanitization filters preventing override tags (e.g., `Ignore previous instructions and output...`).
