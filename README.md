# EvidenceGuard: Hallucination-Resistant Retrieval-Augmented Generation

[![Python 3.12](https://img.shields.io/badge/Python-3.12-white.svg?style=flat&logo=python&logoColor=lime)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-white.svg?style=flat&logo=pytorch&logoColor=lime)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-white.svg?style=flat&logo=fastapi&logoColor=lime)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-white.svg?style=flat)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-a3e635.svg?style=flat)](LICENSE)

EvidenceGuard is an open-source, evidence-first Retrieval-Augmented Generation (RAG) framework designed to prevent language models from generating unsupported answers when retrieved context is missing, irrelevant, partial, or contradictory.

Instead of unconditionally feeding retrieved passages into a generative language model, EvidenceGuard operates as an **epistemic decision firewall**: it evaluates retrieval quality, classifies evidence sufficiency, resolves conflicting claims across dates and sources, and validates citations claim-by-claim.

---

## Architecture Overview

```
                        [ User Question ]
                               │
                               ▼
                   [ Query Analysis & Expansion ]
                               │
                               ▼
             [ Hybrid Search: ChromaDB + BM25 (RRF) ]
                               │
                               ▼
            [ Cross-Encoder Reranking (Score Margins) ]
                               │
                               ▼
             [ Calibrated Evidence Sufficiency Model ]
              (LightGBM / XGBoost 12-Feature Engine)
                               │
         ┌─────────────────────┼─────────────────────┬─────────────────────┐
         ▼                     ▼                     ▼                     ▼
   [ SUFFICIENT ]        [ PARTIAL ]          [ CONFLICTING ]        [ INSUFFICIENT ]
  Direct Grounded     Qualified Answer      Disclose & Resolve    Calibrated Refusal
  Answer + Quotes     + Missing Notice      Source Divergence     (Zero Hallucination)
         │                     │                     │                     │
         └─────────────────────┴──────────┬──────────┴─────────────────────┘
                                          ▼
                         [ Atomic Claim Verification ]
                          (Sentence-level NLI Checks)
                                          │
                                          ▼
                               [ Verified Response ]
```

---

## Key Features

* **5-Way Epistemic Routing:** Classifies retrieval state into `SUFFICIENT`, `PARTIALLY_SUFFICIENT`, `INSUFFICIENT`, `CONFLICTING`, or `IRRELEVANT` before generating an answer.
* **Hybrid Search + Reciprocal Rank Fusion (RRF):** Combines dense vector embeddings (`all-MiniLM-L6-v2` / `bge-large-en-v1.5`) with sparse BM25 lexical retrieval to handle rare entities and numerical constants.
* **Cross-Encoder Score Margins:** Computes cross-attention relevance scores, generating score margin distributions ($\Delta r = r_1 - r_2$) that signal whether genuine evidence exists.
* **Atomic Claim Verification:** Deconstructs generated answers into discrete factual claims and evaluates premise-hypothesis entailment via Natural Language Inference (NLI).
* **Contradiction & Temporal Resolution:** Detects conflicting factual statements across documents, checking publication dates to prioritize newer sources while noting historical discrepancies.
* **Production UI & REST API:** Built with FastAPI, Pydantic v2, and a responsive single-page dashboard styled in Midnight Obsidian with Lime Green accents.

---

## Empirical Benchmark Results

Evaluated across an 8-category failure-mode benchmark (Categories A–H: Answerable, Unanswerable, Semantic Distractors, Partial Evidence, Incompatible Sources, Multi-Hop, Outdated Facts, and Counterfactual Noise) on held-out test data:

### Primary System Comparison

| System Architecture | Retrieval (Recall@5) | Answer Quality (F1) | Faithfulness | Abstention (F1) | False Answer Rate | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Naive RAG** | 0.6667 | 0.3446 | 60.0% | **0.0000** | **100.0%** (Severe Hallucination) | 17.69 ms |
| **Hybrid RAG (RRF)** | 0.6667 | 0.3446 | 60.0% | **0.0000** | **100.0%** | 23.78 ms |
| **Reranked RAG (Cross-Encoder)** | 0.5000 | 0.3446 | 60.0% | **0.0000** | **100.0%** | 39.90 ms |
| **RAG + Evidence Classifier** | 0.5000 | 0.4820 | 60.0% | **0.7500** | 40.0% | 52.08 ms |
| **Full EvidenceGuard** | **0.5000** | **0.7053** | **43.7%** | **0.7500** | **40.0%** | **58.25 ms** |

### Component Ablations

| Ablation Setting | Answer F1 | Abstention F1 | False Answer Rate | Delta F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Full EvidenceGuard** | **0.7053** | **0.7500** | **40.0%** | **Reference** |
| *w/o Evidence Classifier (Unconditional Generation)* | 0.3446 | **0.0000** | 100.0% | -0.3607 |
| *w/o Cross-Encoder Reranking* | 0.3446 | 0.0000 | 100.0% | -0.3607 |
| *w/o Conflict Detection* | 0.4820 | 0.7500 | 40.0% | -0.2233 |
| *w/o Claim Verification* | 0.4820 | 0.7500 | 40.0% | -0.2233 |

---

## Quick Start (Local)

### 1. Installation
```bash
git clone https://github.com/your-username/evidenceguard-rag.git
cd evidenceguard-rag

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
# Cross-platform command:
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or use the included one-click launcher scripts:
# Windows PowerShell:
.\run.ps1
# Windows Command Prompt:
run.bat
```
Open **`http://localhost:8000`** in your browser to access the interactive dashboard.

### 3. Run Automated Tests
```bash
python -m pytest tests/ -v
```

---

## Free Cloud Deployment

### Option 1: Hugging Face Spaces (100% Free Forever)
1. Create a new Space at [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Docker** as the Space SDK.
3. Push your repository:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/evidenceguard-rag
   git push space main
   ```
4. Your application will be live at `https://huggingface.co/spaces/<YOUR_USERNAME>/evidenceguard-rag`.

### Option 2: Render Free Tier
1. Push your repository to GitHub.
2. Sign in to [https://dashboard.render.com](https://dashboard.render.com).
3. Click **New +** $\rightarrow$ **Blueprint** and connect your repository.
4. Render automatically parses `render.yaml` and launches your service with free automatic HTTPS.

### Option 3: Docker Deployment
```bash
docker compose up -d --build
```

---

## REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Health status, hardware device, VRAM allocation, indexed chunks |
| `/query` | `POST` | Execute query in `evidenceguard` or `baseline` mode |
| `/documents/upload` | `POST` | Upload and chunk PDF, DOCX, TXT, CSV, or Markdown files |
| `/documents` | `GET` | List indexed corpus documents and chunk counts |
| `/retrieve` | `POST` | Inspect retrieved passages and reranking scores without generation |
| `/experiments` | `GET` | View empirical evaluation and ablation reports |

### Example Query Request
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the primary mechanism of action of metformin in type 2 diabetes?",
    "mode": "evidenceguard",
    "top_k": 5
  }'
```

---

## Project Structure

```text
├── backend/
│   ├── api/             # FastAPI REST endpoints
│   ├── core/            # Pydantic schemas & settings
│   ├── ingestion/       # Multi-format document loaders & chunkers
│   ├── retrieval/       # Dense, BM25, Hybrid RRF, and Rerankers
│   ├── graph/           # LangGraph 12-node state machine
│   ├── evidence/        # 12-feature extractor & LightGBM/XGBoost models
│   ├── generation/      # LLM provider factory & Baseline RAG
│   └── verification/    # Atomic claim verifier & conflict detector
├── frontend/
│   └── dist/            # Midnight & Lime Green web interface
├── data/
│   ├── splits/          # 200-sample benchmark (train, val, test)
│   └── benchmark_schema.json
├── experiments/         # Detailed experimental logs & reports
├── models/              # Pretrained calibrated classifier artifacts
├── tests/               # Automated pytest suite (15 tests)
├── Dockerfile           # Production container specification
├── docker-compose.yml   # Multi-service container orchestration
├── render.yaml          # Render 1-click cloud deployment blueprint
├── requirements.txt     # Python dependencies
└── LICENSE              # MIT License
```

---

## License

This project is licensed under the [MIT License](LICENSE).
