# EvidenceGuard RAG: Hardware & Compute Environment Report

**Date:** November 2026  
**System Profile:** Local Development & Research Workstation  
**Project:** EvidenceGuard RAG — Hallucination-Resistant Retrieval-Augmented Generation  

---

## 1. System Hardware Specifications

| Component | Specification | Details & Metrics |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 Pro 64-bit | Build 19045 / NT 10.0 |
| **CPU** | Intel Core i7-14700K | 20 Cores (8 Performance Cores + 12 Efficient Cores), 28 Threads, up to 5.6 GHz Turbo |
| **System RAM** | 64 GB DDR5 | Total: 68,470,374,400 bytes (~63.76 GiB available to host) |
| **Discrete GPU** | NVIDIA GeForce RTX 4080 SUPER | Architecture: Ada Lovelace (AD103), Compute Capability 8.9 |
| **GPU VRAM** | 16,376 MiB (16 GB GDDR6X) | Memory Bus: 256-bit, Bandwidth: 736 GB/s, 10,240 CUDA Cores, 320 4th Gen Tensor Cores |
| **CUDA Driver / Runtime** | Driver: 610.88 / CUDA UMD 13.3 | PyTorch CUDA: 12.8 (Torch 2.11.0+cu128, FP16/BF16/FP8 supported) |
| **Storage (Primary / Data)** | D:\ NVMe SSD | Total Free Space: ~649.5 GB (Dedicated workspace & model cache) |
| **Storage (OS / System)** | C:\ NVMe SSD | Total Free Space: ~135.5 GB |
| **Python Environment** | Python 3.12.3 64-bit | Global environment with Torch 2.11+cu128, Scikit-learn, LightGBM, XGBoost, FastAPI |
| **Node.js Environment** | Node.js v24.20.0, npm 11.19.0 | Used for modern React/Vite/Next.js frontend dashboard |

---

## 2. Local Model Feasibility & Capability Assessment

The workstation features an **RTX 4080 SUPER (16 GB VRAM)** and **64 GB System RAM**. This places the development environment in a tier capable of executing both local inference and targeted parameter-efficient fine-tuning (PEFT/QLoRA), while retaining fast turn-around times.

### 2.1 Dense Embedding Models (Local Execution)
* **Candidate Models:**
  - `BAAI/bge-large-en-v1.5` (1024 dims, ~1.34 GB FP16)
  - `BAAI/bge-base-en-v1.5` (768 dims, ~438 MB FP16)
  - `sentence-transformers/all-MiniLM-L6-v2` (384 dims, ~90 MB FP16)
* **VRAM Consumption:** < 1.5 GB VRAM.
* **Throughput:** > 500 passages/second batch inference on Ada Lovelace Tensor Cores.
* **Verdict:** **100% Local Execution**. Eliminates external API latency, privacy leaks, and rate-limiting during document ingestion and retrieval.

### 2.2 Cross-Encoder Rerankers (Local Execution)
* **Candidate Models:**
  - `BAAI/bge-reranker-large` (560M params, ~2.2 GB FP16)
  - `BAAI/bge-reranker-base` (278M params, ~1.1 GB FP16)
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, ~90 MB FP16)
* **VRAM Consumption:** 0.2 GB to 2.2 GB VRAM.
* **Latency:** 15–40 ms for top-20 passage reranking.
* **Verdict:** **100% Local Execution**. Cross-encoders require passage pair scoring, which is cost-prohibitive over commercial APIs but blisteringly fast on local GPU.

### 2.3 Evidence Sufficiency & Abstention Classifiers (Local Execution)
* **Feature-Based Classifiers:** Logistic Regression, Random Forest, XGBoost, LightGBM trained on calibrated retrieval signals, semantic overlap, citation coverage, and cross-encoder score distributions.
  - Compute Cost: Negligible (< 100 MB RAM, < 5 ms inference).
* **Transformer-Based NLI / Sufficiency Encoders:**
  - `cross-encoder/nli-deberta-v3-base` or `microsoft/deberta-v3-large` (435M params, ~1.7 GB FP16).
* **Verdict:** **100% Local Execution**. Delivers sub-10ms decision routing before generation.

### 2.4 Generative LLM & Fine-Tuning Feasibility
* **Inference:**
  - 3B parameter models (e.g., `Llama-3.2-3B-Instruct`, `Qwen2.5-3B-Instruct`): ~6 GB in BF16, 2.5 GB in 4-bit NF4. Very fast local generation (~85 tokens/sec).
  - 7B/8B parameter models (e.g., `Llama-3.1-8B-Instruct`, `Qwen2.5-7B-Instruct`, `Mistral-7B-Instruct-v0.3`): ~9 GB in 8-bit, 5.5 GB in 4-bit NF4.
* **Fine-Tuning (QLoRA / LoRA):**
  - Fine-tuning a 3B model (LoRA r=16, alpha=32, batch size 4, gradient accumulation 4) fits easily in ~8 GB VRAM.
  - Fine-tuning an 8B model (QLoRA 4-bit base weights, 16-bit LoRA adapters) fits in ~11-13 GB VRAM.
* **Remote Commercial APIs:**
  - Gemini 1.5/2.0 Flash / Pro, OpenAI GPT-4o / GPT-4o-mini, Anthropic Claude 3.5 Sonnet, Groq Llama 3.3.
  - Cost: Free tier / low-cost ($0.075 to $0.15 per 1M tokens on Flash / mini).
* **Verdict:** **Hybrid Architecture**.
  - Provide a pluggable LLM provider interface (`LocalHFProvider`, `OllamaProvider`, `OpenAIProvider`, `GeminiProvider`).
  - Baseline and lightweight tests run locally or via fast APIs; fine-tuning experiments (Phase 7) leverage local QLoRA on 3B/7B models and DeBERTa-v3 sufficiency classifiers.

---

## 3. Compute Budget & Allocation Plan

| Component | Target Engine / Device | Peak Memory | Latency Budget | Target Cost |
| :--- | :--- | :--- | :--- | :--- |
| **Document Ingestion & Chunking** | CPU (i7-14700K 20 cores) | ~2 GB RAM | Batch (offline) | $0.00 |
| **Sparse Retrieval (BM25)** | CPU (In-memory BM25Okapi / Rank-BM25) | ~500 MB RAM | < 5 ms | $0.00 |
| **Dense Vector Index (Chroma/FAISS)**| GPU / CPU Accelerated FAISS | ~1 GB RAM/VRAM | < 10 ms | $0.00 |
| **Dense Embeddings (BGE-Large)** | RTX 4080 SUPER (CUDA) | ~1.5 GB VRAM | < 15 ms | $0.00 |
| **Cross-Encoder Reranker (BGE-Base)**| RTX 4080 SUPER (CUDA) | ~1.2 GB VRAM | < 30 ms | $0.00 |
| **Evidence Sufficiency Classifier** | RTX 4080 SUPER / CPU (XGBoost/LightGBM/DeBERTa)| < 1.0 GB VRAM | < 12 ms | $0.00 |
| **Claim-Level NLI / Verification**| RTX 4080 SUPER (CUDA) | ~1.5 GB VRAM | < 45 ms | $0.00 |
| **LLM Generation** | Configurable (Local Qwen2.5/Llama-3.2 or API)| ~4-8 GB VRAM or API | 300–1200 ms | Free / < $1.00 |
| **QLoRA Fine-Tuning (Phase 7)** | RTX 4080 SUPER (16 GB VRAM) | ~12–14 GB VRAM | Training run (1-2 hrs) | $0.00 |

**Total Compute Overhead:** Completely sustainable within the existing 16 GB VRAM GPU and 64 GB RAM workstation. Zero cloud GPU rental required.

---

## 4. Hardware Safety & Memory Isolation Guidelines

1. **VRAM Budget Cap:** Never allocate more than 14.5 GB of VRAM to PyTorch processes to prevent Windows Desktop Window Manager (DWM) stuttering or Out-of-Memory (OOM) crashes.
2. **Dynamic Unloading:** When switching between heavy QLoRA training and full-pipeline evaluation, explicitly flush `torch.cuda.empty_cache()` and run garbage collection.
3. **Quantization Standards:** Utilize `bitsandbytes` NF4 (NormalFloat4) with double quantization and BF16 compute dtype for local LLMs.
