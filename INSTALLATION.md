# EvidenceGuard: Installation & Deployment Guide

## 1. Local Environment Setup

### 1.1 Requirements
* **Operating System:** Windows 10/11 Pro or Linux (Ubuntu 22.04+)
* **Python:** Python 3.12 (64-bit recommended)
* **CUDA Hardware:** NVIDIA GPU with Compute Capability 8.0+ (RTX 3080/4080/4090 or A100/H100)
* **Memory:** Minimum 16 GB RAM (64 GB recommended for large indexing)
* **Storage:** 10 GB free SSD disk space for vector databases and cached model checkpoints.

### 1.2 Step-by-Step Installation
```bash
# 1. Clone repository
git clone https://github.com/EvidenceGuard/EvidenceGuard-RAG.git
cd EvidenceGuard-RAG

# 2. (Optional) Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux:
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
```

---

## 2. Running Automated Tests

Run the full pytest suite (15 unit, integration, and API tests):
```bash
python -m pytest tests/ -v
```

---

## 3. Running Experiments

```bash
# Generate benchmark splits (Categories A-H)
python scripts/build_benchmark_dataset.py

# Train & calibrate Evidence Sufficiency Classifiers (LightGBM, XGBoost, Logistic Reg)
python experiments/abstention/train_evidence_classifier.py

# Run Advanced Retrieval Experiments (Dense vs BM25 vs RRF vs Cross-Encoder)
python experiments/retrieval/run_retrieval_experiments.py

# Run RAFT Fine-Tuning & Adapter Training
python experiments/finetuning/prepare_finetuning_dataset.py
python experiments/finetuning/train_qlora_rag.py

# Run Master Benchmark & 6-Component Ablation Study
python experiments/ablations/run_benchmark_and_ablations.py
```

---

## 4. Launching the Application

### 4.1 Running with Python / Uvicorn (FastAPI + React Dashboard)
```bash
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or run the startup script:
.\run.ps1
# (or run.bat)
```
Open your browser at `http://localhost:8000` to access the full interactive interface.

### 4.2 Running with Docker Compose
```bash
docker-compose up --build
```
This launches both the EvidenceGuard API (`http://localhost:8000`) and the MLflow Tracking Server (`http://localhost:5000`).
