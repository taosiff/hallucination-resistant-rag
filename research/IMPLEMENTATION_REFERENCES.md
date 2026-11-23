# EvidenceGuard: Implementation References & Citations

This document catalogs all verified academic publications, official code repositories, pretrained model weights, benchmarks, and licensing information utilized throughout the EvidenceGuard RAG system.

---

## 1. Academic Publications & arXiv Identifiers

1. **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**
   * *Authors:* Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
   * *Venue:* International Conference on Learning Representations (ICLR 2024)
   * *arXiv:* [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
   * *Code:* [https://github.com/AkariAsai/self-rag](https://github.com/AkariAsai/self-rag)
   * *Artifacts:* `selfrag/selfrag_llama2_7b`

2. **Corrective Retrieval Augmented Generation (CRAG)**
   * *Authors:* Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling
   * *Venue:* arXiv preprint / Association for Computational Linguistics (ACL 2024)
   * *arXiv:* [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)
   * *Code:* [https://github.com/HuskyInSalt/CRAG](https://github.com/HuskyInSalt/CRAG)

3. **RAFT: Adapting Language Model to Domain Specific RAG**
   * *Authors:* Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, Joseph E. Gonzalez
   * *Venue:* UC Berkeley / arXiv preprint (2024)
   * *arXiv:* [arXiv:2403.10131](https://arxiv.org/abs/2403.10131)
   * *Code:* [https://github.com/ShishirPatil/gorilla/tree/main/raft](https://github.com/ShishirPatil/gorilla/tree/main/raft)

4. **Benchmarking Large Language Models in Retrieval-Augmented Generation (RGB Benchmark)**
   * *Authors:* Jiawei Chen, Hongyu Lin, Xianpei Han, Le Sun
   * *Venue:* AAAI Conference on Human Computation and Artificial Intelligence (AAAI 2024)
   * *arXiv:* [arXiv:2309.01431](https://arxiv.org/abs/2309.01431)
   * *Code:* [https://github.com/chen700564/RGB](https://github.com/chen700564/RGB)

5. **RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models**
   * *Authors:* Cheng Yuan, Yu-Neng Chuang, Ruixiang Tang, Xia Hu
   * *Venue:* Association for Computational Linguistics (ACL 2024)
   * *arXiv:* [arXiv:2401.00396](https://arxiv.org/abs/2401.00396)
   * *Code:* [https://github.com/ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth)

6. **ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems**
   * *Authors:* Jon Saad-Falcon, Omar Khattab, Christopher Potts, Matei Zaharia
   * *Venue:* North American Chapter of the Association for Computational Linguistics (NAACL 2024)
   * *arXiv:* [arXiv:2311.09476](https://arxiv.org/abs/2311.09476)
   * *Code:* [https://github.com/stanford-futuredata/ARES](https://github.com/stanford-futuredata/ARES)

7. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**
   * *Authors:* Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert
   * *Venue:* European Chapter of the Association for Computational Linguistics (EACL 2024 Demo)
   * *arXiv:* [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
   * *Code:* [https://github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas)

8. **Know What You Don't Know: Unanswerable Questions for SQuAD**
   * *Authors:* Pranav Rajpurkar, Robin Jia, Percy Liang
   * *Venue:* Association for Computational Linguistics (ACL 2018)
   * *arXiv:* [arXiv:1806.03822](https://arxiv.org/abs/1806.03822)
   * *Dataset:* [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)

9. **Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)**
   * *Authors:* Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan
   * *Venue:* Association for Computational Linguistics (ACL 2023)
   * *arXiv:* [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)

10. **Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods**
    * *Authors:* Gordon V. Cormack, Charles L. A. Clarke, Stefan Buettcher
    * *Venue:* ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2009)
    * *DOI:* 10.1145/1571941.1572114

---

## 2. Pretrained Models & Hugging Face Hub References

| Component | Hugging Face Model Identifier | Parameters | Primary License | Intended Role |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Embeddings** | `BAAI/bge-large-en-v1.5` | 335M | MIT License | Semantic vector indexing and dense candidate retrieval |
| **Dense Embeddings (Fast)** | `sentence-transformers/all-MiniLM-L6-v2` | 22M | Apache 2.0 | Ultra-fast local embedding baseline |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-base` | 278M | MIT License | Cross-attention scoring of top-20 hybrid candidates |
| **Cross-Encoder Reranker (Fast)**| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | Apache 2.0 | High-speed reranking baseline |
| **NLI / Claim Verification** | `cross-encoder/nli-deberta-v3-base` | 86M | Apache 2.0 | Atomic claim entailment and contradiction detection |
| **Generative LLM (Local)** | `Qwen/Qwen2.5-3B-Instruct` / `meta-llama/Llama-3.2-3B-Instruct` | ~3B | Apache 2.0 / Llama 3.2 Community | Evidence-conditioned local answer generation & QLoRA |

---

## 3. Benchmark Datasets & Public Corpus References

| Benchmark / Dataset | Source / License | Tasks / Categories Evaluated | Format |
| :--- | :--- | :--- | :--- |
| **RGB Benchmark** | GitHub `chen700564/RGB` (MIT) | Noise Robustness, Negative Rejection, Information Integration, Counterfactual Robustness | JSON (`query`, `passage_list`, `ground_truth`) |
| **RAGTruth** | ParticleMedia / Hugging Face (CC-BY-4.0) | Word/sentence-level hallucination labels, open-domain QA, summarization | JSONL with span offsets and hallucination labels |
| **SQuAD 2.0** | Stanford (CC BY-SA 4.0) | Answerable questions + 50k plausible unanswerable questions | JSON hierarchical contexts and questions |
| **HotpotQA** | CMU / Stanford (CC BY-SA 4.0) | Multi-hop reasoning across multiple Wikipedia passages | JSON with `supporting_facts` annotations |
| **FEVER** | University of Sheffield (CC BY-SA 3.0) | Claim fact verification: `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO` | JSONL with claim strings and evidence doc IDs |

---

## 4. Software Libraries & Tooling Versions

* **LangGraph:** `langgraph>=0.2.30` — StateGraph orchestration, conditional routing.
* **LangChain:** `langchain>=0.3.0`, `langchain-core>=0.3.0` — Base abstraction layer.
* **ChromaDB:** `chromadb>=0.5.0` — Local persistent vector database.
* **Rank-BM25:** `rank_bm25>=0.2.2` — Pure-Python fast Okapi BM25 implementation.
* **Transformers & PEFT:** `transformers>=4.45.0`, `peft>=0.13.0`, `accelerate>=0.34.0` — Model loading, LoRA, and evaluation.
* **Scikit-Learn, LightGBM, XGBoost:** `scikit-learn>=1.5.0`, `lightgbm>=4.5.0`, `xgboost>=2.1.0` — Tabular classifiers for evidence assessment.
* **FastAPI & Uvicorn:** `fastapi>=0.115.0`, `uvicorn>=0.30.0` — High-performance REST service.
* **MLflow:** `mlflow>=2.16.0` — Experiment tracking and artifact registry.
