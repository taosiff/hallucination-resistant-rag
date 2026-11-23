"""
EvidenceGuard ChromaDB Vector Store Implementation
Provides persistent dense vector indexing and metadata-filtered similarity search.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.core.config import settings
from backend.core.models import Chunk, ScoredPassage
from backend.retrieval.embeddings import EmbeddingService

class ChromaVectorStore:
    def __init__(self, collection_name: str = "evidenceguard_corpus", persist_directory: Optional[str] = None):
        self.persist_dir = persist_directory or str(settings.CHROMA_PERSIST_DIR)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = EmbeddingService.get_instance()

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = []
        for c in chunks:
            clean_meta = {}
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif v is None:
                    continue
                else:
                    clean_meta[k] = str(v)
            clean_meta["doc_id"] = c.doc_id
            clean_meta["chunk_id"] = c.id
            metadatas.append(clean_meta)
            
        embeddings = self.embedder.embed_documents(texts)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[ScoredPassage]:
        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(1, self.collection.count())),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        passages: List[ScoredPassage] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return passages
            
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for p_id, text, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns cosine distance (1 - cosine_similarity), convert to similarity
            cosine_sim = max(0.0, 1.0 - dist)
            passages.append(ScoredPassage(
                id=p_id,
                doc_id=meta.get("doc_id", "unknown"),
                title=meta.get("title", meta.get("doc_id", "Document")),
                text=text,
                dense_score=cosine_sim,
                date=meta.get("date"),
                source=meta.get("source"),
                metadata=meta
            ))
        return passages

    def delete_by_doc_id(self, doc_id: str):
        self.collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self.collection.count()
        
    def reset(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
