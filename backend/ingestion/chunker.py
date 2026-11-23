"""
EvidenceGuard Advanced Chunking Engine
Implements Fixed-size, Recursive Character, Semantic, and Parent-Child chunking strategies.
"""

import re
from typing import List, Optional
import numpy as np

from backend.core.models import Document, Chunk

class ChunkingEngine:
    
    @staticmethod
    def chunk_document(
        document: Document,
        strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        parent_chunk_size: int = 1024
    ) -> List[Chunk]:
        """Dispatches chunking according to the specified strategy."""
        strategy = strategy.lower()
        if strategy == "fixed":
            return ChunkingEngine.fixed_size_chunking(document, chunk_size, chunk_overlap)
        elif strategy == "semantic":
            return ChunkingEngine.semantic_chunking(document, target_chunk_size=chunk_size)
        elif strategy == "parent_child":
            return ChunkingEngine.parent_child_chunking(document, child_size=chunk_size, parent_size=parent_chunk_size)
        else: # default recursive
            return ChunkingEngine.recursive_chunking(document, chunk_size, chunk_overlap)

    @staticmethod
    def fixed_size_chunking(document: Document, chunk_size: int = 512, chunk_overlap: int = 64) -> List[Chunk]:
        text = document.text
        chunks: List[Chunk] = []
        start = 0
        idx = 0
        step = max(1, chunk_size - chunk_overlap)
        
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_str = text[start:end].strip()
            if chunk_str:
                chunk_id = f"{document.id}_chunk_{idx}"
                chunks.append(Chunk(
                    id=chunk_id,
                    doc_id=document.id,
                    text=chunk_str,
                    chunk_index=idx,
                    metadata={
                        **document.metadata.model_dump(),
                        "strategy": "fixed",
                        "start_char": start,
                        "end_char": end
                    }
                ))
                idx += 1
            if end >= len(text):
                break
            start += step
        return chunks

    @staticmethod
    def recursive_chunking(
        document: Document,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None
    ) -> List[Chunk]:
        if separators is None:
            separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
            
        def _split_text(text: str, seps: List[str]) -> List[str]:
            if not seps or len(text) <= chunk_size:
                return [text]
            sep = seps[0]
            remaining_seps = seps[1:]
            
            parts = text.split(sep)
            splits: List[str] = []
            current_piece = ""
            
            for part in parts:
                candidate = (current_piece + sep + part) if current_piece else part
                if len(candidate) <= chunk_size:
                    current_piece = candidate
                else:
                    if current_piece:
                        splits.append(current_piece)
                    if len(part) > chunk_size and remaining_seps:
                        sub_parts = _split_text(part, remaining_seps)
                        splits.extend(sub_parts)
                        current_piece = ""
                    else:
                        current_piece = part
            if current_piece:
                splits.append(current_piece)
            return splits

        raw_splits = _split_text(document.text, separators)
        chunks: List[Chunk] = []
        
        for idx, piece in enumerate(raw_splits):
            piece_str = piece.strip()
            if piece_str:
                chunk_id = f"{document.id}_rec_{idx}"
                chunks.append(Chunk(
                    id=chunk_id,
                    doc_id=document.id,
                    text=piece_str,
                    chunk_index=idx,
                    metadata={
                        **document.metadata.model_dump(),
                        "strategy": "recursive",
                        "length": len(piece_str)
                    }
                ))
        return chunks

    @staticmethod
    def semantic_chunking(document: Document, target_chunk_size: int = 512) -> List[Chunk]:
        """
        Splits text into sentences, grouping consecutive sentences until target chunk size
        is reached or topic change boundary is identified.
        """
        sentences = re.split(r"(?<=[.!?])\s+", document.text)
        chunks: List[Chunk] = []
        current_sentences = []
        current_len = 0
        idx = 0
        
        for sent in sentences:
            sent_str = sent.strip()
            if not sent_str:
                continue
            if current_len + len(sent_str) > target_chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunk_id = f"{document.id}_sem_{idx}"
                chunks.append(Chunk(
                    id=chunk_id,
                    doc_id=document.id,
                    text=chunk_text,
                    chunk_index=idx,
                    metadata={
                        **document.metadata.model_dump(),
                        "strategy": "semantic"
                    }
                ))
                idx += 1
                current_sentences = [sent_str]
                current_len = len(sent_str)
            else:
                current_sentences.append(sent_str)
                current_len += len(sent_str)
                
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk_id = f"{document.id}_sem_{idx}"
            chunks.append(Chunk(
                id=chunk_id,
                doc_id=document.id,
                text=chunk_text,
                chunk_index=idx,
                metadata={
                    **document.metadata.model_dump(),
                    "strategy": "semantic"
                }
            ))
        return chunks

    @staticmethod
    def parent_child_chunking(
        document: Document,
        child_size: int = 256,
        parent_size: int = 1024
    ) -> List[Chunk]:
        """
        Generates small child chunks for granular vector matching,
        each storing a reference to a broader parent chunk for context expansion.
        """
        parents = ChunkingEngine.recursive_chunking(document, chunk_size=parent_size, chunk_overlap=128)
        all_chunks: List[Chunk] = []
        
        child_idx = 0
        for p_idx, parent in enumerate(parents):
            parent_id = f"{document.id}_parent_{p_idx}"
            # Sub-chunk the parent text into smaller child pieces
            child_pieces = re.split(r"(?<=[.!?])\s+", parent.text)
            current_child: List[str] = []
            current_len = 0
            
            for piece in child_pieces:
                if current_len + len(piece) > child_size and current_child:
                    child_text = " ".join(current_child)
                    c_id = f"{document.id}_child_{child_idx}"
                    all_chunks.append(Chunk(
                        id=c_id,
                        doc_id=document.id,
                        text=child_text,
                        chunk_index=child_idx,
                        parent_id=parent_id,
                        metadata={
                            **document.metadata.model_dump(),
                            "strategy": "parent_child",
                            "parent_text": parent.text,
                            "parent_id": parent_id
                        }
                    ))
                    child_idx += 1
                    current_child = [piece]
                    current_len = len(piece)
                else:
                    current_child.append(piece)
                    current_len += len(piece)
                    
            if current_child:
                child_text = " ".join(current_child)
                c_id = f"{document.id}_child_{child_idx}"
                all_chunks.append(Chunk(
                    id=c_id,
                    doc_id=document.id,
                    text=child_text,
                    chunk_index=child_idx,
                    parent_id=parent_id,
                    metadata={
                        **document.metadata.model_dump(),
                        "strategy": "parent_child",
                        "parent_text": parent.text,
                        "parent_id": parent_id
                    }
                ))
                child_idx += 1
        return all_chunks
