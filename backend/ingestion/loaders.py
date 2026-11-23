"""
EvidenceGuard Multi-Format Document Ingestion Loaders
Extracts text, metadata, cleans content, and normalizes documents.
"""

import hashlib
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
import pypdf
import docx
import csv

from backend.core.models import Document, DocumentMetadata

def sanitize_text(text: str) -> str:
    """
    Sanitize raw extracted text:
    - Normalizes unicode spaces and control characters
    - Strips malicious prompt injection markers
    - Replaces consecutive whitespace
    """
    if not text:
        return ""
    # Strip dangerous injection phrases / tags
    text = re.sub(r"(?i)<\s*system\s*>", "[sanitized_tag]", text)
    text = re.sub(r"(?i)<\s*/\s*system\s*>", "[/sanitized_tag]", text)
    text = re.sub(r"(?i)ignore\s+all\s+previous\s+instructions", "[sanitized_prompt_override]", text)
    # Normalize whitespaces
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

class DocumentLoader:
    """Universal document loader for PDF, DOCX, TXT, MD, CSV"""
    
    @staticmethod
    def load_from_file(file_path: str, doc_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = path.suffix.lower()
        if ext == ".pdf":
            raw_text = DocumentLoader._read_pdf(path)
        elif ext in [".docx", ".doc"]:
            raw_text = DocumentLoader._read_docx(path)
        elif ext == ".csv":
            raw_text = DocumentLoader._read_csv(path)
        elif ext in [".txt", ".md", ".markdown", ".json"]:
            raw_text = DocumentLoader._read_text(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        cleaned_text = sanitize_text(raw_text)
        if not cleaned_text:
            raise ValueError(f"Extracted content is empty for: {path.name}")
            
        doc_id = doc_id or f"doc_{path.stem}_{compute_hash(cleaned_text)[:8]}"
        content_hash = compute_hash(cleaned_text)
        
        meta = DocumentMetadata(
            doc_id=doc_id,
            title=metadata.get("title", path.stem) if metadata else path.stem,
            source=str(path),
            file_type=ext.replace(".", ""),
            date=metadata.get("date") if metadata else None,
            author=metadata.get("author") if metadata else None,
            version=metadata.get("version", 1) if metadata else 1,
            content_hash=content_hash,
            extra=metadata.get("extra", {}) if metadata else {}
        )
        return Document(id=doc_id, text=cleaned_text, metadata=meta)

    @staticmethod
    def _read_pdf(path: Path) -> str:
        text_parts = []
        reader = pypdf.PdfReader(str(path))
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)

    @staticmethod
    def _read_docx(path: Path) -> str:
        doc = docx.Document(str(path))
        return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    @staticmethod
    def _read_csv(path: Path) -> str:
        lines = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers:
                lines.append("Columns: " + ", ".join(headers))
            for row in reader:
                if any(row):
                    lines.append(" | ".join(row))
        return "\n".join(lines)

    @staticmethod
    def _read_text(path: Path) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
