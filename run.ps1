Write-Host "Starting EvidenceGuard RAG Server on http://localhost:8000 ..." -ForegroundColor Cyan
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
