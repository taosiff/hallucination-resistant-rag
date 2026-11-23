# EvidenceGuard Cloud Deployment Automation Script (PowerShell)
param (
    [string]$Platform = "render",  # render, railway, fly, aws, huggingface
    [string]$GitRemote = ""
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   EvidenceGuard RAG: Cloud Deployment Helper             " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if ($GitRemote -ne "") {
    Write-Host "Configuring git remote: $GitRemote" -ForegroundColor Green
    git remote add origin $GitRemote 2>$null
    git remote set-url origin $GitRemote
    git branch -M main
    git push -u origin main
    Write-Host "Repository pushed to remote origin!" -ForegroundColor Green
}

switch ($Platform.ToLower()) {
    "render" {
        Write-Host "To deploy to Render:" -ForegroundColor Yellow
        Write-Host "1. Push your repo to GitHub."
        Write-Host "2. Go to https://dashboard.render.com -> New -> Blueprint."
        Write-Host "3. Connect this repository; Render will automatically detect render.yaml."
    }
    "railway" {
        Write-Host "To deploy to Railway:" -ForegroundColor Yellow
        Write-Host "1. Go to https://railway.app -> New Project -> Deploy from GitHub repo."
        Write-Host "2. Railway will automatically build via Dockerfile & railway.json."
    }
    "fly" {
        Write-Host "To deploy to Fly.io:" -ForegroundColor Yellow
        Write-Host "Run: fly launch (uses fly.toml)"
        Write-Host "Run: fly deploy"
    }
    "aws" {
        Write-Host "To deploy to AWS App Runner / ECR:" -ForegroundColor Yellow
        Write-Host "Run: aws ecr create-repository --repository-name evidenceguard-rag"
        Write-Host "Run: docker build -t evidenceguard-rag ."
    }
    "huggingface" {
        Write-Host "To deploy to Hugging Face Spaces:" -ForegroundColor Yellow
        Write-Host "1. Create a new Space at https://huggingface.co/new-space (SDK: Docker)."
        Write-Host "2. git remote add space <YOUR_SPACE_GIT_URL>"
        Write-Host "3. git push space main"
    }
    default {
        Write-Host "Unsupported platform. See DEPLOYMENT.md for step-by-step guides." -ForegroundColor Red
    }
}
