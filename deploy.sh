#!/usr/bin/env bash
set -e

PLATFORM=${1:-"render"}
GIT_REMOTE=${2:-""}

echo "=========================================================="
echo "   EvidenceGuard RAG: Cloud Deployment Helper             "
echo "=========================================================="

if [ -n "$GIT_REMOTE" ]; then
    echo "Configuring git remote: $GIT_REMOTE"
    git remote add origin "$GIT_REMOTE" 2>/dev/null || git remote set-url origin "$GIT_REMOTE"
    git branch -M main
    git push -u origin main
    echo "Repository pushed to remote origin!"
fi

case "$PLATFORM" in
    render)
        echo "To deploy to Render:"
        echo "1. Push to GitHub: git push origin main"
        echo "2. Visit https://dashboard.render.com -> New -> Blueprint"
        echo "3. Connect repository; Render will configure automatically using render.yaml"
        ;;
    railway)
        echo "To deploy to Railway:"
        echo "1. Visit https://railway.app -> New Project -> Deploy from GitHub repo"
        echo "2. Auto-builds via Dockerfile and railway.json"
        ;;
    fly)
        echo "Deploying via Fly.io..."
        fly launch --copy-config
        fly deploy
        ;;
    aws)
        echo "AWS Deployment via ECR & App Runner..."
        aws ecr create-repository --repository-name evidenceguard-rag || true
        ;;
    huggingface)
        echo "Hugging Face Spaces deployment:"
        echo "1. Create Docker Space on Hugging Face"
        echo "2. git remote add space <YOUR_SPACE_GIT_URL>"
        echo "3. git push space main"
        ;;
    *)
        echo "See DEPLOYMENT.md for all deployment targets."
        ;;
esac
