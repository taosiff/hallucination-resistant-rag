# EvidenceGuard RAG: Production Cloud Deployment Guide

This guide covers deploying **EvidenceGuard RAG** to cloud environments so it runs independently on a public or private server without needing your local machine.

---

## 1. Cloud Deployment Options At-a-Glance

| Platform | Best For | Setup Complexity | Free / Low-Cost Tier | Key Config File |
| :--- | :--- | :--- | :--- | :--- |
| **Render** | Fastest 1-click cloud setup | ⭐ Easy | Yes (Free / $7/mo) | `render.yaml` |
| **Railway** | Quick container deployment | ⭐ Easy | Yes ($5 credit / usage) | `railway.json` |
| **Hugging Face Spaces** | Free ML/AI hosting with public demo | ⭐ Easy | 100% Free CPU / GPU | `Dockerfile` |
| **AWS App Runner** | Production enterprise scaling | ⭐⭐ Moderate | Pay-as-you-go | `Dockerfile` + AWS CLI |
| **Cloud VPS (DigitalOcean/Linode)** | Full sovereignty & persistent data | ⭐⭐ Moderate | $4 - $12/mo | `docker-compose.yml` |

---

## 2. Option 1: Render (Recommended for Simplicity)

Render automatically builds your repository using the included `render.yaml` blueprint.

### Step 1: Push your local repo to GitHub
```powershell
# In PowerShell:
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/EvidenceGuard-RAG.git
git push -u origin main
```

### Step 2: Connect to Render
1. Go to [https://dashboard.render.com](https://dashboard.render.com) and log in.
2. Click **New +** -> **Blueprint**.
3. Select your GitHub repository.
4. Render automatically detects `render.yaml` and deploys:
   - Python 3.12 environment
   - FastAPI REST API + Built-in React UI dashboard
   - Persistent 10GB disk for ChromaDB vector embeddings
5. Your service is live with free automatic SSL at:  
   `https://evidenceguard-rag.onrender.com`

---

## 3. Option 2: Railway

1. Push your repo to GitHub (as in Option 1).
2. Go to [https://railway.app](https://railway.app).
3. Click **New Project** -> **Deploy from GitHub repo** -> Select `EvidenceGuard-RAG`.
4. Railway detects `Dockerfile` and `railway.json` and builds the container automatically.
5. In Railway settings, click **Generate Domain** to get your public HTTPS URL.

---

## 4. Option 3: Hugging Face Spaces (100% Free Cloud Hosting)

Hugging Face Spaces provides free hosting for AI/ML containers:
1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
2. Name your space (e.g. `evidenceguard-rag`).
3. Select **Space SDK: Docker** (Blank).
4. Hugging Face will give you a Git remote URL (e.g. `https://huggingface.co/spaces/<USER>/evidenceguard-rag`).
5. In your local terminal:
   ```powershell
   git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/evidenceguard-rag
   git push space main
   ```
6. Hugging Face builds your Docker container and serves the interactive dashboard and API publicly at:  
   `https://huggingface.co/spaces/<YOUR_USERNAME>/evidenceguard-rag`

---

## 5. Option 4: AWS Deployment (AWS App Runner / ECS)

Because the AWS CLI is already installed on your workstation, you can deploy to AWS container services:

### Step 1: Authenticate AWS CLI
```powershell
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and default region (e.g., us-east-1)
```

### Step 2: Build & Push Docker Image to Amazon ECR
```powershell
# 1. Start Docker Desktop on your workstation
# 2. Authenticate Docker with Amazon ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# 3. Create ECR repository
aws ecr create-repository --repository-name evidenceguard-rag --region us-east-1

# 4. Tag and push container image
docker build -t evidenceguard-rag .
docker tag evidenceguard-rag:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/evidenceguard-rag:latest
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/evidenceguard-rag:latest
```

### Step 3: Deploy on AWS App Runner
In AWS Console, open **App Runner** -> **Create service** -> Select the ECR image URI -> Port `8000`.  
App Runner provisions a fully managed serverless container with load balancing, auto-scaling, and HTTPS.

---

## 6. Option 5: Self-Hosted Cloud VPS (DigitalOcean / Hetzner / AWS EC2)

For complete data sovereignty on any Linux server:
```bash
# 1. SSH into your cloud server
ssh root@your-server-ip

# 2. Clone repository
git clone https://github.com/<YOUR_USERNAME>/EvidenceGuard-RAG.git
cd EvidenceGuard-RAG

# 3. Launch with Docker Compose
docker compose up -d --build
```
Your service will be accessible on port 8000 (and MLflow on port 5000).

---

## 7. Production Environment Configuration

In any cloud platform's Environment Variables dashboard, configure:
```env
ENVIRONMENT=production
DEFAULT_LLM_PROVIDER=mock          # Or "gemini" / "openai"
GEMINI_API_KEY=your_gemini_key     # (Optional)
OPENAI_API_KEY=your_openai_key     # (Optional)
ENABLE_MLFLOW=false                # Set to true if running separate MLflow instance
```
