# Multi-stage / lightweight Python 3.11 base image for RailWay AI Chatbot
FROM python:3.11-slim

# Set environment variables to optimize Python execution in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set container working directory
WORKDIR /app

# Install system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-optimized PyTorch and runtime packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, knowledge base, dataset, and trained model artifacts
COPY app.py inference.py knowledge_base.json ./
COPY model_artifacts/ ./model_artifacts/
COPY data/ ./data/

# Create a non-privileged user and switch to it for runtime security
RUN useradd -u 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose default Streamlit web port
EXPOSE 8501

# Healthcheck targeting Streamlit core health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py"]
