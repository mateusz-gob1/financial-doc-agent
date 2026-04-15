FROM python:3.11-slim

WORKDIR /app

# System deps (for PyMuPDF binary wheel and pdfplumber)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so cold-starts don't hit the network
RUN python -c "\
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction; \
SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')"

# Copy application code
COPY agents/   agents/
COPY api/      api/
COPY schemas/  schemas/
COPY tools/    tools/
COPY frontend/ frontend/
COPY data/red_flags/ data/red_flags/
COPY data/demo_data.json data/demo_data.json

# Create writable dirs for runtime state (chroma_db + history.db)
RUN mkdir -p data/chroma_db

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
