# Multi-Format RAG Containerfile with GPU Support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for multi-format support
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        curl \
        build-essential \
        python3-pip \
        python3-dev \
        git \
        # Document processing dependencies
        libxml2-dev \
        libxslt1-dev \
        libffi-dev \
        libjpeg-dev \
        libpng-dev \
        zlib1g-dev \
        libfreetype6-dev \
        liblcms2-dev \
        libwebp-dev \
        tcl8.6-dev \
        tk8.6-dev \
        python3-tk \
        # For better file type detection
        file \
        libmagic1 \
        # For various document formats
        poppler-utils \
        pandoc \
        && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 (for Ubuntu 22.04)
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-dev python3.12-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symlink for pip and upgrade pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    python3.12 -m pip install --upgrade pip setuptools wheel

# Working directory
WORKDIR /app

# Create non-root user
RUN useradd -m appuser

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install GPU-enabled PyTorch and all dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    # Install PyTorch with CUDA support first
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    # Install other dependencies from requirements.txt
    pip3 install --no-cache-dir -r requirements.txt && \
    # Clean up pip cache
    pip3 cache purge

# Copy application files
COPY main.py .
COPY .env .

# Create static directory and copy frontend files
RUN mkdir -p /app/static
COPY static/ ./static/ 2>/dev/null || echo "No static directory found, skipping..."

# Switch to root to set up directories
USER 0

# Create directories with proper permissions
RUN mkdir -p /app/documents /app/vectordb /app/models /app/logs /tmp/uploads && \
    chmod -R 777 /app/documents /app/vectordb /app/models /app/logs /tmp/uploads && \
    chown -R appuser:appuser /app/documents /app/vectordb /app/models /app/logs /tmp/uploads

# Ensure static directory has proper permissions (if it exists)
RUN if [ -d "/app/static" ]; then \
        chmod -R 755 /app/static && \
        chown -R appuser:appuser /app/static; \
    fi

# Create VOLUME declarations for persistent data
VOLUME ["/app/vectordb", "/app/documents", "/app/models"]

# Switch back to non-root user
USER appuser

# Set environment variables
ENV DOCUMENTS_DIR=/app/documents \
    CHROMA_DB_PATH=/app/vectordb \
    PYTHONPATH=/app \
    SENTENCE_TRANSFORMERS_HOME=/app/models \
    # Multi-format processing settings
    MAX_FILE_SIZE=100 \
    CHUNK_SIZE=500 \
    CHUNK_OVERLAP=50

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Command to run the application
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
