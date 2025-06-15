# Base image with CUDA support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        curl \
        build-essential \
        python3-pip \
        python3-dev \
        git \
        && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 (for Ubuntu 22.04)
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-dev python3.12-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symlink for pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Working directory
WORKDIR /app

# Create non-root user
RUN useradd -m appuser

# Copy application files
COPY main.py .
COPY .env .

# Create static directory and copy frontend files
RUN mkdir -p /app/static

# Copy frontend to container
COPY static/index.html /app/static/index.html

# Install GPU-enabled PyTorch and dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir setuptools wheel && \
    # Install PyTorch with CUDA support
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    # Install other dependencies
    pip3 install --no-cache-dir fastapi uvicorn pydantic && \
    pip3 install --no-cache-dir numpy scikit-learn && \
    pip3 install --no-cache-dir sentence-transformers && \
    pip3 install --no-cache-dir PyPDF2 python-multipart python-dotenv

# Switch to root to set up directories
USER 0

# Create directories with proper permissions
RUN mkdir -p /app/documents /app/vectordb /app/models /app/logs /tmp/uploads && \
    chmod -R 777 /app/documents /app/vectordb /app/models /app/logs /tmp/uploads && \
    chown -R appuser:appuser /app/documents /app/vectordb /app/models /app/logs /tmp/uploads

# Ensure static directory has proper permissions
RUN chmod -R 755 /app/static && \
    chown -R appuser:appuser /app/static

# Switch back to non-root user
USER appuser

# Set environment variables
ENV DOCUMENTS_DIR=/app/documents \
    CHROMA_DB_PATH=/app/vectordb \
    PYTHONPATH=/app \
    SENTENCE_TRANSFORMERS_HOME=/app/models

# Expose port
EXPOSE 8080

# Command to run the application
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
