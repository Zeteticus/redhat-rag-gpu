#!/bin/bash
set -e  # Exit on any error

echo "🔧 Starting build and deployment process..."

# Stop any existing containers
echo "📦 Cleaning up existing containers..."
podman stop redhat-rag-gpu 2>/dev/null || true
podman rm redhat-rag-gpu 2>/dev/null || true

# Fix directory permissions issue
echo "📁 Setting up directories..."
if [ -d "data/vectordb" ]; then
    echo "ℹ️ data/vectordb already exists, handling permissions carefully"
    # Create a backup of existing vector data
    echo "📑 Creating backup of existing vector data..."
    mkdir -p data/backups
    timestamp=$(date +%Y%m%d_%H%M%S)
    tar -czf "data/backups/vectordb_backup_${timestamp}.tar.gz" data/vectordb 2>/dev/null || true
    
    # Instead of changing permissions, we'll use a fresh directory with correct permissions
#    echo "🔄 Creating fresh vectordb directory with correct permissions..."
#    mv data/vectordb "data/vectordb_old_${timestamp}"
#    mkdir -p data/vectordb
#    chmod 777 data/vectordb
    
    # Copy back important files if needed (optional)
    # cp -r "data/vectordb_old_${timestamp}"/* data/vectordb/ 2>/dev/null || true
else
#    echo "📁 Creating new vectordb directory..."
#    mkdir -p data/vectordb
    chmod 777 data/vectordb
fi

# Make sure other required directories exist
mkdir -p data/logs documents
chmod 777 data/logs documents

# Check if container image exists
echo "🔍 Checking for existing container image..."
if ! podman images | grep -q "localhost/redhat-rag-gpu"; then
    echo "🏗️ Building container image..."
    if [ -f "Containerfile" ]; then
        podman build -t localhost/redhat-rag-gpu:latest -f Containerfile .
    elif [ -f "Dockerfile" ]; then
        podman build -t localhost/redhat-rag-gpu:latest -f Dockerfile .
    else
        echo "❌ Error: No Containerfile or Dockerfile found. Please create one first."
        exit 1
    fi
else
    echo "✅ Container image already exists"
fi

# Deploy container - trying different GPU access methods
echo "🚀 Deploying container..."

# Method 1: Simple device mounting
echo "Attempting to deploy with GPU access (Method 1)..."
if podman run -d \
  --name redhat-rag-gpu \
  --publish 127.0.0.1:8080:8080 \
  --volume "$(pwd)/documents:/app/documents:Z" \
  --volume "$(pwd)/data/vectordb:/app/vectordb:Z" \
  --volume "$(pwd)/data/logs:/app/logs:Z" \
  --security-opt=label=disable \
  --device nvidia.com/gpu=GPU-75ed8236-f727-c07f-4634-b221094255c8 \
  localhost/redhat-rag-gpu:latest; then
    echo "✅ Container deployed with GPU access (Method 1)"
else
    echo "⚠️ Method 1 failed, trying Method 2..."
    # Method 2: Comprehensive device mapping
    if podman run -d \
      --name redhat-rag-gpu \
      --publish 127.0.0.1:8080:8080 \
      --volume "$(pwd)/documents:/app/documents:Z" \
      --volume "$(pwd)/data/vectordb:/app/vectordb:Z" \
      --volume "$(pwd)/data/logs:/app/logs:Z" \
      --security-opt=label=disable \
      --device nvidia.com/gpu=GPU-75ed8236-f727-c07f-4634-b221094255c8 \
      localhost/redhat-rag-gpu:latest; then
        echo "✅ Container deployed with GPU access (Method 2)"
    else
        echo "⚠️ Method 2 failed, trying Method 3 (CPU fallback)..."
        # Method 3: CPU fallback with environment variables
        if podman run -d \
          --name redhat-rag-gpu \
          --security-opt=label=disable \
          --publish 127.0.0.1:8080:8080 \
	  --device nvidia.com/gpu=GPU-75ed8236-f727-c07f-4634-b221094255c8
          --volume "$(pwd)/documents:/app/documents:Z" \
          --volume "$(pwd)/data/vectordb:/app/vectordb:Z" \
          --volume "$(pwd)/data/logs:/app/logs:Z" \
          localhost/redhat-rag-gpu:latest; then
            echo "✅ Container deployed without GPU access (CPU fallback)"
            echo "⚠️ Note: Container will run, but will use CPU instead of GPU"
        else
            echo "❌ All deployment methods failed"
            exit 1
        fi
    fi
fi

# Verify container is running
echo "🔍 Checking container status..."
if podman ps | grep -q redhat-rag-gpu; then
    echo "✅ Container is running"
    echo "📋 Container logs (first 20 lines):"
    podman logs redhat-rag-gpu | head -n 20
    echo ""
    echo "🌐 Access the application at: http://localhost:8080"
    echo "📊 API documentation at: http://localhost:8080/docs"
    echo "💓 Health check at: http://localhost:8080/health"
    echo "🖥️ Check GPU access with: podman exec redhat-rag-gpu nvidia-smi"
else
    echo "❌ Container failed to start"
    podman logs redhat-rag-gpu
    exit 1
fi
