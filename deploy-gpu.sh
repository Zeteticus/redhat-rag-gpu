#!/bin/bash
set -e  # Exit on any error

echo "🔧 Starting build and deployment process..."

# Stop any existing containers
echo "📦 Cleaning up existing containers..."
podman stop redhat-rag-gpu 2>/dev/null || true
podman rm redhat-rag-gpu 2>/dev/null || true

# Fix directory permissions issue
echo "📁 Setting up directories..."
if [ -d "data/persistent/vectordb" ]; then
    echo "ℹ️ data/persistent/vectordb already exists, handling permissions carefully"
    # Create a backup of existing vector data
#    echo "📑 Creating backup of existing vector data..."
#    mkdir -p data/backups
#    timestamp=$(date +%Y%m%d_%H%M%S)
#    tar -czf "data/backups/vectordb_backup_${timestamp}.tar.gz" data/vectordb 2>/dev/null || true
#    echo "✅ Backup created: data/backups/vectordb_backup_${timestamp}.tar.gz"
else
    echo "📁 Creating new data directories..."
    mkdir -p data/persistent/vectordb
fi

# Make sure all required directories exist with proper permissions
mkdir -p data/logs data/persistent/documents data/models
chmod 777 data/persistent/vectordb data/logs data/persistent/documents data/models

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

# Method 1: CDI with all GPUs
echo "Attempting to deploy with GPU access (Method 1: CDI all GPUs)..."
if podman run -d \
  --name redhat-rag-gpu \
  --publish 127.0.0.1:8080:8080 \
  --volume "$(pwd)/data/persistent/documents:/app/documents:Z" \
  --volume "$(pwd)/data/persistent/vectordb:/app/vectordb:Z" \
  --volume "$(pwd)/data/models:/app/models:Z" \
  --volume "$(pwd)/data/logs:/app/logs:Z" \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  localhost/redhat-rag-gpu:latest; then
    echo "✅ Container deployed with GPU access (Method 1)"
else
    echo "⚠️ Method 1 failed, trying Method 2..."
    
    # Method 2: Specific GPU device
    echo "Attempting GPU access with specific device..."
    if podman run -d \
      --name redhat-rag-gpu \
      --publish 127.0.0.1:8080:8080 \
      --volume "$(pwd)/data/persistent/documents:/app/documents:Z" \
      --volume "$(pwd)/data/persistent/vectordb:/app/vectordb:Z" \
      --volume "$(pwd)/data/models:/app/models:Z" \
      --volume "$(pwd)/data/logs:/app/logs:Z" \
      --security-opt=label=disable \
      --device nvidia.com/gpu=0 \
      localhost/redhat-rag-gpu:latest; then
        echo "✅ Container deployed with GPU access (Method 2)"
    else
        echo "⚠️ Method 2 failed, trying Method 3..."
        
        # Method 3: Legacy device mapping
        echo "Attempting legacy device mapping..."
        if podman run -d \
          --name redhat-rag-gpu \
          --publish 127.0.0.1:8080:8080 \
          --volume "$(pwd)/data/persistent/documents:/app/documents:Z" \
          --volume "$(pwd)/data/persistent/vectordb:/app/vectordb:Z" \
          --volume "$(pwd)/data/models:/app/models:Z" \
          --volume "$(pwd)/data/logs:/app/logs:Z" \
          --security-opt=label=disable \
          --device /dev/nvidia0 \
          --device /dev/nvidiactl \
          --device /dev/nvidia-uvm \
          localhost/redhat-rag-gpu:latest; then
            echo "✅ Container deployed with GPU access (Method 3)"
        else
            echo "⚠️ Method 3 failed, trying Method 4 (CPU fallback)..."
            
            # Method 4: CPU fallback
            if podman run -d \
              --name redhat-rag-gpu \
              --publish 127.0.0.1:8080:8080 \
              --volume "$(pwd)/data/persistent/documents:/app/documents:Z" \
              --volume "$(pwd)/data/data/persistent/vectordb:/app/vectordb:Z" \
              --volume "$(pwd)/data/models:/app/models:Z" \
              --volume "$(pwd)/data/logs:/app/logs:Z" \
              --security-opt=label=disable \
              localhost/redhat-rag-gpu:latest; then
                echo "✅ Container deployed without GPU access (CPU fallback)"
                echo "⚠️ Note: Container will run, but will use CPU instead of GPU"
            else
                echo "❌ All deployment methods failed"
                echo "📋 Last container logs:"
                podman logs redhat-rag-gpu 2>/dev/null || echo "No logs available"
                exit 1
            fi
        fi
    fi
fi

# Wait a moment for container to initialize
echo "⏳ Waiting for container to initialize..."
sleep 5

# Verify container is running
echo "🔍 Checking container status..."
if podman ps | grep -q redhat-rag-gpu; then
    echo "✅ Container is running"
    
    # Check GPU status
    echo "🔍 Checking GPU access..."
    if podman exec redhat-rag-gpu nvidia-smi >/dev/null 2>&1; then
        echo "✅ GPU access confirmed"
        podman exec redhat-rag-gpu nvidia-smi | head -n 15
    else
        echo "⚠️ GPU not accessible - running in CPU mode"
    fi
    
    echo ""
    echo "📋 Container logs (last 20 lines):"
    podman logs redhat-rag-gpu | tail -n 20
    echo ""
    echo "🌐 Access the application at: http://localhost:8080"
    echo "📊 API documentation at: http://localhost:8080/docs"
    echo "💓 Health check at: http://localhost:8080/health"
    echo "📊 GPU info endpoint: http://localhost:8080/gpu-info"
    echo ""
    echo "🔧 Useful commands:"
    echo "  View logs: podman logs -f redhat-rag-gpu"
    echo "  Enter container: podman exec -it redhat-rag-gpu /bin/bash"
    echo "  Stop container: podman stop redhat-rag-gpu"
    echo "  Check GPU: podman exec redhat-rag-gpu nvidia-smi"
else
    echo "❌ Container failed to start"
    echo "📋 Container logs:"
    podman logs redhat-rag-gpu 2>/dev/null || echo "No logs available"
    exit 1
fi

# Show current data status
echo ""
echo "📈 Current data status:"
echo "Vector DB files:"
ls -la data/vectordb/ 2>/dev/null || echo "  (empty - will be created on first document processing)"
echo ""
echo "Documents:"
ls -la data/persistent/documents/ 2>/dev/null || echo "  (empty - upload PDFs via web interface)"
