#!/bin/bash
set -e  # Exit on any error

echo "🔧 Starting improved deployment process..."

# Aggressively clean up any existing containers
echo "🧹 Cleaning up existing containers..."
podman stop redhat-rag-gpu 2>/dev/null || true
podman rm -f redhat-rag-gpu 2>/dev/null || true
sleep 2  # Give podman time to fully clean up

# Check NVIDIA setup
echo "🔍 Checking NVIDIA setup..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver information:"
    nvidia-smi | head -n 5
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected $GPU_COUNT GPU(s)"
else
    echo "⚠️ nvidia-smi not found - GPU functionality may not be available"
    GPU_COUNT=0
fi

# Check if SELinux is enabled
echo "🔍 Checking SELinux status..."
if command -v getenforce &> /dev/null; then
    SELINUX_STATUS=$(getenforce)
    echo "SELinux status: $SELINUX_STATUS"
else
    echo "⚠️ SELinux tools not found"
    SELINUX_STATUS="Unknown"
fi

# Check if NVIDIA container runtime hook exists
echo "🔍 Checking NVIDIA container hooks..."
if [ -f "/usr/bin/nvidia-container-runtime-hook" ]; then
    echo "NVIDIA container runtime hook found"
else
    echo "⚠️ NVIDIA container runtime hook not found"
fi

# Make sure directories exist with correct permissions
echo "📁 Ensuring directories exist with correct permissions..."
mkdir -p data/vectordb data/logs documents
chmod 777 data/vectordb data/logs documents

# Check if vectordb directory is empty
if [ "$(ls -A data/vectordb 2>/dev/null)" ]; then
    echo "ℹ️ Note: vectordb directory contains existing data"
else
    echo "ℹ️ vectordb directory is empty"
fi

echo "🚀 Deploying container..."

# Try CPU method first to ensure basic functionality
echo "Attempting to deploy with CPU only (safe method)..."
if podman run -d \
  --name redhat-rag-gpu \
  --security-opt=label=disable \
  --publish 127.0.0.1:8080:8080 \
  --volume "$(pwd)/documents:/app/documents:Z" \
  --volume "$(pwd)/data/vectordb:/app/vectordb:Z" \
  --volume "$(pwd)/data/logs:/app/logs:Z" \
  --replace \
  localhost/redhat-rag-gpu:latest; then
    echo "✅ Container deployed in CPU mode"
    echo "ℹ️ Now attempting to upgrade to GPU mode..."
    
    # Get container ID
    CONTAINER_ID=$(podman ps -q -f name=redhat-rag-gpu)
    
    # Stop container for GPU upgrade
    podman stop redhat-rag-gpu
    podman rm -f redhat-rag-gpu
    
    # Now try with GPU - Method A (nvidia-container-toolkit)
    if [ $GPU_COUNT -gt 0 ]; then
        echo "Attempting GPU Method A (modern method with nvidia-container-toolkit)..."
        if podman run -d \
          --name redhat-rag-gpu \
          --security-opt=label=disable \
          --publish 127.0.0.1:8080:8080 \
          --volume "$(pwd)/documents:/app/documents:Z" \
          --volume "$(pwd)/data/vectordb:/app/vectordb:Z" \
          --volume "$(pwd)/data/logs:/app/logs:Z" \
          --device nvidia.com/gpu=all \
          --replace \
          localhost/redhat-rag-gpu:latest; then
            echo "✅ Container deployed with GPU access (Method A)"
            GPU_SUCCESS=true
        else
            echo "⚠️ GPU Method A failed, trying Method B..."
            GPU_SUCCESS=false
        fi
        
        # If Method A failed, try Method B (direct device mapping)
        if [ "$GPU_SUCCESS" = false ]; then
            echo "Attempting GPU Method B (direct device mapping)..."
            # Find all NVIDIA devices
            NVIDIA_DEVICES=""
            for i in $(find /dev -name "nvidia*"); do
                NVIDIA_DEVICES+=" --device $i:$i"
            done
            
            if podman run -d \
              --name redhat-rag-gpu \
              --security-opt=label=disable \
              --publish 127.0.0.1:8080:8080 \
              --volume "$(pwd)/documents:/app/documents:Z" \
              --volume "$(pwd)/data/vectordb:/app/vectordb:Z" \
              --volume "$(pwd)/data/logs:/app/logs:Z" \
              $NVIDIA_DEVICES \
              --replace \
              localhost/redhat-rag-gpu:latest; then
                echo "✅ Container deployed with GPU access (Method B)"
                GPU_SUCCESS=true
            else
                echo "⚠️ GPU Method B failed, falling back to CPU mode..."
                
                # Fall back to CPU mode
                podman run -d \
                  --name redhat-rag-gpu \
                  --security-opt=label=disable \
                  --publish 127.0.0.1:8080:8080 \
                  --volume "$(pwd)/documents:/app/documents:Z" \
                  --volume "$(pwd)/data/vectordb:/app/vectordb:Z" \
                  --volume "$(pwd)/data/logs:/app/logs:Z" \
                  --replace \
                  localhost/redhat-rag-gpu:latest
                  
                echo "✅ Container deployed without GPU access (CPU fallback)"
                echo "⚠️ Note: Container will run, but will use CPU instead of GPU"
            fi
        fi
    else
        echo "⚠️ No GPUs detected, using CPU mode only"
        
        # Redeploy in CPU mode
        podman run -d \
          --name redhat-rag-gpu \
          --security-opt=label=disable \
          --publish 127.0.0.1:8080:8080 \
          --volume "$(pwd)/documents:/app/documents:Z" \
          --volume "$(pwd)/data/vectordb:/app/vectordb:Z" \
          --volume "$(pwd)/data/logs:/app/logs:Z" \
          --replace \
          localhost/redhat-rag-gpu:latest
          
        echo "✅ Container deployed in CPU-only mode"
    fi
else
    echo "❌ Failed to deploy container even in CPU mode"
    echo "This indicates a more fundamental issue with the container or podman"
    exit 1
fi

# Verify container is running
echo "🔍 Checking container status..."
if podman ps | grep -q redhat-rag-gpu; then
    echo "✅ Container is running"
    
    # Give container a moment to initialize
    echo "⏳ Waiting for container to initialize..."
    sleep 5
    
    echo "📋 Container logs (last 20 lines):"
    podman logs redhat-rag-gpu | tail -n 20
    echo ""
    
    # Check if GPU is detected inside container
    echo "🔍 Checking for GPU inside container..."
    if podman exec redhat-rag-gpu nvidia-smi &> /dev/null; then
        echo "✅ GPU detected inside container!"
        podman exec redhat-rag-gpu nvidia-smi | head -n 5
    else
        echo "⚠️ GPU not detected inside container, running in CPU mode"
    fi
    
    echo ""
    echo "🌐 Access the application at: http://localhost:8080"
    echo "📊 API documentation at: http://localhost:8080/docs"
    echo "💓 Health check at: http://localhost:8080/health"
else
    echo "❌ Container failed to start"
    echo "📋 Container logs:"
    podman logs redhat-rag-gpu
    exit 1
fi
