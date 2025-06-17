#!/bin/bash
set -e  # Exit on any error

echo "🔧 Starting build and deployment process with bulletproof persistence..."

# Configuration
PERSISTENT_DATA_DIR="$HOME/rag-persistent-data"
CONTAINER_NAME="redhat-rag-gpu"
IMAGE_NAME="localhost/redhat-rag-gpu:latest"

# Stop any existing containers
echo "📦 Cleaning up existing containers..."
podman stop $CONTAINER_NAME 2>/dev/null || true
podman rm $CONTAINER_NAME 2>/dev/null || true

# Set up persistent directories (bulletproof approach)
echo "📁 Setting up bulletproof persistent storage..."
mkdir -p "$PERSISTENT_DATA_DIR"/{vectordb,documents,models,logs,backups}
chmod 777 "$PERSISTENT_DATA_DIR"/{vectordb,documents,models,logs}

# Backup existing data from old location if it exists
if [ -d "data/persistent/vectordb" ] && [ "$(ls -A data/persistent/vectordb 2>/dev/null)" ]; then
    echo "📑 Migrating data from old location..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Create backup
    tar -czf "$PERSISTENT_DATA_DIR/backups/migration_backup_${timestamp}.tar.gz" data/persistent/ 2>/dev/null || true
    
    # Copy data to new location
    cp -r data/persistent/vectordb/* "$PERSISTENT_DATA_DIR/vectordb/" 2>/dev/null || true
    cp -r data/persistent/documents/* "$PERSISTENT_DATA_DIR/documents/" 2>/dev/null || true
    
    echo "✅ Data migrated from data/persistent/ to $PERSISTENT_DATA_DIR/"
fi

# Check if container image exists
echo "🔍 Checking for existing container image..."
if ! podman images | grep -q "$IMAGE_NAME"; then
    echo "🏗️ Building container image..."
    if [ -f "Containerfile" ]; then
        podman build -t $IMAGE_NAME -f Containerfile .
    elif [ -f "Dockerfile" ]; then
        podman build -t $IMAGE_NAME -f Dockerfile .
    else
        echo "❌ Error: No Containerfile or Dockerfile found. Please create one first."
        exit 1
    fi
else
    echo "✅ Container image already exists"
fi

# Function to deploy with GPU method
deploy_with_method() {
    local method="$1"
    local gpu_args="$2"
    
    echo "Attempting deployment: $method"
    
    if podman run -d \
      --name $CONTAINER_NAME \
      --publish 127.0.0.1:8080:8080 \
      --volume "$PERSISTENT_DATA_DIR/documents:/app/documents:Z" \
      --volume "$PERSISTENT_DATA_DIR/vectordb:/app/vectordb:Z" \
      --volume "$PERSISTENT_DATA_DIR/models:/app/models:Z" \
      --volume "$PERSISTENT_DATA_DIR/logs:/app/logs:Z" \
      --security-opt=label=disable \
      --env DOCUMENTS_DIR=/app/documents \
      --env CHROMA_DB_PATH=/app/vectordb \
      --env SENTENCE_TRANSFORMERS_HOME=/app/models \
      $gpu_args \
      $IMAGE_NAME; then
        echo "✅ Container deployed with $method"
        return 0
    else
        echo "⚠️ $method failed"
        return 1
    fi
}

# Deploy container - trying different GPU access methods
echo "🚀 Deploying container with bulletproof persistence..."

# Try different GPU methods
if deploy_with_method "GPU Method 1 (CDI all GPUs)" "--device nvidia.com/gpu=all"; then
    :
elif deploy_with_method "GPU Method 2 (CDI GPU 0)" "--device nvidia.com/gpu=0"; then
    :
elif deploy_with_method "GPU Method 3 (Legacy devices)" "--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm"; then
    :
elif deploy_with_method "CPU Fallback" ""; then
    echo "⚠️ Note: Container will run in CPU mode (no GPU acceleration)"
else
    echo "❌ All deployment methods failed"
    podman logs $CONTAINER_NAME 2>/dev/null || echo "No logs available"
    exit 1
fi

# Wait for container to initialize
echo "⏳ Waiting for container to initialize..."
sleep 5

# Verify container is running
echo "🔍 Verifying container status..."
if ! podman ps | grep -q $CONTAINER_NAME; then
    echo "❌ Container failed to start"
    podman logs $CONTAINER_NAME 2>/dev/null || echo "No logs available"
    exit 1
fi

echo "✅ Container is running"

# Test persistence is working
echo "🧪 Testing persistence setup..."

# Create test file on host
test_file="$PERSISTENT_DATA_DIR/vectordb/persistence_test_$(date +%s).txt"
echo "persistence-test" > "$test_file"

# Check if container can see it
if podman exec $CONTAINER_NAME cat "/app/vectordb/$(basename "$test_file")" >/dev/null 2>&1; then
    echo "✅ Host → Container persistence: WORKING"
    rm -f "$test_file"
else
    echo "❌ Host → Container persistence: FAILED"
    echo "⚠️ Volume mounts may not be working correctly"
fi

# Check for existing vector data
echo "🔍 Checking for existing vector embeddings..."
if [ -f "$PERSISTENT_DATA_DIR/vectordb/vectors.npy" ]; then
    echo "✅ Found existing vector embeddings:"
    ls -lh "$PERSISTENT_DATA_DIR/vectordb/"*.{npy,json} 2>/dev/null || true
    echo "📅 Vector data age:"
    stat "$PERSISTENT_DATA_DIR/vectordb/vectors.npy" | grep Modify || true
    
    echo "⏳ Waiting for container to load existing data..."
    sleep 10
    
    # Check if container loaded existing data
    if podman logs $CONTAINER_NAME | grep -q "Loading existing vector store data"; then
        echo "🎉 SUCCESS: Container loaded existing embeddings!"
        echo "⚡ Fast startup achieved - no reprocessing needed"
    else
        echo "⚠️ WARNING: Container did not load existing data"
        echo "📋 Check logs for issues:"
        podman logs $CONTAINER_NAME | grep -i "vector\|loading\|processing" | tail -5
    fi
else
    echo "ℹ️ No existing vector data found - will be created on first document processing"
fi

# Check GPU status
echo "🔍 Checking GPU access..."
if podman exec $CONTAINER_NAME nvidia-smi >/dev/null 2>&1; then
    echo "✅ GPU access confirmed"
    podman exec $CONTAINER_NAME nvidia-smi | head -n 15
else
    echo "⚠️ GPU not accessible - running in CPU mode"
fi

# Show container logs (last 20 lines)
echo ""
echo "📋 Container startup logs (last 20 lines):"
podman logs $CONTAINER_NAME | tail -n 20

# Final status
echo ""
echo "🎯 Deployment Status:"
echo "🌐 Web Interface: http://localhost:8080"
echo "📊 API Documentation: http://localhost:8080/docs"
echo "💓 Health Check: http://localhost:8080/health"
echo "🖥️ GPU Info: http://localhost:8080/gpu-info"
echo ""
echo "💾 Persistent Data Location: $PERSISTENT_DATA_DIR"
echo "📊 Vector Embeddings: $PERSISTENT_DATA_DIR/vectordb/"
echo "📄 Documents: $PERSISTENT_DATA_DIR/documents/"
echo ""
echo "🔧 Useful Commands:"
echo "  View logs: podman logs -f $CONTAINER_NAME"
echo "  Enter container: podman exec -it $CONTAINER_NAME /bin/bash"
echo "  Stop container: podman stop $CONTAINER_NAME"
echo "  Check data: ls -la $PERSISTENT_DATA_DIR/vectordb/"
echo ""

# Show current data status
echo "📈 Current persistent data status:"
echo "Vector DB files:"
ls -la "$PERSISTENT_DATA_DIR/vectordb/" 2>/dev/null | head -10 || echo "  (empty - will be created on first document processing)"
echo ""
echo "Documents:"
ls -la "$PERSISTENT_DATA_DIR/documents/" 2>/dev/null | head -5 || echo "  (empty - upload PDFs via web interface)"

echo ""
echo "🧪 To test persistence:"
echo "  1. Upload a document and wait for processing to complete"
echo "  2. Run: podman stop $CONTAINER_NAME && podman rm $CONTAINER_NAME"
echo "  3. Rerun this script - should show 'Container loaded existing embeddings!'"
echo ""
echo "✅ Deployment with bulletproof persistence complete!"
