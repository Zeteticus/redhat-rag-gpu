#!/bin/bash
set -e  # Exit on any error

echo "🔧 Starting enhanced multi-format RAG build and deployment..."

# Configuration
PERSISTENT_DATA_DIR="$HOME/rag-persistent-data"
CONTAINER_NAME="redhat-rag-gpu"
IMAGE_NAME="localhost/redhat-rag-gpu:latest"

# CRITICAL FIX: Robust container cleanup function
cleanup_containers() {
    echo "🧹 Performing thorough container cleanup..."
    
    # Stop all containers with this name
    for container in $(podman ps -q --filter "name=${CONTAINER_NAME}"); do
        echo "Stopping container: $container"
        podman stop $container 2>/dev/null || true
    done
    
    # Remove all containers with this name (running or stopped)
    for container in $(podman ps -aq --filter "name=${CONTAINER_NAME}"); do
        echo "Removing container: $container"
        podman rm -f $container 2>/dev/null || true
    done
    
    # Double-check cleanup worked
    if podman ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo "⚠️ Container still exists, force removing by ID..."
        CONTAINER_ID=$(podman ps -a --filter "name=${CONTAINER_NAME}" --format "{{.ID}}" | head -1)
        if [ ! -z "$CONTAINER_ID" ]; then
            podman rm -f $CONTAINER_ID || true
        fi
    fi
    
    echo "✅ Container cleanup complete"
}

# CRITICAL FIX: GPU device creation function
setup_gpu_devices() {
    echo "🔧 Setting up GPU devices..."
    
    # Load NVIDIA kernel modules
    sudo modprobe nvidia 2>/dev/null || true
    sudo modprobe nvidia-uvm 2>/dev/null || true
    sudo modprobe nvidia-modeset 2>/dev/null || true
    
    # Create device files using nvidia-smi
    sudo nvidia-smi -pm 1 >/dev/null 2>&1 || true
    
    # Check which devices exist
    echo "Available NVIDIA devices:"
    ls -la /dev/nvidia* 2>/dev/null || echo "No NVIDIA devices found"
    
    # Manually create missing critical devices
    if [ ! -e "/dev/nvidia-modeset" ] && grep -q nvidia-modeset /proc/devices; then
        MAJOR=$(awk '/nvidia-modeset/ {print $1}' /proc/devices)
        if [ ! -z "$MAJOR" ]; then
            sudo mknod /dev/nvidia-modeset c $MAJOR 0
            sudo chmod 666 /dev/nvidia-modeset
            echo "Created /dev/nvidia-modeset"
        fi
    fi
    
    if [ ! -e "/dev/nvidia-uvm" ] && grep -q nvidia-uvm /proc/devices; then
        MAJOR=$(awk '/nvidia-uvm/ {print $1}' /proc/devices)
        if [ ! -z "$MAJOR" ]; then
            sudo mknod /dev/nvidia-uvm c $MAJOR 0
            sudo chmod 666 /dev/nvidia-uvm
            echo "Created /dev/nvidia-uvm"
        fi
    fi
}

# Initial cleanup
cleanup_containers

# Set up persistent directories (bulletproof approach)
echo "📁 Setting up bulletproof persistent storage..."
mkdir -p "$PERSISTENT_DATA_DIR"/{vectordb,documents,models,logs,backups,static}
chmod 777 "$PERSISTENT_DATA_DIR"/{vectordb,documents,models,logs}
chmod 755 "$PERSISTENT_DATA_DIR"/{static,backups}

# Handle static files
echo "🌐 Preparing static files..."
if [ -d "static" ]; then
    echo "📋 Copying static files to persistent storage..."
    cp -r static/* "$PERSISTENT_DATA_DIR/static/" 2>/dev/null || echo "Static directory is empty"
else
    echo "ℹ️ No static directory found - web interface will use API-only mode"
    # Create a basic index.html for API access
    cat > "$PERSISTENT_DATA_DIR/static/index.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Red Hat RAG API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #cc0000; }
        .link { display: inline-block; margin: 10px 15px 10px 0; padding: 10px 20px; background: #0066cc; color: white; text-decoration: none; border-radius: 4px; }
        .link:hover { background: #004499; }
        .formats { background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 20px 0; }
        .format-list { columns: 2; column-gap: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Red Hat Documentation RAG - Multi-Format</h1>
        <p>GPU-accelerated intelligent document search with comprehensive format support.</p>

        <div>
            <a href="/docs" class="link">📚 API Documentation</a>
            <a href="/health" class="link">💓 Health Check</a>
            <a href="/api/stats" class="link">📊 Statistics</a>
            <a href="/api/formats" class="link">📄 Supported Formats</a>
            <a href="/gpu-info" class="link">🖥️ GPU Info</a>
        </div>

        <div class="formats">
            <h3>📋 Supported Document Formats</h3>
            <div class="format-list">
                <div>• PDF (.pdf)</div>
                <div>• Microsoft Word (.docx)</div>
                <div>• Microsoft Excel (.xlsx/.xls)</div>
                <div>• Plain Text (.txt)</div>
                <div>• CSV (.csv)</div>
                <div>• EPUB (.epub)</div>
                <div>• HTML (.html/.htm)</div>
                <div>• Rich Text (.rtf)</div>
                <div>• OpenDocument (.odt)</div>
                <div>• MOBI (.mobi) - experimental</div>
            </div>
        </div>

        <h3>🔧 Quick API Examples</h3>
        <pre>
# Upload a document
curl -X POST "http://localhost:8080/api/documents/upload" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@your-document.pdf"

# Search documents
curl -X POST "http://localhost:8080/api/search" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "SELinux configuration", "max_results": 10}'

# Check system stats
curl "http://localhost:8080/api/stats"
        </pre>
    </div>
</body>
</html>
EOF
fi

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
    echo "🏗️ Building enhanced multi-format container image..."
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

# CRITICAL FIX: Setup GPU devices before deployment
setup_gpu_devices

# COMPLETELY FIXED: Function to deploy with GPU method
deploy_with_method() {
    local method="$1"
    local gpu_args="$2"
    local env_args="$3"

    echo "Attempting deployment: $method"
    
    # CRITICAL: Clean up any existing container before each attempt
    cleanup_containers
    
    # Build the command
    local cmd="podman run -d --name $CONTAINER_NAME --replace"
    cmd="$cmd --publish 127.0.0.1:8080:8080"
    cmd="$cmd --volume '$PERSISTENT_DATA_DIR/documents:/app/documents:Z'"
    cmd="$cmd --volume '$PERSISTENT_DATA_DIR/vectordb:/app/vectordb:Z'"
    cmd="$cmd --volume '$PERSISTENT_DATA_DIR/models:/app/models:Z'"
    cmd="$cmd --volume '$PERSISTENT_DATA_DIR/logs:/app/logs:Z'"
    cmd="$cmd --volume '$PERSISTENT_DATA_DIR/static:/app/static:Z'"
    cmd="$cmd --security-opt=label=disable"
    cmd="$cmd --env DOCUMENTS_DIR=/app/documents"
    cmd="$cmd --env CHROMA_DB_PATH=/app/vectordb"
    cmd="$cmd --env SENTENCE_TRANSFORMERS_HOME=/app/models"
    cmd="$cmd --env MAX_FILE_SIZE=100"
    cmd="$cmd --env CHUNK_SIZE=500"
    cmd="$cmd --env CHUNK_OVERLAP=50"
    
    # Add environment args if provided
    if [ ! -z "$env_args" ]; then
        cmd="$cmd $env_args"
    fi
    
    # Add GPU args if provided
    if [ ! -z "$gpu_args" ]; then
        cmd="$cmd $gpu_args"
    fi
    
    cmd="$cmd $IMAGE_NAME"
    
    echo "Executing: $cmd"
    
    # Execute the command and check result
    if eval $cmd 2>/dev/null; then
        # Wait a moment for container to start
        sleep 3
        
        # Verify container is actually running
        if podman ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            echo "✅ $method: SUCCESS - Container is running"
            return 0
        else
            echo "❌ $method: Container created but not running"
            podman logs $CONTAINER_NAME 2>/dev/null || true
            cleanup_containers
            return 1
        fi
    else
        echo "⚠️ $method: Command failed"
        cleanup_containers
        return 1
    fi
}

# Deploy container - trying different GPU access methods with proper error handling
echo "🚀 Deploying enhanced multi-format container..."

DEPLOYMENT_SUCCESS=false

# Method 1: CDI all GPUs (most comprehensive)
if [ -e "/dev/nvidia0" ] && [ -e "/dev/nvidia-modeset" ] && [ -e "/dev/nvidia-uvm" ]; then
    if deploy_with_method "GPU Method 1 (CDI all GPUs)" "--device nvidia.com/gpu=all" ""; then
        DEPLOYMENT_SUCCESS=true
    fi
fi

# Method 2: CDI GPU 0 (single GPU)
if [ "$DEPLOYMENT_SUCCESS" = false ] && [ -e "/dev/nvidia0" ]; then
    if deploy_with_method "GPU Method 2 (CDI GPU 0)" "--device nvidia.com/gpu=0" ""; then
        DEPLOYMENT_SUCCESS=true
    fi
fi

# Method 3: Legacy device mounting (manual device files)
if [ "$DEPLOYMENT_SUCCESS" = false ] && [ -e "/dev/nvidia0" ] && [ -e "/dev/nvidiactl" ]; then
    GPU_DEVICES="--device /dev/nvidia0 --device /dev/nvidiactl"
    if [ -e "/dev/nvidia-uvm" ]; then
        GPU_DEVICES="$GPU_DEVICES --device /dev/nvidia-uvm"
    fi
    if [ -e "/dev/nvidia-modeset" ]; then
        GPU_DEVICES="$GPU_DEVICES --device /dev/nvidia-modeset"
    fi
    
    if deploy_with_method "GPU Method 3 (Legacy devices)" "$GPU_DEVICES" ""; then
        DEPLOYMENT_SUCCESS=true
    fi
fi

# Method 4: CPU Fallback (always works)
if [ "$DEPLOYMENT_SUCCESS" = false ]; then
    if deploy_with_method "CPU Fallback" "" "--env CUDA_VISIBLE_DEVICES=''"; then
        echo "⚠️ Note: Container will run in CPU mode (no GPU acceleration)"
        DEPLOYMENT_SUCCESS=true
    fi
fi

# Check final deployment status
if [ "$DEPLOYMENT_SUCCESS" = false ]; then
    echo "❌ All deployment methods failed"
    echo ""
    echo "🔍 Diagnostics:"
    echo "Available NVIDIA devices:"
    ls -la /dev/nvidia* 2>/dev/null || echo "No NVIDIA devices found"
    echo ""
    echo "Container status:"
    podman ps -a --filter name=$CONTAINER_NAME
    echo ""
    echo "Recent logs (if any):"
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

# Test multi-format support
echo "🔍 Testing multi-format support..."
sleep 3
format_response=$(curl -s "http://localhost:8080/api/formats" 2>/dev/null || echo "[]")
if echo "$format_response" | grep -q "supported_formats"; then
    echo "✅ Multi-format support active"
    available_formats=$(echo "$format_response" | grep -o '"\..*"' | wc -l)
    echo "📄 Available formats: $available_formats"
else
    echo "⚠️ Multi-format support not yet ready (container still initializing)"
fi

# Show container logs (last 20 lines)
echo ""
echo "📋 Container startup logs (last 20 lines):"
podman logs $CONTAINER_NAME | tail -n 20

# Final status
echo ""
echo "🎯 Enhanced Multi-Format RAG Deployment Status:"
echo "🌐 Web Interface: http://localhost:8080"
echo "📊 API Documentation: http://localhost:8080/docs"
echo "💓 Health Check: http://localhost:8080/health"
echo "📄 Supported Formats: http://localhost:8080/api/formats"
echo "🖥️ GPU Info: http://localhost:8080/gpu-info"
echo "📈 Statistics: http://localhost:8080/api/stats"
echo ""
echo "💾 Persistent Data Location: $PERSISTENT_DATA_DIR"
echo "📊 Vector Embeddings: $PERSISTENT_DATA_DIR/vectordb/"
echo "📄 Documents: $PERSISTENT_DATA_DIR/documents/"
echo "🌐 Static Files: $PERSISTENT_DATA_DIR/static/"
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
ls -la "$PERSISTENT_DATA_DIR/documents/" 2>/dev/null | head -5 || echo "  (empty - upload documents via web interface)"
echo ""
echo "Supported file types for upload:"
echo "  PDF, DOCX, XLSX, XLS, TXT, CSV, EPUB, HTML, RTF, ODT, MOBI"

echo ""
echo "🧪 To test multi-format processing:"
echo "  1. Upload different document types via the API or web interface"
echo "  2. Check processing with: curl http://localhost:8080/api/stats"
echo "  3. Search across formats: curl -X POST http://localhost:8080/api/search -H 'Content-Type: application/json' -d '{\"query\":\"your search\"}'"
echo ""
echo "✅ Enhanced Multi-Format RAG deployment complete!"
