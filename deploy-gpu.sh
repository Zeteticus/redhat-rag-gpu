#!/bin/bash
# Deploy Red Hat RAG with GPU support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_header() { echo -e "${CYAN}🎩 $1${NC}"; }

# Configuration
CONTAINER_NAME="redhat-rag-gpu"
IMAGE_NAME="localhost/redhat-rag-gpu:latest"
PORT="8080"

log_header "Deploying Red Hat RAG with GPU Acceleration"

# Check for NVIDIA GPU
log_info "Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. NVIDIA GPU drivers may not be installed."
    log_warning "Will attempt to continue without GPU acceleration."
    NVIDIA_AVAILABLE=false
else
    # Check NVIDIA GPU is working
    if ! nvidia-smi &> /dev/null; then
        log_error "nvidia-smi failed to run. NVIDIA GPU drivers may not be functioning correctly."
        log_warning "Will attempt to continue without GPU acceleration."
        NVIDIA_AVAILABLE=false
    else
        NVIDIA_AVAILABLE=true
        # Show GPU information
        log_info "GPU information:"
        nvidia-smi

        # Check CUDA version
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        log_info "Detected CUDA Version: $CUDA_VERSION"
    fi
fi

# Create directories with proper permissions
log_info "Setting up directories..."
mkdir -p documents data/{vectordb,models,logs,backups} static
#chmod -R 777 documents data

# Create default .env file if it doesn't exist
if [ ! -f ".env" ]; then
    log_info "Creating default .env file..."
    cat > .env << 'EOF'
DOCUMENTS_DIR=/app/documents
CHROMA_DB_PATH=/app/vectordb
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
MAX_RESULTS=20
MIN_CONFIDENCE=0.3
LOG_LEVEL=INFO
GPU_BATCH_SIZE=32
EOF
    log_success "Created default .env file"
fi

# Build the container
log_info "Building GPU-enabled container (this may take a while)..."
podman build -t $IMAGE_NAME -f Containerfile .

# If the build is successful, deploy the container
if [ $? -eq 0 ]; then
    log_success "Container build successful"
    
    # Stop existing container if it exists
    log_info "Stopping any existing containers..."
    podman stop $CONTAINER_NAME 2>/dev/null || true
    podman rm $CONTAINER_NAME 2>/dev/null || true
    
    # Deploy with GPU access if available
    log_info "Deploying container..."
    
    if [ "$NVIDIA_AVAILABLE" = true ]; then
        log_info "Deploying with GPU support..."
        PODMAN_CMD="podman run -d \
          --name $CONTAINER_NAME \
          --publish 127.0.0.1:$PORT:8080 \
          --volume $(pwd)/documents:/app/documents:rw \
          --volume $(pwd)/data/vectordb:/app/vectordb:rw \
          --volume $(pwd)/data/logs:/app/logs:rw \
          --env-file .env \
          --gpus all \
          --replace \
          $IMAGE_NAME"
    else
        log_info "Deploying without GPU support..."
        PODMAN_CMD="podman run -d \
          --name $CONTAINER_NAME \
          --publish 127.0.0.1:$PORT:8080 \
          --volume $(pwd)/documents:/app/documents:rw \
          --volume $(pwd)/data/vectordb:/app/vectordb:rw \
          --volume $(pwd)/data/logs:/app/logs:rw \
          --env-file .env \
          --replace \
          $IMAGE_NAME"
    fi
    
    echo $PODMAN_CMD
    eval $PODMAN_CMD
    
    # Check if container is running
    if podman ps | grep -q $CONTAINER_NAME; then
        log_success "Container deployed successfully"
        
        log_info "Waiting for service to start..."
        for i in {1..30}; do
            if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
                log_success "Service is accessible!"
                break
            fi
            echo -n "."
            sleep 2
            
            if [ $i -eq 30 ]; then
                log_warning "Service not responding within timeout"
            fi
        done
        
        # Check GPU status through the API
        log_info "Checking GPU status through API..."
        GPU_INFO=$(curl -s http://127.0.0.1:$PORT/gpu-info)
        echo $GPU_INFO
        
        log_header "🎉 Red Hat RAG Deployed Successfully!"
        if [[ $GPU_INFO == *"true"* ]]; then
            echo -e "${GREEN}GPU acceleration is enabled!${NC}"
        else
            echo -e "${YELLOW}GPU acceleration is not available. Using CPU only.${NC}"
        fi
        
        echo ""
        echo -e "${GREEN}🌐 Access Points:${NC}"
        echo "   Frontend:      http://localhost:$PORT"
        echo "   API Docs:      http://localhost:$PORT/docs"
        echo "   Health Check:  http://localhost:$PORT/health"
        echo "   GPU Info:      http://localhost:$PORT/gpu-info"
        
        echo ""
        echo -e "${YELLOW}📋 Management Commands:${NC}"
        echo "   View logs:     podman logs $CONTAINER_NAME"
        echo "   Stop:          podman stop $CONTAINER_NAME"
        echo "   Start:         podman start $CONTAINER_NAME"
        echo "   Restart:       podman restart $CONTAINER_NAME"
        
        # If GPU is available, check GPU usage
        if [ "$NVIDIA_AVAILABLE" = true ]; then
            log_info "Checking GPU usage by container (may take a moment)..."
            sleep 5
            nvidia-smi
        fi
    else
        log_error "Container failed to start"
        podman logs $CONTAINER_NAME
    fi
else
    log_error "Container build failed"
fi
