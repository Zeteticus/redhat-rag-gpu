# Setting Up GPU-Accelerated RAG for Red Hat Documentation

This guide will help you set up a GPU-accelerated Retrieval-Augmented Generation (RAG) system for Red Hat documentation, which will significantly speed up embedding generation and search operations.

## Prerequisites

1. **NVIDIA GPU**: A compatible NVIDIA GPU
2. **NVIDIA Drivers**: Properly installed NVIDIA drivers
3. **CUDA Toolkit**: Compatible CUDA version (11.8+ recommended)
4. **Podman**: With GPU support enabled

## Step 1: Verify GPU Setup

Before proceeding, verify that your GPU is properly detected and working:

```bash
# Check if GPU is detected
nvidia-smi

# Check CUDA version
nvcc --version
```

If `nvidia-smi` works correctly, you should see information about your GPU and CUDA version.

## Step 2: Set Up Directory Structure

Create a new directory for your GPU-accelerated RAG system:

```bash
# Create a new directory
mkdir -p redhat-rag-gpu
cd redhat-rag-gpu

# Create required subdirectories
mkdir -p documents data/{vectordb,logs,backups} static
chmod -R 777 documents data
```

## Step 3: Create Configuration Files

Create the following files in your `redhat-rag-gpu` directory:

### 1. main.py

Save the complete `main.py` file with GPU support (from the "Complete main.py with GPU Support" artifact).

### 2. Containerfile

Save the Containerfile for GPU support (from the "Containerfile for GPU-Accelerated RAG" artifact).

### 3. .env

Save the environment configuration (from the "Environment Configuration (.env)" artifact).

### 4. deploy-gpu.sh

Save the deployment script (from the "deploy-gpu.sh Script" artifact).

### 5. backup-gpu-vectordb.sh

Save the backup script (from the "backup-gpu-vectordb.sh Script" artifact).

## Step 4: Make Scripts Executable

```bash
chmod +x deploy-gpu.sh
chmod +x backup-gpu-vectordb.sh
```

## Step 5: Copy Red Hat Documentation

If you want to use your existing Red Hat documentation PDFs, copy them to the new `documents` directory:

```bash
# Assuming your original RAG system is in ../redhat-rag
cp ../redhat-rag/documents/*.pdf documents/
```

## Step 6: Deploy the GPU-Accelerated System

Run the deployment script:

```bash
./deploy-gpu.sh
```

The script will:
1. Check for GPU availability
2. Build the container with GPU support
3. Deploy the container with proper GPU access
4. Verify that GPU acceleration is working

## Step 7: Verify GPU Acceleration

Once the system is deployed, you can verify that GPU acceleration is working by:

1. Checking the logs:
   ```bash
   podman logs redhat-rag-gpu
   ```
   Look for messages like "GPU detected" and "Using device: cuda:0"

2. Visiting the GPU info endpoint:
   ```bash
   curl http://localhost:8080/gpu-info
   ```
   This should show details about your GPU, including availability and memory usage

## Step 8: Set Up Regular Backups

Run the backup script to create your first backup and optionally set up automatic daily backups:

```bash
./backup-gpu-vectordb.sh
```

## Usage

The GPU-accelerated RAG system is now ready to use! Here's how to interact with it:

- **Web Interface**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health
- **GPU Information**: http://localhost:8080/gpu-info

## Key Management Commands

```bash
# View container logs
podman logs redhat-rag-gpu

# Stop the container
podman stop redhat-rag-gpu

# Start the container
podman start redhat-rag-gpu

# Restart the container
podman restart redhat-rag-gpu

# Create a manual backup
./backup-gpu-vectordb.sh
```

## Troubleshooting

### GPU Not Detected

If the system reports that the GPU is not available:

1. Verify that your GPU is working with `nvidia-smi`
2. Check that Podman is configured to access the GPU
3. Make sure the container has GPU access with `--device nvidia.com/gpu=all`

### Memory Issues

If you encounter GPU memory issues:

1. Reduce the batch size in the `.env` file:
   ```
   GPU_BATCH_SIZE=16  # Try a smaller value
   ```

2. Monitor GPU memory usage while processing documents:
   ```bash
   watch -n 1 nvidia-smi
   ```

## Performance Comparison

With GPU acceleration, you should see significant performance improvements:

- **Embedding Generation**: 5-20x faster depending on your GPU
- **Search Operations**: 2-5x faster for large document collections
- **Overall Processing**: Much more efficient handling of large document sets

Enjoy your much faster RAG system with GPU acceleration!
