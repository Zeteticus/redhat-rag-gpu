# Red Hat Documentation RAG - Multi-Format Support

A GPU-accelerated Retrieval-Augmented Generation (RAG) system for Red Hat documentation with comprehensive multi-format document processing capabilities.

## 🚀 New Multi-Format Capabilities

### Supported File Formats

| Format | Extension | Library Used | Status | Description |
|--------|-----------|--------------|--------|-------------|
| **PDF** | `.pdf` | PyMuPDF (fitz) | ✅ Full | Enhanced extraction with metadata |
| **Microsoft Word** | `.docx` | python-docx | ✅ Full | Modern Word documents |
| **Microsoft Excel** | `.xlsx` | openpyxl | ✅ Full | Modern Excel spreadsheets |
| **Legacy Excel** | `.xls` | xlrd | ✅ Full | Legacy Excel files |
| **Plain Text** | `.txt` | Built-in | ✅ Full | Text files with encoding detection |
| **CSV** | `.csv` | Built-in | ✅ Full | Comma-separated values |
| **EPUB** | `.epub` | ebooklib | ✅ Full | eBook format |
| **HTML** | `.html`, `.htm` | BeautifulSoup | ✅ Full | Web pages and HTML documents |
| **Rich Text** | `.rtf` | striprtf | ✅ Full | Rich Text Format |
| **OpenDocument** | `.odt` | odfpy | ✅ Full | OpenOffice/LibreOffice documents |
| **MOBI** | `.mobi` | python-kindle | 🔄 Partial | Kindle format (experimental) |

### Key Enhancements

1. **Intelligent Format Detection**: Automatic file type detection using extensions and MIME types
2. **Unified Processing Pipeline**: Single interface for all document types
3. **Format-Specific Metadata**: Preserves document structure and metadata
4. **Robust Error Handling**: Graceful fallbacks for corrupted or unsupported files
5. **Batch Processing**: Efficient processing of multiple file types
6. **GPU Acceleration**: Maintained for all formats during embedding generation

## 🛠️ Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.12
- Podman or Docker
- Red Hat Enterprise Linux 9 (recommended)

### Quick Start

1. **Clone and build the enhanced system:**
   ```bash
   git clone https://github.com/Zeteticus/redhat-rag-gpu.git
   cd redhat-rag-gpu
   
   # Copy the enhanced files
   cp enhanced_main.py main.py
   cp enhanced_requirements.txt requirements.txt
   cp enhanced_Containerfile Containerfile
   ```

2. **Deploy with GPU support:**
   ```bash
   ./deploy-gpu.sh
   ```

3. **Access the system:**
   - Web Interface: http://localhost:8080
   - API Documentation: http://localhost:8080/docs
   - Health Check: http://localhost:8080/health
   - Supported Formats: http://localhost:8080/api/formats

## 📝 Usage Examples

### Upload Different File Types

```bash
# Upload a PDF manual
curl -X POST "http://localhost:8080/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@rhel9-installation-guide.pdf"

# Upload a Word document
curl -X POST "http://localhost:8080/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@security-procedures.docx"

# Upload an Excel spreadsheet
curl -X POST "http://localhost:8080/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@system-configurations.xlsx"

# Upload an EPUB technical book
curl -X POST "http://localhost:8080/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@redhat-handbook.epub"
```

### Search Across All Formats

```bash
# Search with format filtering
curl -X POST "http://localhost:8080/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SELinux configuration",
    "filters": {
      "file_type": ".pdf",
      "category": "security"
    },
    "max_results": 10
  }'

# Search all formats
curl -X POST "http://localhost:8080/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "container deployment with Podman",
    "max_results": 20
  }'
```

### Get System Statistics

```bash
# View processing statistics by file type
curl "http://localhost:8080/api/stats"

# Check supported formats
curl "http://localhost:8080/api/formats"

# List all processed documents
curl "http://localhost:8080/api/documents"
```

## 🔧 Configuration

### Environment Variables

```bash
# Document processing
DOCUMENTS_DIR=/app/documents          # Document storage directory
MAX_FILE_SIZE=100                     # Maximum file size in MB
CHUNK_SIZE=500                        # Text chunk size for processing
CHUNK_OVERLAP=50                      # Overlap between chunks

# GPU acceleration
GPU_BATCH_SIZE=32                     # Batch size for GPU processing
EMBEDDING_MODEL=all-MiniLM-L6-v2      # Embedding model name

# Vector storage
CHROMA_DB_PATH=/app/vectordb          # Vector database path
MIN_CONFIDENCE=0.3                    # Minimum search confidence
MAX_RESULTS=20                        # Maximum search results
```

### File Size Limits

- Default maximum file size: 100MB
- Configurable via `MAX_FILE_SIZE` environment variable
- Larger files are processed in chunks to maintain memory efficiency

## 📊 Performance Considerations

### GPU Acceleration

The system automatically optimizes batch sizes based on available GPU memory:

- **Large GPUs (>16GB)**: Batch size 128
- **Medium GPUs (8-16GB)**: Batch size 64  
- **Small GPUs (<8GB)**: Batch size 32

### Format-Specific Optimizations

1. **PDF Files**: PyMuPDF provides 5-10x faster extraction than PyPDF2
2. **Office Documents**: Native library support for better metadata preservation
3. **Large Spreadsheets**: Row-by-row processing to handle large datasets
4. **eBooks**: Chapter-based processing for better content organization
5. **HTML/Web**: Intelligent text extraction with tag removal

## 🔍 API Endpoints

### Enhanced Endpoints

- `GET /api/formats` - List supported file formats and availability
- `POST /api/documents/upload` - Upload documents (any supported format)
- `GET /api/documents` - List documents with file type information
- `POST /api/search` - Search with file type filtering
- `GET /api/stats` - Statistics including file type breakdown

### Search Filters

```json
{
  "query": "your search query",
  "filters": {
    "category": "security",           // Content category
    "version": "rhel9",              // RHEL version
    "file_type": ".pdf",             // Specific file type
    "source": "filename.docx"       // Specific document
  },
  "max_results": 10,
  "min_confidence": 0.3
}
```

## 🧪 Testing Multi-Format Support

### Test Document Processing

```bash
# Create test documents directory
mkdir -p test-docs

# Add sample files of different types
cp sample.pdf test-docs/
cp manual.docx test-docs/
cp data.xlsx test-docs/
cp guide.epub test-docs/
cp config.txt test-docs/

# Process all formats
curl -X POST "http://localhost:8080/api/documents/reprocess"

# Verify processing
curl "http://localhost:8080/api/stats" | jq '.file_types'
```

### Verify Format Support

```bash
# Check which formats are available
curl "http://localhost:8080/api/formats" | jq '.supported_formats'

# Health check including format availability
curl "http://localhost:8080/health" | jq '.supported_formats'
```

## 🚨 Troubleshooting

### Common Issues

1. **"Unsupported file type" error**
   ```bash
   # Check available formats
   curl "http://localhost:8080/api/formats"
   
   # Install missing libraries if needed
   pip install openpyxl ebooklib beautifulsoup4
   ```

2. **Memory issues with large files**
   ```bash
   # Reduce batch size
   export GPU_BATCH_SIZE=16
   
   # Increase file size limit if needed
   export MAX_FILE_SIZE=200  # 200MB
   ```

3. **Format-specific extraction failures**
   ```bash
   # Check container logs for specific errors
   podman logs redhat-rag-gpu | grep "extraction failed"
   
   # Try reprocessing with fallback
   curl -X POST "http://localhost:8080/api/documents/reprocess"
   ```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Set debug environment
export LOG_LEVEL=DEBUG

# Check extraction details
podman logs redhat-rag-gpu | grep -E "(Extracted|Processing|Error)"
```

## 🔄 Migration from PDF-Only Version

### Automatic Migration

The enhanced system automatically:
1. Detects existing PDF embeddings
2. Preserves vector database
3. Processes new file formats alongside existing PDFs
4. Maintains backward compatibility

### Manual Verification

```bash
# Check existing data
curl "http://localhost:8080/api/stats" | jq '{total_chunks, file_types}'

# Verify PDF files still accessible
curl "http://localhost:8080/api/documents" | jq '.documents[] | select(.file_type == ".pdf")'
```

## 📈 Monitoring and Metrics

### Performance Metrics

- **Processing Speed**: Format-specific extraction times
- **GPU Utilization**: Embedding generation efficiency  
- **Storage Usage**: Vector database growth by file type
- **Search Performance**: Response times across formats

### Dashboard Access

```bash
# View comprehensive statistics
curl "http://localhost:8080/api/stats" | jq '.'

# Monitor GPU usage
curl "http://localhost:8080/gpu-info" | jq '.memory'
```

## 🤝 Contributing

### Adding New Formats

1. Implement extractor in `MultiFormatExtractor` class
2. Register format in `_register_extractors()` method
3. Add file extension to `SUPPORTED_EXTENSIONS`
4. Update documentation and tests

### Example: Adding New Format

```python
def _extract_custom_format(self, file_path: Path) -> List[Dict[str, Any]]:
    """Extract text from custom format"""
    try:
        # Your extraction logic here
        return [{
            'page': 1,
            'text': extracted_text,
            'source': file_path.name,
            'section': 'Custom Document',
            'metadata': {'custom_field': 'value'}
        }]
    except Exception as e:
        logger.error(f"Custom format extraction failed: {e}")
        return []
```

## 📄 License

This project maintains the same license as the original Red Hat documentation RAG system.

## 🔗 Resources

- [Red Hat Enterprise Linux Documentation](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

*Enhanced multi-format support brings comprehensive document processing capabilities to your Red Hat documentation RAG system, enabling intelligent search across diverse technical documentation formats.*
