#!/usr/bin/env python3
"""
Red Hat Documentation RAG Backend
FastAPI-based backend for intelligent document search and retrieval
Optimized for RHEL with GPU Acceleration
"""

import os
import re
import json
import time
import logging
import hashlib
import shutil
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import uvicorn
import PyPDF2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    DOCUMENTS_DIR = os.getenv('DOCUMENTS_DIR', './documents')
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './vectordb')
    MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
    MAX_RESULTS = int(os.getenv('MAX_RESULTS', '20'))
    MIN_CONFIDENCE = float(os.getenv('MIN_CONFIDENCE', '0.3'))
    GPU_BATCH_SIZE = int(os.getenv('GPU_BATCH_SIZE', '32'))

    # Red Hat specific patterns
    RHEL_VERSION_PATTERN = r'(?:RHEL|Red Hat Enterprise Linux)\s*(\d+(?:\.\d+)?)'
    CATEGORY_KEYWORDS = {
        'installation': ['install', 'setup', 'deployment', 'bootstrap'],
        'networking': ['network', 'ip', 'dns', 'dhcp', 'firewall', 'iptables'],
        'security': ['security', 'selinux', 'authentication', 'ssl', 'tls', 'encryption'],
        'storage': ['storage', 'filesystem', 'disk', 'lvm', 'raid', 'mount'],
        'virtualization': ['kvm', 'qemu', 'libvirt', 'virtual', 'hypervisor'],
        'containers': ['container', 'podman', 'docker', 'kubernetes', 'openshift'],
        'troubleshooting': ['troubleshoot', 'debug', 'error', 'problem', 'issue']
    }

# Pydantic Models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    max_results: Optional[int] = Field(default=10, ge=1, le=50)
    min_confidence: Optional[float] = Field(default=0.3, ge=0.0, le=1.0)

class SearchResult(BaseModel):
    id: str
    title: str
    content: str
    source: str
    confidence: float
    category: str
    version: str
    tags: List[str]
    page: int
    section: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_count: int
    response_time_ms: int
    query: str
    filters_applied: Dict[str, Any]

class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_queries: int
    avg_response_time_ms: float
    system_status: str
    last_updated: str
    categories: Dict[str, int]
    versions: Dict[str, int]
    gpu_enabled: bool
    gpu_info: Optional[str] = None

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    id: str
    content: str
    source: str
    page: int
    section: str
    title: str
    category: str
    version: str
    tags: List[str]
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class GPUVectorStore:
    """
    A vector storage implementation with GPU acceleration.
    Uses basic file storage and numpy for vector operations.
    """

    def __init__(self, db_path: str = './vectordb'):
        self.db_path = Path(db_path)
        self.vectors_file = self.db_path / 'vectors.npy'
        self.metadata_file = self.db_path / 'metadata.json'
        self.documents_file = self.db_path / 'documents.json'
        self.ids_file = self.db_path / 'ids.json'

        self.model = None
        self.device = "cpu"
        self.gpu_info = None
        self._stats = {
            'total_queries': 0,
            'response_times': [],
            'last_updated': datetime.now()
        }

        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)

    def ensure_storage_directory(self):
        """Ensure the storage directory exists and is writable"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)

            # Check if directory is writable by attempting to create a test file
            test_file = self.db_path / "write_test.txt"
            with open(test_file, 'w') as f:
                f.write("Test write access")
            os.remove(test_file)

            # Create subdirectories for better organization
            os.makedirs(self.db_path / "data", exist_ok=True)
            os.makedirs(self.db_path / "metadata", exist_ok=True)

            return True
        except Exception as e:
            logger.error(f"Storage directory error: {str(e)}")
            return False

    async def initialize(self):
        """Initialize the vector store with GPU support if available"""
        try:
            # Check storage directory
            if not self.ensure_storage_directory():
                logger.error("Failed to ensure storage directory is accessible")
                raise Exception("Storage directory is not accessible for read/write")

            # Check for GPU availability
            if torch.cuda.is_available():
                self.device = f"cuda:{torch.cuda.current_device()}"
                self.gpu_info = f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
                logger.info(f"GPU detected: {self.gpu_info}")
                logger.info(f"Using device: {self.device}")
            else:
                self.device = "cpu"
                logger.warning("No GPU detected, falling back to CPU")

            # Initialize the model with the appropriate device
            logger.info(f"Loading embedding model: {Config.MODEL_NAME} on {self.device}")
            self.model = SentenceTransformer(Config.MODEL_NAME, device=self.device)

            # Initialize empty data structures if files don't exist
            if not self.vectors_file.exists():
                self._save_vectors(np.array([]).reshape(0, 384))  # Default embedding size for all-MiniLM-L6-v2

            if not self.metadata_file.exists():
                self._save_metadata([])

            if not self.documents_file.exists():
                self._save_documents([])

            if not self.ids_file.exists():
                self._save_ids([])

            # Load existing data
            if self.vectors_file.exists() and self.metadata_file.exists() and self.documents_file.exists() and self.ids_file.exists():
                logger.info("Loading existing vector store data...")

                # Verify data integrity
                try:
                    vectors = self._load_vectors()
                    metadata = self._load_metadata()
                    documents = self._load_documents()
                    ids = self._load_ids()

                    if len(metadata) != len(documents) or len(metadata) != len(ids):
                        logger.warning("Data integrity issue detected - lengths don't match")
                        if vectors.size > 0 and vectors.shape[0] != len(metadata):
                            logger.warning("Vector count doesn't match metadata count")

                    logger.info(f"Loaded {len(metadata)} items from persistent storage")
                except Exception as e:
                    logger.error(f"Error loading data: {str(e)}")
                    logger.warning("Creating new empty data store")

                    # Initialize empty data structures
                    self._save_vectors(np.array([]).reshape(0, 384))
                    self._save_metadata([])
                    self._save_documents([])
                    self._save_ids([])

            logger.info(f"Vector store initialized successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def _load_vectors(self) -> np.ndarray:
        """Load vectors from file"""
        if self.vectors_file.exists():
            return np.load(str(self.vectors_file), allow_pickle=True)
        return np.array([]).reshape(0, 384)

    def _save_vectors(self, vectors: np.ndarray):
        """Save vectors to file with fix for numpy.save() .npy extension behavior"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.vectors_file), exist_ok=True)
            
            # Create temp filename without .npy since numpy will add it
            base_name = str(self.vectors_file).replace('.npy', '')
            temp_base = base_name + '.tmp'  # e.g., /app/vectordb/vectors.tmp
            
            # numpy.save will create vectors.tmp.npy
            np.save(temp_base, vectors)
            
            # The actual temp file created by numpy
            actual_temp_file = temp_base + '.npy'  # e.g., /app/vectordb/vectors.tmp.npy
            
            # Move the actual temp file to final location
            shutil.move(actual_temp_file, str(self.vectors_file))
            
            logger.info(f"Saved {vectors.shape[0]} vectors to {self.vectors_file}")
            
        except Exception as e:
            # Cleanup any temp files that might exist
            cleanup_files = [temp_base + '.npy', temp_base] if 'temp_base' in locals() else []
            for cleanup_file in cleanup_files:
                if os.path.exists(cleanup_file):
                    try:
                        os.remove(cleanup_file)
                    except:
                        pass
            
            # Fallback: Try direct save without temp file
            try:
                logger.warning(f"Temp file method failed ({e}), trying direct save...")
                # Remove .npy from target since numpy will add it back
                direct_target = str(self.vectors_file).replace('.npy', '')
                np.save(direct_target, vectors)
                logger.info("Direct save successful")
            except Exception as e2:
                logger.error(f"Direct save also failed: {e2}")
                raise e2

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return []

    def _save_metadata(self, metadata: List[Dict[str, Any]]):
        """Save metadata to file with error handling and atomic writes"""
        try:
            # Save to temporary file first
            temp_file = str(self.metadata_file) + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(metadata, f)

            # Then rename to actual file
            shutil.move(temp_file, str(self.metadata_file))

            logger.info(f"Saved metadata for {len(metadata)} items")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise

    def _load_documents(self) -> List[str]:
        """Load documents from file"""
        if self.documents_file.exists():
            with open(self.documents_file, 'r') as f:
                return json.load(f)
        return []

    def _save_documents(self, documents: List[str]):
        """Save documents to file with error handling and atomic writes"""
        try:
            # Save to temporary file first
            temp_file = str(self.documents_file) + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(documents, f)

            # Then rename to actual file
            shutil.move(temp_file, str(self.documents_file))

            logger.info(f"Saved {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error saving documents: {str(e)}")
            raise

    def _load_ids(self) -> List[str]:
        """Load IDs from file"""
        if self.ids_file.exists():
            with open(self.ids_file, 'r') as f:
                return json.load(f)
        return []

    def _save_ids(self, ids: List[str]):
        """Save IDs to file with error handling and atomic writes"""
        try:
            # Save to temporary file first
            temp_file = str(self.ids_file) + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(ids, f)

            # Then rename to actual file
            shutil.move(temp_file, str(self.ids_file))

            logger.info(f"Saved {len(ids)} IDs")
        except Exception as e:
            logger.error(f"Error saving IDs: {str(e)}")
            raise

    def add_chunks(self, chunks):
        """Add document chunks to the vector store with GPU acceleration"""
        if not chunks:
            return

        try:
            # Load existing data
            vectors = self._load_vectors()
            metadata = self._load_metadata()
            documents = self._load_documents()
            ids = self._load_ids()

            # Generate embeddings with GPU acceleration if available
            texts = [chunk.content for chunk in chunks]

            # Get current batch size based on available memory
            batch_size = Config.GPU_BATCH_SIZE  # Default batch size
            if torch.cuda.is_available():
                # For large GPUs, increase batch size
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                if gpu_mem > 16:
                    batch_size = 128
                elif gpu_mem > 8:
                    batch_size = 64
                logger.info(f"Using GPU with batch size {batch_size}")

            # Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=True,
                    batch_size=batch_size
                )
                all_embeddings.append(batch_embeddings)
                # Free up CUDA memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Combine batches
            if len(all_embeddings) > 1:
                new_embeddings = np.vstack(all_embeddings)
            else:
                new_embeddings = all_embeddings[0]

            # Prepare data
            new_ids = [chunk.id for chunk in chunks]
            new_metadata = []
            new_documents = []

            for chunk in chunks:
                new_metadata.append({
                    'source': chunk.source,
                    'page': chunk.page,
                    'section': chunk.section,
                    'title': chunk.title,
                    'category': chunk.category,
                    'version': chunk.version,
                    'tags': json.dumps(chunk.tags),
                    **chunk.metadata
                })
                new_documents.append(chunk.content)

            # Append to existing data
            if vectors.size > 0:
                vectors = np.vstack([vectors, new_embeddings])
            else:
                vectors = new_embeddings

            metadata.extend(new_metadata)
            documents.extend(new_documents)
            ids.extend(new_ids)

            # Save updated data
            self._save_vectors(vectors)
            self._save_metadata(metadata)
            self._save_documents(documents)
            self._save_ids(ids)

            # Update stats
            self._stats['last_updated'] = datetime.now()

            logger.info(f"Added {len(chunks)} chunks to vector store using {self.device}")

        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {str(e)}")
            raise

    def search(self, query: str, filters: Dict[str, Any] = None,
               max_results: int = 10, min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """Search for relevant documents using GPU acceleration for query embedding"""
        start_time = time.time()

        try:
            # Load data
            vectors = self._load_vectors()
            metadata = self._load_metadata()
            documents = self._load_documents()
            ids = self._load_ids()

            if vectors.size == 0:
                logger.warning("Vector store is empty")
                return []

            # Generate query embedding on GPU if available
            query_embedding = self.model.encode([query])[0]

            # Calculate cosine similarity (optimized for larger datasets)
            # Move operation to GPU if possible, otherwise use CPU
            if self.device != "cpu" and vectors.shape[0] > 10000:
                # For very large datasets, use GPU for similarity calculation
                logger.info(f"Using GPU for similarity calculation with {vectors.shape[0]} vectors")

                # Convert to torch tensors on GPU
                vectors_tensor = torch.tensor(vectors, device=self.device)
                query_tensor = torch.tensor(query_embedding, device=self.device)

                # Normalize tensors
                vectors_tensor = vectors_tensor / torch.norm(vectors_tensor, dim=1, keepdim=True)
                query_tensor = query_tensor / torch.norm(query_tensor)

                # Calculate similarities
                similarities = torch.matmul(vectors_tensor, query_tensor).cpu().numpy()
            else:
                # Use numpy for smaller datasets
                dot_products = np.dot(vectors, query_embedding)
                magnitudes = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_embedding)
                similarities = dot_products / magnitudes

            # Apply filters
            filtered_indices = []
            for i, meta in enumerate(metadata):
                if filters:
                    if filters.get('category') and meta.get('category') != filters.get('category'):
                        continue
                    if filters.get('version') and meta.get('version') != filters.get('version'):
                        continue

                confidence = similarities[i]
                if confidence >= min_confidence:
                    filtered_indices.append(i)

            # Sort by similarity and take top results
            sorted_indices = [idx for idx in sorted(filtered_indices, key=lambda i: similarities[i], reverse=True)]

            results = []
            for i in sorted_indices[:max_results]:
                meta = metadata[i]
                results.append({
                    'id': ids[i],
                    'title': meta['title'],
                    'content': documents[i],
                    'source': meta['source'],
                    'confidence': round(float(similarities[i]), 3),
                    'category': meta['category'],
                    'version': meta['version'],
                    'tags': json.loads(meta.get('tags', '[]')),
                    'page': meta['page'],
                    'section': meta['section'],
                    'metadata': {
                        k: v for k, v in meta.items()
                        if k not in ['title', 'source', 'category', 'version', 'tags', 'page', 'section']
                    }
                })

            # Update stats
            response_time = int((time.time() - start_time) * 1000)
            self._stats['total_queries'] += 1
            self._stats['response_times'].append(response_time)
            if len(self._stats['response_times']) > 100:
                self._stats['response_times'] = self._stats['response_times'][-100:]

            logger.info(f"Search completed: {len(results)} results in {response_time}ms using {self.device}")
            return results

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            # Load metadata
            metadata = self._load_metadata()

            # Calculate stats
            categories = {}
            versions = {}

            for meta in metadata:
                cat = meta.get('category', 'unknown')
                ver = meta.get('version', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
                versions[ver] = versions.get(ver, 0) + 1

            avg_response_time = (
                sum(self._stats['response_times']) / len(self._stats['response_times'])
                if self._stats['response_times'] else 0
            )

            return {
                'total_chunks': len(metadata),
                'total_queries': self._stats['total_queries'],
                'avg_response_time_ms': round(avg_response_time, 2),
                'categories': categories,
                'versions': versions,
                'last_updated': self._stats['last_updated'].isoformat(),
                'gpu_enabled': self.device != "cpu",
                'gpu_info': self.gpu_info
            }

        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}

class DocumentProcessor:
    """Handles PDF processing and text extraction"""

    def __init__(self):
        self.config = Config()

    def detect_rhel_version(self, text: str) -> str:
        """Detect RHEL version from document text"""
        match = re.search(self.config.RHEL_VERSION_PATTERN, text, re.IGNORECASE)
        if match:
            version = match.group(1)
            major_version = version.split('.')[0]
            return f"rhel{major_version}"

        # Additional detection for RHEL 10 with specific keywords
        if re.search(r'RHEL\s*10|Red Hat Enterprise Linux\s*10', text, re.IGNORECASE) or \
           (re.search(r'RHEL|Red Hat Enterprise Linux', text, re.IGNORECASE) and
            re.search(r'(2025|latest release|newest version)', text, re.IGNORECASE)):
            return "rhel10"

        return "unknown"

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF with page and section information"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pages = []

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        pages.append({
                            'page': page_num,
                            'text': text,
                            'source': pdf_path.name
                        })

                logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
                return pages

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return []

    def categorize_content(self, text: str) -> str:
        """Categorize content based on keywords"""
        text_lower = text.lower()
        scores = {}

        for category, keywords in self.config.CATEGORY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[category] = score

        return max(scores, key=scores.get) if scores else "general"

    def extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from text"""
        tags = set()
        text_lower = text.lower()

        # Technical terms
        tech_patterns = [
            r'\b(systemctl|systemd|firewalld|selinux|podman|docker)\b',
            r'\b(yum|dnf|rpm|subscription-manager)\b',
            r'\b(ssh|http|https|ftp|nfs|samba)\b',
            r'\b(tcp|udp|ip|dns|dhcp)\b'
        ]

        for pattern in tech_patterns:
            matches = re.findall(pattern, text_lower)
            tags.update(matches)

        return list(tags)[:10]  # Limit to 10 tags

    def extract_section_title(self, text: str) -> str:
        """Extract section title from text"""
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and (line.isupper() or line.startswith('Chapter') or
                        line.startswith('Section') or len(line.split()) <= 8):
                return line[:100]  # Limit length
        return "Content"

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks"""
        chunk_size = chunk_size or self.config.CHUNK_SIZE
        overlap = overlap or self.config.CHUNK_OVERLAP

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def process_document(self, pdf_path: Path) -> List[DocumentChunk]:
        """Process a PDF document into chunks"""
        pages = self.extract_text_from_pdf(pdf_path)
        if not pages:
            return []

        chunks = []
        doc_title = pdf_path.stem.replace('-', ' ').title()

        # Detect document-level metadata
        full_text = ' '.join([page['text'] for page in pages])
        doc_version = self.detect_rhel_version(full_text)
        doc_category = self.categorize_content(full_text)

        for page_data in pages:
            page_text = page_data['text']
            page_chunks = self.chunk_text(page_text)

            for i, chunk_text in enumerate(page_chunks):
                chunk_id = hashlib.md5(
                    f"{pdf_path.name}_{page_data['page']}_{i}".encode()
                ).hexdigest()

                chunk = DocumentChunk(
                    id=chunk_id,
                    content=chunk_text,
                    source=pdf_path.name,
                    page=page_data['page'],
                    section=self.extract_section_title(chunk_text),
                    title=doc_title,
                    category=self.categorize_content(chunk_text) or doc_category,
                    version=self.detect_rhel_version(chunk_text) or doc_version,
                    tags=self.extract_tags(chunk_text),
                    metadata={
                        'file_size': pdf_path.stat().st_size,
                        'processed_at': datetime.now().isoformat(),
                        'chunk_index': i,
                        'total_chunks': len(page_chunks)
                    }
                )
                chunks.append(chunk)

        logger.info(f"Processed {pdf_path.name}: {len(chunks)} chunks created")
        return chunks

class DocumentManager:
    """Manages document processing and indexing"""

    def __init__(self, vector_store: GPUVectorStore):
        self.config = Config()
        self.processor = DocumentProcessor()
        self.vector_store = vector_store
        self.processed_files = set()

        # Ensure documents directory exists
        os.makedirs(self.config.DOCUMENTS_DIR, exist_ok=True)

    async def scan_and_process_documents(self):
        """Scan documents directory and process new PDFs"""
        docs_path = Path(self.config.DOCUMENTS_DIR)
        pdf_files = list(docs_path.glob("*.pdf"))

        logger.info(f"Found {len(pdf_files)} PDF files")

        for pdf_path in pdf_files:
            if pdf_path.name not in self.processed_files:
                await self.process_document(pdf_path)
                self.processed_files.add(pdf_path.name)

    async def process_document(self, pdf_path: Path):
        """Process a single document"""
        try:
            logger.info(f"Processing document: {pdf_path.name}")
            chunks = self.processor.process_document(pdf_path)

            if chunks:
                self.vector_store.add_chunks(chunks)
                logger.info(f"Successfully processed {pdf_path.name}")
            else:
                logger.warning(f"No content extracted from {pdf_path.name}")

        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")

    async def add_document(self, file_content: bytes, filename: str):
        """Add a new document from uploaded content"""
        try:
            # First try to write to a temporary location
            temp_dir = "/tmp/uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = Path(temp_dir) / filename

            with open(temp_path, 'wb') as f:
                f.write(file_content)

            # Then try to copy to the actual documents directory
            target_path = Path(self.config.DOCUMENTS_DIR) / filename
            try:
                shutil.copy2(temp_path, target_path)
                pdf_path = target_path
                logger.info(f"Document saved to permanent location: {target_path}")
            except Exception as e:
                # If copy fails, use the temp file
                logger.warning(f"Could not copy to documents dir: {str(e)}, using temp file")
                pdf_path = temp_path

            # Process the document
            await self.process_document(pdf_path)
            self.processed_files.add(filename)

            return {"message": f"Document {filename} processed successfully"}

        except Exception as e:
            logger.error(f"Error adding document {filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

# Global instances
vector_store = None
doc_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Red Hat Documentation RAG Backend with GPU Support")

    global vector_store
    vector_store = GPUVectorStore(db_path=Config.CHROMA_DB_PATH)
    await vector_store.initialize()

    global doc_manager
    doc_manager = DocumentManager(vector_store)

    # Process existing documents
    await doc_manager.scan_and_process_documents()

    logger.info("Backend initialization complete")
    yield

    # Shutdown
    logger.info("Shutting down backend")
    # Free GPU memory if applicable
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# FastAPI application
app = FastAPI(
    title="Red Hat Documentation RAG API (GPU-Accelerated)",
    description="Intelligent search and retrieval for Red Hat system administration documentation with GPU acceleration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML"""
    static_path = Path("static/index.html")
    if static_path.exists():
        return FileResponse("static/index.html")
    else:
        return JSONResponse(
            content={"message": "Red Hat Documentation RAG API (GPU-Accelerated)", "docs": "/docs"},
            status_code=200
        )

@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search through documentation"""
    start_time = time.time()

    try:
        results = vector_store.search(
            query=request.query,
            filters=request.filters,
            max_results=request.max_results,
            min_confidence=request.min_confidence
        )

        # Convert raw results to SearchResult model
        search_results = []
        for result in results:
            search_results.append(SearchResult(**result))

        response_time = int((time.time() - start_time) * 1000)

        return SearchResponse(
            results=search_results,
            total_count=len(search_results),
            response_time_ms=response_time,
            query=request.query,
            filters_applied=request.filters
        )

    except Exception as e:
        logger.error(f"Search API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    try:
        stats = vector_store.get_stats()

        # Count total documents
        docs_path = Path(Config.DOCUMENTS_DIR)
        total_docs = len(list(docs_path.glob("*.pdf")))

        return StatsResponse(
            total_documents=total_docs,
            total_chunks=stats.get('total_chunks', 0),
            total_queries=stats.get('total_queries', 0),
            avg_response_time_ms=stats.get('avg_response_time_ms', 0),
            system_status="healthy",
            last_updated=stats.get('last_updated', datetime.now().isoformat()),
            categories=stats.get('categories', {}),
            versions=stats.get('versions', {}),
            gpu_enabled=stats.get('gpu_enabled', False),
            gpu_info=stats.get('gpu_info')
        )

    except Exception as e:
        logger.error(f"Stats API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a new PDF document"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        content = await file.read()
        result = await doc_manager.add_document(content, file.filename)

        return JSONResponse(
            content=result,
            status_code=201
        )

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{filename}")
async def get_document(filename: str):
    """Retrieve a document file"""
    # First try the main documents directory
    file_path = Path(Config.DOCUMENTS_DIR) / filename

    if not file_path.exists():
        # If not found, try the temporary uploads directory
        temp_path = Path("/tmp/uploads") / filename
        if temp_path.exists():
            file_path = temp_path
        else:
            raise HTTPException(status_code=404, detail="Document not found")

    return FileResponse(
        path=file_path,
        media_type='application/pdf',
        filename=filename
    )

@app.get("/api/documents")
async def list_documents():
    """List all available documents"""
    try:
        docs_path = Path(Config.DOCUMENTS_DIR)
        pdf_files = list(docs_path.glob("*.pdf"))

        # Also check temp directory
        temp_path = Path("/tmp/uploads")
        if temp_path.exists():
            temp_files = list(temp_path.glob("*.pdf"))
            # Only include temp files that aren't in the main directory
            main_filenames = [f.name for f in pdf_files]
            pdf_files.extend([f for f in temp_files if f.name not in main_filenames])

        documents = []
        for pdf_path in pdf_files:
            stat = pdf_path.stat()
            documents.append({
                'filename': pdf_path.name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'processed': pdf_path.name in doc_manager.processed_files,
                'location': 'temporary' if '/tmp/' in str(pdf_path) else 'permanent'
            })

        return {"documents": documents}

    except Exception as e:
        logger.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/reprocess")
async def reprocess_documents():
    """Reprocess all documents"""
    try:
        doc_manager.processed_files.clear()
        await doc_manager.scan_and_process_documents()

        return {"message": "Document reprocessing initiated"}

    except Exception as e:
        logger.error(f"Reprocess error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu-info")
async def gpu_info():
    """Get detailed GPU information"""
    gpu_data = {
        "gpu_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }

    if torch.cuda.is_available():
        gpu_data["device_name"] = torch.cuda.get_device_name(0)
        gpu_data["cuda_version"] = torch.version.cuda

        # Get memory information
        gpu_data["memory"] = {
            "total": torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
            "allocated": torch.cuda.memory_allocated(0) / (1024**3),  # GB
            "cached": torch.cuda.memory_reserved(0) / (1024**3),  # GB
        }

    return gpu_data

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if vector store directory is writable
    vector_store_ok = False
    try:
        if vector_store and vector_store.ensure_storage_directory():
            vector_store_ok = True
    except:
        pass

    # Check if documents directory is writable
    docs_ok = False
    try:
        test_file = Path(Config.DOCUMENTS_DIR) / ".health_check"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        docs_ok = True
    except:
        pass

    # Check GPU status
    gpu_ok = torch.cuda.is_available()
    gpu_info = torch.cuda.get_device_name(0) if gpu_ok else "No GPU available"

    return {
        "status": "healthy" if vector_store_ok and docs_ok else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "checks": {
            "vector_store": "ok" if vector_store_ok else "failed",
            "documents_dir": "ok" if docs_ok else "failed",
            "gpu": "ok" if gpu_ok else "not available"
        },
        "gpu_info": gpu_info if gpu_ok else None
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )
