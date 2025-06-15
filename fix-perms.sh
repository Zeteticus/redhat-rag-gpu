#!/bin/bash
# This script fixes permissions issues with the vectordb directory
# without losing existing data

set -e  # Exit on any error

echo "🔧 Fixing vectordb directory permissions..."

# Create backup directory
mkdir -p data/backups

# Create timestamp for backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup of existing vectordb
if [ -d "data/vectordb" ]; then
    echo "📦 Creating backup of vectordb directory..."
    tar -czf "data/backups/vectordb_backup_${TIMESTAMP}.tar.gz" data/vectordb 2>/dev/null || echo "⚠️ Backup partially created (some files may have been skipped)"
    
    echo "📋 Contents of vectordb before changes:"
    ls -la data/vectordb/
    
    # Move existing directory to temporary location
    echo "🔄 Moving existing vectordb directory..."
    mv data/vectordb "data/vectordb_old_${TIMESTAMP}"
    
    # Create fresh directory with correct permissions
    echo "📁 Creating fresh vectordb directory with correct permissions..."
    mkdir -p data/vectordb
    chmod 777 data/vectordb
    
    # Selectively copy back files with corrected permissions
    echo "📄 Restoring data with correct permissions..."
    find "data/vectordb_old_${TIMESTAMP}" -type f -name "*.npy" -exec cp {} data/vectordb/ \; 2>/dev/null || true
    find "data/vectordb_old_${TIMESTAMP}" -type f -name "*.json" -exec cp {} data/vectordb/ \; 2>/dev/null || true
    
    # Set permissions on copied files
    chmod 666 data/vectordb/* 2>/dev/null || true
    
    echo "📋 Contents of vectordb after changes:"
    ls -la data/vectordb/
    
    echo "✅ Vectordb directory permissions fixed"
    echo "ℹ️ Original directory preserved at data/vectordb_old_${TIMESTAMP}"
    echo "ℹ️ Backup created at data/backups/vectordb_backup_${TIMESTAMP}.tar.gz"
else
    echo "📁 Creating new vectordb directory..."
    mkdir -p data/vectordb
    chmod 777 data/vectordb
    echo "✅ Fresh vectordb directory created with correct permissions"
fi

# Ensure other directories exist with correct permissions
echo "🔍 Checking other required directories..."
for dir in "data/logs" "documents"; do
    if [ ! -d "$dir" ]; then
        echo "📁 Creating $dir directory..."
        mkdir -p "$dir"
    fi
    echo "🔐 Setting permissions for $dir..."
    chmod 777 "$dir"
done

echo "✅ All directories setup complete!"
