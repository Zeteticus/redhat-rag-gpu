#!/bin/bash
# Backup script for GPU-accelerated RAG vector database

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="./data/backups"
VECTORDB_DIR="./data/vectordb"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/vectordb_${TIMESTAMP}.tar.gz"

echo -e "${BLUE}Starting vector database backup...${NC}"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Check if vector database exists
if [ ! -d "$VECTORDB_DIR" ]; then
    echo -e "${YELLOW}Vector database directory not found at $VECTORDB_DIR${NC}"
    exit 1
fi

# Create backup
echo -e "${BLUE}Creating backup of vector database...${NC}"
tar -czf "$BACKUP_FILE" -C "$(dirname "$VECTORDB_DIR")" "$(basename "$VECTORDB_DIR")"

# Check if backup was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Backup created successfully: $BACKUP_FILE${NC}"
    
    # Clean up old backups (keep last 5)
    echo -e "${BLUE}Cleaning up old backups...${NC}"
    ls -t "$BACKUP_DIR"/vectordb_*.tar.gz | tail -n +6 | xargs rm -f 2>/dev/null
    
    # Get backup size
    BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    
    echo -e "${GREEN}Backup completed. Backup size: $BACKUP_SIZE${NC}"
    echo -e "${BLUE}Total backups kept: $(ls "$BACKUP_DIR"/vectordb_*.tar.gz | wc -l)${NC}"
else
    echo -e "${YELLOW}Backup failed!${NC}"
    exit 1
fi

# Add automatic backup to crontab if not already there
if ! crontab -l 2>/dev/null | grep -q "backup-gpu-vectordb.sh"; then
    echo -e "${BLUE}Would you like to add an automatic daily backup at midnight? (y/n)${NC}"
    read -r add_cron
    
    if [[ "$add_cron" =~ ^[Yy] ]]; then
        # Create temporary file with existing crontab
        crontab -l > /tmp/crontab.tmp 2>/dev/null
        
        # Add our backup job
        echo "0 0 * * * $(pwd)/backup-gpu-vectordb.sh >> $(pwd)/data/logs/backup.log 2>&1" >> /tmp/crontab.tmp
        
        # Install new crontab
        crontab /tmp/crontab.tmp
        rm /tmp/crontab.tmp
        
        echo -e "${GREEN}Automatic daily backup scheduled at midnight${NC}"
    fi
fi

echo -e "${GREEN}Done!${NC}"
