#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Default limit if not provided
LIMIT=${1:-500}
WORKERS=${2:-10}

echo "Starting Local Recursive Crawler with limit=$LIMIT and workers=$WORKERS..."
python scripts/local_recursive_crawler.py --limit $LIMIT --workers $WORKERS

echo "Done!"
