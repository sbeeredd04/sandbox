#!/bin/bash
# Clean up zombie DDP processes and free ports

echo "Cleaning up zombie processes..."

# Kill any Python training processes
pkill -9 -f "python.*main.py" 2>/dev/null || true

# Kill any remaining PyTorch distributed processes
pkill -9 -f "torch.*distributed" 2>/dev/null || true

# Wait for processes to die
sleep 2

# Check if ports are free
echo "Checking DDP ports..."
if netstat -tulpn 2>/dev/null | grep -E ":(29500|29501)" > /dev/null; then
    echo "WARNING: DDP ports still in use!"
    netstat -tulpn 2>/dev/null | grep -E ":(29500|29501)"
    echo "Attempting to free ports..."
    lsof -ti:29500 2>/dev/null | xargs -r kill -9 || true
    lsof -ti:29501 2>/dev/null | xargs -r kill -9 || true
    sleep 2
fi

echo "Cleanup complete. Ports status:"
if netstat -tulpn 2>/dev/null | grep -E ":(29500|29501)" > /dev/null; then
    echo "Ports still occupied"
    exit 1
else
    echo "Ports are free"
    exit 0
fi
