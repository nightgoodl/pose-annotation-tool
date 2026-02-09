#!/bin/bash

# Deployment script for pose-annotation-tool
# This script starts both the backend (Python) and frontend (Node.js) servers

echo "=== Pose Annotation Tool Deployment ==="
echo ""

# Check if dist folder exists
if [ ! -d "dist" ]; then
    echo "❌ Error: dist folder not found. Please run 'npm run build' first."
    exit 1
fi

# Kill existing processes on port 7860 and 8084
echo "🔍 Checking for existing processes..."
if lsof -Pi :7860 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 7860 is in use. Killing existing process..."
    kill $(lsof -t -i:7860) 2>/dev/null || true
    sleep 1
fi

if lsof -Pi :8084 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8084 is in use. Killing existing process..."
    kill $(lsof -t -i:8084) 2>/dev/null || true
    sleep 1
fi

# Create log directory
mkdir -p logs

# Start backend server
echo ""
echo "🚀 Starting backend server on port 8084..."
# Use cube environment if available
if [ -f "/root/miniconda3/envs/cube/bin/python" ]; then
    PYTHON_CMD="/root/miniconda3/envs/cube/bin/python"
    echo "   Using cube environment"
elif [ -f "/root/miniconda3/bin/python" ]; then
    PYTHON_CMD="/root/miniconda3/bin/python"
    echo "   Using miniconda base"
else
    PYTHON_CMD="python3"
    echo "   Using system python3"
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=7
echo "   CUDA device: 7"

nohup $PYTHON_CMD server/data_server_mv.py 8084 > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend to start
echo "   Waiting for backend to initialize..."
sleep 3

# Check if backend port is listening
BACKEND_READY=0
for i in {1..10}; do
    if netstat -tuln 2>/dev/null | grep -q ":8084" || ss -tuln 2>/dev/null | grep -q ":8084"; then
        BACKEND_READY=1
        break
    fi
    sleep 1
done

if [ $BACKEND_READY -eq 0 ]; then
    echo "❌ Backend failed to start. Check logs/backend.log"
    cat logs/backend.log
    exit 1
fi

# Start frontend server
echo ""
echo "🚀 Starting frontend server on port 7860..."
# Setup Node.js path
if [ -d "$HOME/.nvm/versions/node" ]; then
    export NVM_DIR="$HOME/.nvm"
    NODE_VERSION=$(ls -1 $NVM_DIR/versions/node | tail -1)
    export PATH="$NVM_DIR/versions/node/$NODE_VERSION/bin:$PATH"
fi
nohup npm run serve > logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

# Wait for frontend to start
echo "   Waiting for frontend to initialize..."
sleep 3

# Check if frontend port is listening
FRONTEND_READY=0
for i in {1..10}; do
    if netstat -tuln 2>/dev/null | grep -q ":7860" || ss -tuln 2>/dev/null | grep -q ":7860"; then
        FRONTEND_READY=1
        break
    fi
    sleep 1
done

if [ $FRONTEND_READY -eq 0 ]; then
    echo "❌ Frontend failed to start. Check logs/frontend.log"
    cat logs/frontend.log
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Save PIDs
echo "$BACKEND_PID" > logs/backend.pid
echo "$FRONTEND_PID" > logs/frontend.pid

echo ""
echo "✅ Deployment successful!"
echo ""
echo "📡 Access URLs:"
echo "   Local:  http://localhost:7860/cuhk-02/scene/"
echo "   Tunnel: http://wstunnel-http-train.meshy.art/cuhk-02/scene/"
echo ""
echo "🔧 Backend API:"
echo "   Local:  http://localhost:7860/cuhk-02/api/"
echo "   Tunnel: http://wstunnel-http-train.meshy.art/cuhk-02/api/"
echo ""
echo "📋 Logs:"
echo "   Backend:  tail -f logs/backend.log"
echo "   Frontend: tail -f logs/frontend.log"
echo ""
echo "🛑 To stop servers:"
echo "   ./stop.sh"
echo ""
