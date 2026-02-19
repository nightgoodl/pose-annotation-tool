#!/bin/bash

# Stop script for pose-annotation-tool
# This script stops both the backend and frontend servers

echo "=== Stopping Pose Annotation Tool ==="
echo ""

# Stop backend
if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    if ps -p $BACKEND_PID > /dev/null; then
        echo "🛑 Stopping backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        sleep 1
        # Force kill if still running
        if ps -p $BACKEND_PID > /dev/null; then
            kill -9 $BACKEND_PID 2>/dev/null || true
        fi
    else
        echo "⚠️  Backend process not found"
    fi
    rm logs/backend.pid
else
    echo "⚠️  No backend PID file found"
fi

# Stop frontend
if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null; then
        echo "🛑 Stopping frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        sleep 1
        # Force kill if still running
        if ps -p $FRONTEND_PID > /dev/null; then
            kill -9 $FRONTEND_PID 2>/dev/null || true
        fi
    else
        echo "⚠️  Frontend process not found"
    fi
    rm logs/frontend.pid
else
    echo "⚠️  No frontend PID file found"
fi

# Fallback: kill any remaining processes on these ports
if lsof -Pi :7860 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "🔧 Cleaning up remaining processes on port 7860..."
    kill $(lsof -t -i:7860) 2>/dev/null || true
fi

if lsof -Pi :8084 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "🔧 Cleaning up remaining processes on port 8084..."
    kill $(lsof -t -i:8084) 2>/dev/null || true
fi

echo ""
echo "✅ All servers stopped"
echo ""
