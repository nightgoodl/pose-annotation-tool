# Quick Start Guide

## First Time Setup

```bash
# 1. Install dependencies
npm install

# 2. Build frontend
npm run build

# 3. Install Python dependencies (if needed)
pip install -r requirements.txt
```

## Deploy

```bash
./deploy.sh
```

That's it! The app is now running.

## Access URLs

**Through Tunnel (Production):**
- 🌐 Frontend: http://wstunnel-http-train.meshy.art/cuhk-02/scene/
- 🔌 API: http://wstunnel-http-train.meshy.art/cuhk-02/api/

**Local Testing:**
- 🌐 Frontend: http://localhost:7860/cuhk-02/scene/
- 🔌 API: http://localhost:7860/cuhk-02/api/

## Common Commands

```bash
# Start servers
./deploy.sh

# Stop servers
./stop.sh

# Check configuration
./test-config.sh

# View logs
tail -f logs/backend.log   # Backend log
tail -f logs/frontend.log  # Frontend log

# Rebuild frontend
npm run build

# Development mode (with hot reload)
python3 server/data_server_mv.py 8084  # Terminal 1
npm run dev                             # Terminal 2
```

## Troubleshooting

### Ports in use?
```bash
./stop.sh
```

### Build issues?
```bash
rm -rf dist node_modules
npm install
npm run build
```

### Backend issues?
```bash
pip install -r requirements.txt
```

## Architecture

```
Port 7860 (Exposed)              Port 8084 (Internal)
┌─────────────────┐             ┌──────────────────┐
│  Node.js Server │────────────▶│  Python Backend  │
│                 │   Proxy     │                  │
│ /cuhk-02/scene/ │             │ /api/...         │
│ /cuhk-02/api/   │────────────▶│ /data/...        │
│ /cuhk-02/data/  │             └──────────────────┘
└─────────────────┘
        ▲
        │
    Tunnel from
http://wstunnel-http-train.meshy.art/cuhk-02
```

## Files

- `deploy.sh` - Start both servers
- `stop.sh` - Stop both servers  
- `test-config.sh` - Test configuration
- `serve.cjs` - Production server
- `logs/` - Server logs and PIDs

## Need Help?

See detailed guides:
- 📖 [DEPLOYMENT.md](DEPLOYMENT.md) - Complete deployment guide
- 📋 [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md) - Technical details
