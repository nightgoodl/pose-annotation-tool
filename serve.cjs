const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

const app = express();
const PORT = 7860;

// Log all requests for debugging
app.use((req, res, next) => {
  console.log(`[Request] ${req.method} ${req.url} (path: ${req.path})`);
  next();
});

// Proxy /api and /data to backend without stripping prefix
// Use context array to explicitly match paths
app.use(['/api', '/data'], createProxyMiddleware({
  target: 'http://localhost:8084',
  changeOrigin: true,
  logLevel: 'silent',  // Disable built-in logs
  pathRewrite: (path, req) => {
    // app.use strips the /api or /data prefix, so add it back
    const prefix = req.originalUrl.startsWith('/api') ? '/api' : '/data';
    console.log('[Proxy] Rewriting:', path, '-> ', prefix + path);
    return prefix + path;
  },
  onProxyReq: (proxyReq, req, res) => {
    console.log('[Proxy] Forwarding:', req.method, req.originalUrl, '-> backend' + proxyReq.path);
  },
  onProxyRes: (proxyRes, req, res) => {
    console.log('[Proxy] Response:', proxyRes.statusCode, 'for', req.originalUrl);
  }
}));

// Block root and /scene/ paths BEFORE serving static files
app.get(['/', '/scene', '/scene/'], (req, res) => {
  console.log('[Block] Blocked access to:', req.path);
  res.status(404).send('');
});

// Serve static files from dist (not root index.html)
const distPath = path.join(__dirname, 'dist');
console.log('[Static] Serving from:', distPath);
app.use(express.static(distPath, {
  index: false,  // Don't serve index.html automatically
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.js')) {
      res.setHeader('Content-Type', 'application/javascript');
    } else if (filePath.endsWith('.css')) {
      res.setHeader('Content-Type', 'text/css');
    }
  }
}));

// SPA fallback - serve index-mv.html for MV routes only
app.use((req, res, next) => {
  console.log('[Fallback] Checking:', req.path);
  
  // Skip if it's an API or data request
  if (req.path.startsWith('/api/') || req.path.startsWith('/data/')) {
    console.log('[Fallback] Skipping API/data request');
    next();
    return;
  }
  
  // Only serve index-mv.html for MV app routes (starting with /mv/)
  const ext = path.extname(req.path);
  if (!ext && req.path.startsWith('/mv/')) {
    const indexPath = path.join(__dirname, 'dist', 'index-mv.html');
    console.log('[Fallback] Serving index-mv.html from:', indexPath);
    res.sendFile(indexPath);
  } else {
    console.log('[Fallback] Passing through');
    next();
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`\n=== Pose Annotation Tool Server ===`);
  console.log(`Server running on port ${PORT}`);
  console.log(`\nLocal URLs:`);
  console.log(`  Frontend: http://localhost:${PORT}/`);
  console.log(`  API:      http://localhost:${PORT}/api/`);
  console.log(`  Data:     http://localhost:${PORT}/data/`);
  console.log(`\nTunnel URLs (via http://wstunnel-http-train.meshy.art/cuhk-02):`);
  console.log(`  Frontend: http://wstunnel-http-train.meshy.art/cuhk-02/`);
  console.log(`  API:      http://wstunnel-http-train.meshy.art/cuhk-02/api/`);
  console.log(`  Data:     http://wstunnel-http-train.meshy.art/cuhk-02/data/`);
  console.log(`\nBackend: http://localhost:8084 (internal only)`);
  console.log(`===================================\n`);
});
