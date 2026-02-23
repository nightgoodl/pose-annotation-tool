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

// Proxy /api and /data to backend (also handles /cuhk-02/api and /cuhk-02/data)
const PREFIX = '/cuhk-02';
const apiProxy = createProxyMiddleware({
  target: 'http://localhost:8084',
  changeOrigin: true,
  logLevel: 'silent',
  pathRewrite: (path, req) => {
    // app.use strips the mount path, so path is relative; add back /api or /data
    const prefix = req.originalUrl.includes('/api') ? '/api' : '/data';
    console.log('[Proxy] Rewriting:', path, '-> ', prefix + path);
    return prefix + path;
  },
  onProxyReq: (proxyReq, req, res) => {
    console.log('[Proxy] Forwarding:', req.method, req.originalUrl, '-> backend' + proxyReq.path);
  },
  onProxyRes: (proxyRes, req, res) => {
    console.log('[Proxy] Response:', proxyRes.statusCode, 'for', req.originalUrl);
  }
});
app.use(['/api', '/data', `${PREFIX}/api`, `${PREFIX}/data`], apiProxy);

// Block root and /scene/ paths BEFORE serving static files (but allow admin)
app.get(['/', '/scene', '/scene/', `${PREFIX}/`, `${PREFIX}/scene`, `${PREFIX}/scene/`], (req, res, next) => {
  // Skip if it's admin path
  if (req.path === '/admin' || req.path === `${PREFIX}/admin` || 
      req.path.startsWith('/admin/') || req.path.startsWith(`${PREFIX}/admin/`)) {
    return next();
  }
  console.log('[Block] Blocked access to:', req.path);
  res.status(404).send('');
});

// Serve static files from dist (not root index.html)
const distPath = path.join(__dirname, 'dist');
console.log('[Static] Serving from:', distPath);
const staticOptions = {
  index: false,  // Don't serve index.html automatically
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.js')) {
      res.setHeader('Content-Type', 'application/javascript');
    } else if (filePath.endsWith('.css')) {
      res.setHeader('Content-Type', 'text/css');
    }
  }
};
app.use(express.static(distPath, staticOptions));
app.use(PREFIX, express.static(distPath, staticOptions));

// SPA fallback - serve index-mv.html for MV routes, index-admin.html for admin routes
app.use((req, res, next) => {
  console.log('[Fallback] Checking:', req.path);
  
  // Skip if it's an API or data request
  if (req.path.includes('/api/') || req.path.includes('/data/')) {
    console.log('[Fallback] Skipping API/data request');
    next();
    return;
  }
  
  const ext = path.extname(req.path);
  
  // Serve index-admin.html for admin routes (starting with /admin/ or /cuhk-02/admin/)
  if (!ext && (req.path.startsWith('/admin/') || req.path.startsWith(`${PREFIX}/admin/`) || req.path === '/admin' || req.path === `${PREFIX}/admin`)) {
    const indexPath = path.join(__dirname, 'dist', 'index-admin.html');
    console.log('[Fallback] Serving index-admin.html from:', indexPath);
    res.sendFile(indexPath);
    return;
  }
  
  // Serve index-mv.html for MV app routes (starting with /mv/ or /cuhk-02/mv/)
  if (!ext && (req.path.startsWith('/mv/') || req.path.startsWith(`${PREFIX}/mv/`))) {
    const indexPath = path.join(__dirname, 'dist', 'index-mv.html');
    console.log('[Fallback] Serving index-mv.html from:', indexPath);
    res.sendFile(indexPath);
    return;
  }
  
  console.log('[Fallback] Passing through');
  next();
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
