# 多视图标注工具公网访问配置指南

## 当前服务配置

- **前端服务**: 端口 3003 (Vite开发服务器)
- **后端服务**: 端口 8084 (Python HTTP服务器)
- **后端监听**: 已配置为 `0.0.0.0`（所有网络接口）

## 配置步骤

### 方案1: 使用 Nginx 反向代理（推荐）

#### 1. 安装 Nginx

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nginx

# CentOS/RHEL
sudo yum install nginx
```

#### 2. 配置 Nginx

创建配置文件 `/etc/nginx/sites-available/annotation-tool`:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # 替换为你的域名或公网IP
    
    # 前端
    location / {
        proxy_pass http://127.0.0.1:3003;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    # 后端API
    location /api/ {
        proxy_pass http://127.0.0.1:8084;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    # 后端数据
    location /data/ {
        proxy_pass http://127.0.0.1:8084;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        # 增加超时时间，因为数据文件可能较大
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

#### 3. 启用配置

```bash
sudo ln -s /etc/nginx/sites-available/annotation-tool /etc/nginx/sites-enabled/
sudo nginx -t  # 测试配置
sudo systemctl restart nginx
```

#### 4. 修改前端配置

编辑 `pose-annotation-tool/vite.config.ts`，移除或注释掉代理配置，因为Nginx会处理：

```typescript
export default defineConfig({
  // ... 其他配置
  server: {
    host: '0.0.0.0',  // 监听所有接口
    port: 3003,
    // 注释掉proxy配置，使用Nginx代理
    // proxy: { ... }
  }
})
```

### 方案2: 直接开放端口

#### 1. 配置防火墙

```bash
# Ubuntu (ufw)
sudo ufw allow 3003/tcp
sudo ufw allow 8084/tcp
sudo ufw reload

# CentOS (firewalld)
sudo firewall-cmd --permanent --add-port=3003/tcp
sudo firewall-cmd --permanent --add-port=8084/tcp
sudo firewall-cmd --reload
```

#### 2. 修改前端Vite配置

编辑 `pose-annotation-tool/vite.config.ts`:

```typescript
export default defineConfig({
  server: {
    host: '0.0.0.0',  // 监听所有接口
    port: 3003,
    proxy: {
      '/api': {
        target: 'http://YOUR_PUBLIC_IP:8084',  // 替换为你的公网IP
        changeOrigin: true,
      },
      '/data': {
        target: 'http://YOUR_PUBLIC_IP:8084',
        changeOrigin: true,
      }
    }
  }
})
```

#### 3. 云服务商安全组配置

如果使用阿里云/腾讯云/AWS等，需要在安全组中开放端口：

- 入站规则添加：TCP 3003
- 入站规则添加：TCP 8084

### 方案3: 使用内网穿透（无公网IP）

#### 使用 frp

1. 下载 frp: https://github.com/fatedier/frp/releases

2. 配置 `frpc.ini`:

```ini
[common]
server_addr = your-frp-server.com
server_port = 7000
token = your_token

[annotation-frontend]
type = tcp
local_ip = 127.0.0.1
local_port = 3003
remote_port = 3003

[annotation-backend]
type = tcp
local_ip = 127.0.0.1
local_port = 8084
remote_port = 8084
```

3. 启动客户端:

```bash
./frpc -c frpc.ini
```

## 启动服务

### 1. 启动后端服务器

```bash
cd /root/csz/yingbo/sam-3d-objects/pose-annotation-tool/server
python3 data_server_mv.py 8084
```

### 2. 启动前端开发服务器

```bash
cd /root/csz/yingbo/sam-3d-objects/pose-annotation-tool
npm run dev
```

## 访问地址

### 使用 Nginx 反向代理
- 访问地址: `http://your-domain.com` 或 `http://YOUR_PUBLIC_IP`

### 直接开放端口
- 前端: `http://YOUR_PUBLIC_IP:3003`
- 后端: `http://YOUR_PUBLIC_IP:8084`

### 使用 frp
- 前端: `http://frp-server.com:3003`
- 后端: `http://frp-server.com:8084`

## 安全建议

1. **配置 HTTPS**: 使用 Let's Encrypt 免费证书
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

2. **添加访问控制**: 在 Nginx 中配置 HTTP Basic Auth
   ```nginx
   location / {
       auth_basic "Annotation Tool";
       auth_basic_user_file /etc/nginx/.htpasswd;
       # ... 其他配置
   }
   ```

3. **限制访问IP**: 只允许特定IP访问
   ```nginx
   location / {
       allow 1.2.3.4;  # 允许的IP
       deny all;
       # ... 其他配置
   }
   ```

## 故障排查

### 检查服务是否运行
```bash
netstat -tlnp | grep -E ':(3003|8084)'
```

### 检查防火墙
```bash
sudo ufw status
# 或
sudo firewall-cmd --list-all
```

### 查看 Nginx 日志
```bash
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log
```

### 测试端口连通性
```bash
# 从外部机器测试
telnet YOUR_PUBLIC_IP 3003
telnet YOUR_PUBLIC_IP 8084
```

## 性能优化

对于多个标注工作者同时使用，建议：

1. **使用 PM2 管理后端进程**
   ```bash
   npm install -g pm2
   pm2 start data_server_mv.py --interpreter python3 --name annotation-backend
   ```

2. **Nginx 缓存静态资源**
   ```nginx
   location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
       expires 1y;
       add_header Cache-Control "public, immutable";
   }
   ```

3. **增加并发连接数**
   编辑 `/etc/nginx/nginx.conf`:
   ```nginx
   worker_processes auto;
   worker_connections 2048;
   ```
