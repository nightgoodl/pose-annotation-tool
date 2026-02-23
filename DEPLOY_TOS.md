# TOS 模式部署指南

本指南用于在新服务器上部署 pose-annotation-tool（TOS 数据源模式）。

## 架构概览

```
Port 7860 (对外)                Port 8084 (内部)
┌──────────────────┐           ┌──────────────────────┐
│  Node.js Server  │──Proxy──▶│  Python Backend      │
│  (serve.cjs)     │          │  (data_server_mv.py) │
│                  │          │                      │
│  静态文件 (dist/) │          │  ┌──────────────┐    │
│  /api/* 代理     │          │  │ tos_client   │    │
│  /data/* 代理    │          │  │ ↕ tosutil    │    │
└──────────────────┘          │  │ ↕ TOS bucket │    │
                              │  └──────────────┘    │
                              │  manifest.json       │
                              │  annotation.db       │
                              └──────────────────────┘
```

数据流：前端请求物体数据 → 后端检查本地缓存 → 未命中则用 tosutil 从 TOS 下载 tar 包 → 解压到 `/tmp/tos_tar_cache/` → 返回数据 → 标注完成后自动删除缓存。

---

## 1. 系统要求

| 项目 | 要求 |
|------|------|
| OS | Linux x86_64 (Ubuntu 20.04+) |
| Python | 3.8+ |
| Node.js | 18+ |
| 磁盘 | 至少 10GB 可用空间（`/tmp` 用于 tar 缓存） |
| 网络 | 可访问 `tos-cn-beijing.volces.com`（TOS 服务端点） |

---

## 2. 获取代码

```bash
git clone <repo_url> pose-annotation-tool
cd pose-annotation-tool
```

确保以下关键文件存在：
- `server/manifest.json` — 物体列表（6MB，已包含在仓库中）
- `server/tos_client.py` — TOS 下载工具
- `server/data_server_mv.py` — 后端服务
- `server/db.py` — 数据库模块
- `dist/` — 前端构建产物

> 如果 `dist/` 不在仓库中（被 .gitignore），需要先构建前端（见第 4 步）。

---

## 3. 安装 tosutil

tosutil 是火山引擎 TOS 的命令行工具，用于从 TOS 下载 tar 数据包。

```bash
# 下载 tosutil（Linux x86_64）
wget https://tos-tools.tos-cn-beijing.volces.com/linux/amd64/tosutil
chmod +x tosutil
mv tosutil /root/tosutil  # 或其他路径，需与 tos_client.py 中的 TOSUTIL_PATH 一致
```

### 配置 tosutil

```bash
/root/tosutil config
```

按提示输入以下信息：

| 配置项 | 值 |
|--------|-----|
| endpoint | `tos-cn-beijing.volces.com` |
| region | `cn-beijing` |
| ak | （向管理员获取 Access Key） |
| sk | （向管理员获取 Secret Key） |

其余选项回车使用默认值。

### 验证连通性

```bash
/root/tosutil ls tos://ycj-data-backup/LASA1M_WDS_TAR_MESH/ -s -limit 3
```

应能看到类似输出：
```
tos://ycj-data-backup/LASA1M_WDS_TAR_MESH/LASA1M_WDS_TAR_MESH/42444908_xxx.tar
```

---

## 4. 安装依赖

### Python 依赖

```bash
pip install numpy trimesh pillow opencv-python
```

### Node.js 依赖

```bash
cd pose-annotation-tool
npm install
```

### 构建前端（如果 dist/ 不存在）

```bash
npm run build
```

构建产物会输出到 `dist/` 目录。

---

## 5. 配置

### 5.1 tosutil 路径

如果 tosutil 安装在非默认路径，编辑 `server/tos_client.py` 修改：

```python
TOSUTIL_PATH = "/root/tosutil"  # 修改为实际路径
```

### 5.2 TOS 缓存目录

默认缓存在 `/tmp/tos_tar_cache/`。可通过环境变量修改：

```bash
export TOS_CACHE_DIR="/data/tos_cache"  # 自定义缓存路径
```

### 5.3 对齐结果保存目录

标注结果保存在 `MV_ALIGNED_ROOT`，默认为：
```
/root/csz/data_partcrafter/LASA1M_ALIGNED_MV
```

如需修改，编辑 `server/data_server_mv.py` 中的 `MV_ALIGNED_ROOT`。

### 5.4 前端路由前缀

当前前端 base path 为 `/cuhk-02/`，如需修改：
1. 修改 `vite.config.ts` 中的 `base`
2. 修改 `serve.cjs` 中的 `PREFIX`
3. 重新构建前端：`npm run build`

### 5.5 端口

| 服务 | 默认端口 | 修改方式 |
|------|----------|----------|
| 前端 (Node.js) | 7860 | 修改 `serve.cjs` 中的 `PORT` |
| 后端 (Python) | 8084 | 启动时传参 `python3 server/data_server_mv.py <port>` |

> 注意：如果修改后端端口，需同步修改 `serve.cjs` 中的 proxy target（`http://localhost:8084`）。

---

## 6. 启动服务

### 方式一：使用部署脚本（推荐）

```bash
# 设置 TOS 模式环境变量后启动
export USE_TOS=1
./deploy.sh
```

> **注意**：需要先修改 `deploy.sh` 第 51 行，在启动命令中添加 `USE_TOS=1`：
> ```bash
> USE_TOS=1 PYTHONUNBUFFERED=1 nohup $PYTHON_CMD server/data_server_mv.py 8084 > logs/backend.log 2>&1 &
> ```

### 方式二：手动启动

```bash
# 终端 1：启动后端
USE_TOS=1 python3 server/data_server_mv.py 8084

# 终端 2：启动前端
npm run serve
```

### 方式三：后台运行

```bash
mkdir -p logs

# 启动后端（TOS 模式）
USE_TOS=1 PYTHONUNBUFFERED=1 nohup python3 server/data_server_mv.py 8084 > logs/backend.log 2>&1 &
echo $! > logs/backend.pid

# 启动前端
nohup npm run serve > logs/frontend.log 2>&1 &
echo $! > logs/frontend.pid
```

---

## 7. 验证部署

### 后端健康检查

```bash
# 检查物体列表 API（应返回 JSON）
curl -s http://localhost:8084/api/mv_objects?page=1\&page_size=3 | python3 -m json.tool | head -20
```

### 前端访问

浏览器打开 `http://<server_ip>:7860/cuhk-02/mv/`

### TOS 下载测试

```bash
# 手动测试下载一个物体
cd pose-annotation-tool/server
USE_TOS=1 python3 -c "
import os; os.environ['USE_TOS']='1'
import tos_client
d = tos_client.ensure_object_cached('42444908', '9f81259a-d99d-406a-a081-1db79453ad8f')
print('OK:', os.listdir(d)[:5])
tos_client.delete_object_cache('42444908', '9f81259a-d99d-406a-a081-1db79453ad8f')
print('Cleaned up')
"
```

---

## 8. 停止服务

```bash
./stop.sh
```

---

## 9. 用户管理

首次部署时数据库 `server/annotation.db` 为空，需要创建管理员账号。

```bash
cd pose-annotation-tool/server
python3 -c "
import db
db.init_db()
uid = db.create_user('admin', 'your_password', role='admin', batch_size=999)
print(f'Created admin user, id={uid}')
"
```

然后访问 `http://<server_ip>:7860/cuhk-02/admin/` 管理用户和场景分配。

---

## 10. 常见问题

### Q: tosutil 下载失败 / 超时
- 检查 `~/.tosutilconfig` 中 ak/sk 是否正确
- 检查网络是否能访问 `tos-cn-beijing.volces.com`
- 尝试手动下载测试：`/root/tosutil cp tos://ycj-data-backup/LASA1M_WDS_TAR_MESH/LASA1M_WDS_TAR_MESH/42444908_9f81259a-d99d-406a-a081-1db79453ad8f.tar /tmp/`

### Q: 启动后物体列表为空
- 确认 `USE_TOS=1` 环境变量已设置
- 确认 `server/manifest.json` 文件存在且非空
- 查看后端日志：`tail -f logs/backend.log`

### Q: 磁盘空间不足
- TOS tar 缓存在 `/tmp/tos_tar_cache/`，标注完成后自动清理
- 手动清理：`rm -rf /tmp/tos_tar_cache/*`
- 可通过 `TOS_CACHE_DIR` 环境变量将缓存放到更大的磁盘

### Q: npm run build 失败
```bash
rm -rf node_modules dist
npm install
npm run build
```

### Q: 如何重新生成 manifest
当 TOS 上的数据有更新时，需要重新生成 manifest：
```bash
# 方式一：从本地 tar 目录生成
python3 server/generate_manifest.py /path/to/tar_dir server/manifest.json

# 方式二：重启服务使 manifest 生效
./stop.sh && USE_TOS=1 ./deploy.sh
```

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `server/data_server_mv.py` | 后端主服务（支持 TOS/本地双模式） |
| `server/tos_client.py` | TOS 下载/缓存管理 |
| `server/generate_manifest.py` | 物体列表 manifest 生成脚本 |
| `server/manifest.json` | 物体列表（31642 物体，292 场景） |
| `server/db.py` | SQLite 用户/场景管理 |
| `server/annotation.db` | SQLite 数据库文件 |
| `serve.cjs` | Node.js 前端服务（代理 + 静态文件） |
| `dist/` | 前端构建产物 |
| `deploy.sh` | 一键部署脚本 |
| `stop.sh` | 停止服务脚本 |

---

## 环境变量汇总

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `USE_TOS` | `0` | 设为 `1` 启用 TOS 数据源模式 |
| `TOS_CACHE_DIR` | `/tmp/tos_tar_cache` | tar 解压缓存目录 |
| `MANIFEST_PATH` | `server/manifest.json` | 物体列表文件路径 |
| `CUDA_VISIBLE_DEVICES` | - | GPU 设备号（渲染服务需要） |
