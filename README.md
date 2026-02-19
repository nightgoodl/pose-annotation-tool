# Pose Annotation Tool

多视角位姿标注工具，用于标注和对齐3D物体的位姿。

## 功能特性

- 🎯 多视角位姿标注
- 🔄 实时mesh渲染预览（nvdiffrast）
- 📊 批量处理支持
- 🎨 直观的3D可视化界面
- 💾 自动保存标注结果

## 快速开始

### 1. 安装依赖

```bash
# 前端依赖
npm install

# 后端依赖（在cube环境中）
/root/miniconda3/envs/cube/bin/pip install -r requirements.txt
```

### 2. 构建前端

```bash
npm run build
```

### 3. 启动服务

```bash
bash deploy.sh
```

服务将在以下地址可用：
- 本地访问: http://localhost:7860/cuhk-02/scene/
- 隧道访问: http://wstunnel-http-train.meshy.art/cuhk-02/scene/

### 4. 停止服务

```bash
bash stop.sh
```

## nvdiffrast渲染服务

本工具使用nvdiffrast进行高性能GPU渲染。

### 测试渲染服务
```bash
bash test_render.sh
```

### 重新安装nvdiffrast（如需要）
```bash
# 安装PyTorch with CUDA
/root/miniconda3/envs/cube/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装nvdiffrast (H100 GPU)
TORCH_CUDA_ARCH_LIST="9.0" /root/miniconda3/envs/cube/bin/pip install \
  git+https://github.com/NVlabs/nvdiffrast.git \
  --no-build-isolation

# 重启服务
bash stop.sh && bash deploy.sh
```

## 系统要求

- **Python**: 3.10+ (推荐使用cube conda环境)
- **Node.js**: 18+
- **GPU**: NVIDIA GPU with CUDA support (推荐H100)
- **CUDA**: 12.1+
- **PyTorch**: 2.5.1+cu121

## 项目结构

```
pose-annotation-tool/
├── server/              # Python后端服务
│   ├── data_server_mv.py      # 多视角数据API
│   └── render_service.py      # nvdiffrast渲染服务
├── src/                 # React前端源码
│   ├── components/      # React组件
│   └── stores/          # 状态管理
├── dist/                # 构建输出（自动生成）
├── logs/                # 运行日志（自动生成）
├── deploy.sh            # 部署脚本
├── stop.sh              # 停止脚本
└── requirements.txt     # Python依赖

```

## 开发

### 开发模式

```bash
# 前端开发服务器
npm run dev

# 后端开发
/root/miniconda3/envs/cube/bin/python server/data_server_mv.py 8084
```

### 构建

```bash
npm run build
```

## 故障排除

### 渲染服务错误

如果看到 `No module named 'nvdiffrast.torch'` 或 `libc10_cuda.so` 错误：

1. 检查后端进程使用的Python环境：
   ```bash
   ps aux | grep data_server_mv.py
   ```

2. 运行渲染测试：
   ```bash
   bash test_render.sh
   ```

3. 如需重新安装nvdiffrast，参见上面的"nvdiffrast渲染服务"章节

### 其他问题

- 检查日志: `tail -f logs/backend.log` 或 `tail -f logs/frontend.log`
- 确认端口未被占用: `netstat -tuln | grep -E '7860|8084'`
- 重启服务: `bash stop.sh && bash deploy.sh`

## 许可证

[添加许可证信息]

## 贡献

[添加贡献指南]
