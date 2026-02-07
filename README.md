# 6D Pose Annotation Tool

A web-based tool for annotating 6D object poses in RGB-D scenes. Supports both single-view and multi-view annotation workflows.

## Deployment

### Prerequisites

- Node.js 18+
- Python 3.8+
- npm or yarn

### Quick Start

**1. Install frontend dependencies:**

```bash
cd pose-annotation-tool
npm install
```

**2. Start the backend data server:**

```bash
# For multi-view annotation (需要 CUDA 环境用于 nvdiffrast 渲染)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3d-objects
export CUDA_HOME=/usr/local/cuda-12.4
cd pose-annotation-tool
python server/data_server_mv.py 8084

# For single-view annotation
python server/data_server.py 8084
```

**3. Start the frontend development server:**

```bash
# Multi-view tool (recommended)
npm run dev -- --port 3003

# Access at: http://localhost:3003/index-mv.html
```

### Production Build

```bash
npm run build
# Output in dist/
```

## Project Structure

```
pose-annotation-tool/
├── src/
│   ├── components/
│   │   ├── MVApp.tsx              # Multi-view main app
│   │   ├── MVFrameView.tsx        # Multi-view frame display
│   │   ├── MVModelViewer.tsx      # 3D model viewer (multi-view)
│   │   ├── MVControlPanel.tsx     # Multi-view control panel
│   │   ├── App.tsx                # Single-view main app
│   │   ├── WorldView.tsx          # World space view (2D+3D overlay)
│   │   ├── ModelViewer.tsx        # 3D model viewer (single-view)
│   │   ├── ControlPanel.tsx       # Single-view control panel
│   │   └── ClassificationPanel.tsx
│   ├── stores/
│   │   ├── mvAnnotationStore.ts   # Multi-view state (Zustand)
│   │   └── annotationStore.ts     # Single-view state (Zustand)
│   ├── types/
│   │   ├── multiview.ts           # Multi-view type definitions
│   │   └── index.ts               # Common type definitions
│   ├── utils/
│   │   └── math.ts                # Math utilities (Umeyama, projection, etc.)
│   ├── main-mv.tsx                # Multi-view entry point
│   └── main.tsx                   # Single-view entry point
├── server/
│   ├── data_server_mv.py          # Multi-view data API server
│   ├── data_server.py             # Single-view data API server
│   ├── render_service.py          # Nvdiffrast GPU rendering service
│   └── mesh_decoder_service.py    # Mesh decoding service
├── index-mv.html                  # Multi-view HTML entry
├── index.html                     # Single-view HTML entry
├── vite.config.ts                 # Vite configuration with proxy
└── package.json
```

## Data Paths (Backend Configuration)

The backend server (`data_server_mv.py`) expects data in the following structure:

```python
# Multi-view reconstruction meshes
MV_RECON_ROOT = "/root/csz/yingbo/MV-SAM3D/reconstruction_lasa1m"
# Structure: {MV_RECON_ROOT}/{scene_id}/{object_id}/mesh.glb

# LASA1M dataset (images, depth, camera params)
LASA1M_ROOT = "/root/csz/data_partcrafter/LASA1M"
# Structure: {LASA1M_ROOT}/{scene_id}/{object_id}/
#   ├── info.json          # Camera intrinsics & extrinsics
#   ├── raw_jpg/           # RGB images
#   ├── mask/              # Point cloud projected masks
#   └── gt/{timestamp}/    # Ground truth depth maps

# Alignment results output
MV_ALIGNED_ROOT = "/root/csz/data_partcrafter/LASA1M_ALIGNED_MV"
# Structure: {MV_ALIGNED_ROOT}/{scene_id}/{object_id}/
#   ├── world_pose.npy     # 4x4 model-to-world transform
#   ├── scale.txt          # Scale factor
#   └── result.json        # Metadata (category, error, point_pairs)
```

## Data Loading Flow

### Multi-View Workflow

1. **Object List Loading** (`/api/mv_objects`)
   - Scans `MV_RECON_ROOT` for available objects with mesh.glb
   - Checks alignment status from `MV_ALIGNED_ROOT`
   - Returns paginated list with category status

2. **Object Data Loading** (`/api/mv_object_data`)
   - Loads mesh URL from `MV_RECON_ROOT`
   - Samples N frames uniformly from `LASA1M_ROOT/raw_jpg/`
   - Loads camera intrinsics/extrinsics from `info.json`
   - Provides depth map URLs and mask URLs

3. **Pose Saving** (`/api/save_mv_pose`)
   - Saves `world_pose.npy` (4x4 matrix)
   - Saves `scale.txt` (scale factor)
   - Saves `result.json` (metadata)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mv_objects` | GET | List all objects with pagination |
| `/api/mv_object_data` | GET | Get object data (mesh, frames, cameras) |
| `/api/save_mv_pose` | POST | Save annotated pose |
| `/api/render_mesh` | POST | Render mesh with nvdiffrast (GPU) |
| `/api/refresh_cache` | GET | Refresh object list cache |
| `/data/mv_mesh/{scene}/{obj}/mesh.glb` | GET | Serve mesh file |
| `/data/lasa1m/{scene}/{obj}/...` | GET | Serve LASA1M data files |

## Alignment Algorithm

### Umeyama Algorithm

Solves for similarity transform: `s * R * P_local + t ≈ P_world`

```typescript
import { solveUmeyama } from './utils/math';

const result = solveUmeyama(srcPoints, dstPoints);
// Returns: { rotation, translation, scale, transformMatrix, error }
```

### RANSAC + Umeyama

For robust alignment with outlier rejection:

```typescript
import { solveUmeyamaRANSAC } from './utils/math';

const result = solveUmeyamaRANSAC(srcPoints, dstPoints, {
  maxIterations: 100,
  inlierThreshold: 0.05  // 5cm threshold
});
// Returns: { ..., inlierIndices, outlierIndices }
```

## Coordinate System Convention

- **Camera Coordinate**: OpenCV standard (+X right, +Y down, +Z forward)
- **World Coordinate**: Z-up
- **RT Matrix**: camera-to-world (`P_world = R @ P_cam + t`)

### Unprojection Formula

```
x_cam = (u - cx) * depth / fx
y_cam = (v - cy) * depth / fy
z_cam = depth
P_world = RT @ [x_cam, y_cam, z_cam, 1]
```

## Features

### Multi-View Rendering

- **GPU 加速渲染**: 使用 nvdiffrast 进行实时 GPU 渲染，替代传统的凸包投影
- **渲染模式切换**: 控制面板中可切换 GPU 渲染和凸包渲染模式
- **实时更新**: pose 变化时自动重新渲染
- **IoU 计算**: 基于渲染的 alpha 通道计算与 GT mask 的 IoU

### Rendering API

`POST /api/render_mesh` 接受以下参数：

```json
{
  "mesh_path": "/path/to/mesh.glb",
  "pose": [[...], [...], [...], [...]],  // 4x4 object-to-world 矩阵
  "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240},
  "extrinsics": [[...], [...], [...], [...]],  // 4x4 camera-to-world 矩阵
  "image_size": [480, 640]  // [H, W]
}
```

返回 PNG 图像（RGBA，带 alpha 通道）。

## Dependencies

### Frontend
- React 18
- Three.js + @react-three/fiber + @react-three/drei
- Zustand (state management)
- TailwindCSS (styling)
- Lucide React (icons)

### Backend
- Python 3.11+
- PyTorch with CUDA
- nvdiffrast (GPU rasterization)
- trimesh (mesh loading)
- numpy

## Key Math Functions

| Function | Purpose |
|----------|---------|
| `solveUmeyama(src, dst)` | Umeyama registration |
| `solveUmeyamaRANSAC(src, dst, opts)` | RANSAC + Umeyama |
| `unprojectPoint(u, v, depth, K, RT)` | Pixel to world coordinate |
| `projectPoint(point, K, RT)` | World to pixel coordinate |
| `inverse4(matrix)` | 4x4 matrix inverse |
| `svd3x3(matrix)` | 3x3 SVD decomposition |
