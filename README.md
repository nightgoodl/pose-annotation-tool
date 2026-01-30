# 6D Pose Annotation Tool

RGB-D 场景物体位姿标注工具，用于将 3D CAD 模型对齐到 2D 图像中的物体。

## 项目结构

```
pose-annotation-tool/
├── src/
│   ├── components/
│   │   ├── App.tsx              # 主应用组件
│   │   ├── WorldView.tsx        # 左侧视图：世界空间 (2D+3D叠加)
│   │   ├── ModelViewer.tsx      # 右侧视图：模型空间 (CAD查看)
│   │   ├── ClassificationPanel.tsx  # 守门人分类面板
│   │   ├── ControlPanel.tsx     # 控制面板 (点对管理、对齐操作)
│   │   └── index.ts
│   ├── stores/
│   │   └── annotationStore.ts   # Zustand 状态管理
│   ├── types/
│   │   └── index.ts             # TypeScript 类型定义
│   ├── utils/
│   │   └── math.ts              # 数学工具 (Umeyama、反投影等)
│   ├── main.tsx
│   └── index.css
├── package.json
└── README.md
```

## 核心功能

### 1. 守门人分类流程 (Gatekeeper Workflow)

在标点操作前强制分类：
- **Valid (有效)**: 解锁标注功能
- **Fixed (固定装)**: 自动保存跳过
- **Invalid (无效)**: 直接跳过

### 2. 双屏交互

| 视图 | 内容 | 坐标系 | 交互 |
|------|------|--------|------|
| **左侧 (世界空间)** | RGB图像 + 幽灵线框 | World Space | 点击获取世界坐标 |
| **右侧 (模型空间)** | CAD模型 | Model Space (原点) | 点击获取局部坐标 |

### 3. 坐标系约定

基于 `COORDINATE_SYSTEM_SUMMARY.md`:

- **相机坐标系**: OpenCV标准 (+X右, +Y下, +Z前)
- **世界坐标系**: Z-up
- **RT矩阵**: camera-to-world (`P_world = R @ P_cam + t`)

### 4. 反投影公式

```typescript
// 像素坐标 → 相机坐标
x_cam = (u - cx) * depth / fx
y_cam = (v - cy) * depth / fy
z_cam = depth

// 相机坐标 → 世界坐标 (RT是camera-to-world)
P_world = R @ P_cam + t
```

### 5. Umeyama 算法

求解带尺度的刚体变换: `s * R * P_local + t ≈ P_world`

```typescript
import { solveUmeyama } from './utils/math';

const result = solveUmeyama(srcPoints, dstPoints);
// result = { rotation, translation, scale, transformMatrix, error }
```

## 输入数据格式

```typescript
interface AnnotationInput {
  objectId: string;
  rgbImage: string;              // 图像 URL
  depthMap: Float32Array;        // 深度图 (与RGB对齐)
  depthWidth: number;
  depthHeight: number;
  maskImage: string;             // 分割掩码 URL
  cadModel: string;              // 3D模型 (.glb/.obj)
  cameraIntrinsics: CameraIntrinsics;  // 3x3 内参 K
  cameraExtrinsics: Matrix4;     // 4x4 camera-to-world RT
  initialCoarsePose: Matrix4;    // 4x4 Model-to-World 初始位姿
}
```

## 输出数据格式

```json
{
  "objectId": "...",
  "category": "valid",
  "worldPose": [/* 16位数组 Model-to-World */],
  "scale": 1.05,
  "points": [/* 点对列表 */],
  "timestamp": 1234567890
}
```

## 使用方法

### 开发模式

```bash
cd pose-annotation-tool
npm install
npm run dev
```

### 构建

```bash
npm run build
```

### 集成使用

```tsx
import { PoseAnnotationTool } from './components';

<PoseAnnotationTool
  input={annotationInput}
  onSave={(result) => console.log('保存:', result)}
  onSkip={(reason) => console.log('跳过:', reason)}
/>
```

## 依赖

- React 18
- Three.js + @react-three/fiber + @react-three/drei
- Zustand (状态管理)
- TailwindCSS (样式)
- Lucide React (图标)

## 关键数学函数

| 函数 | 用途 |
|------|------|
| `solveUmeyama(src, dst)` | Umeyama 配准算法 |
| `unprojectPoint(u, v, depth, K, RT)` | 像素反投影到世界坐标 |
| `projectPoint(point, K, RT)` | 世界坐标投影到像素 |
| `inverse4(matrix)` | 4x4 矩阵求逆 |
| `svd3x3(matrix)` | 3x3 SVD 分解 |

## 文件修改列表

| 文件 | 说明 |
|------|------|
| `src/types/index.ts` | 类型定义 |
| `src/utils/math.ts` | 数学工具函数 |
| `src/stores/annotationStore.ts` | Zustand状态管理 |
| `src/components/ModelViewer.tsx` | 模型空间视图 |
| `src/components/WorldView.tsx` | 世界空间视图 |
| `src/components/ClassificationPanel.tsx` | 分类面板 |
| `src/components/ControlPanel.tsx` | 控制面板 |
| `src/components/App.tsx` | 主应用组件 |
| `src/main.tsx` | 入口文件 (更新) |
