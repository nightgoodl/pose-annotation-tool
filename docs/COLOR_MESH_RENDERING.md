# Mesh 颜色渲染方案

> 本文档记录了如何在 2D 投影视图中渲染带颜色的 mesh，供未来需要时参考。

## 概述

原始的投影方式只渲染 mesh 的线框（边），不包含颜色信息。本方案实现了两种渲染模式：
1. **轮廓模式 (outline)**：快速渲染凸包轮廓，使用平均颜色填充
2. **颜色模式 (color)**：渲染所有三角形面片，保留顶点颜色

## 实现方案

### 1. Store 状态扩展

在 `annotationStore.ts` 中添加渲染模式状态：

```typescript
interface AnnotationStore {
  // ...
  meshRenderMode: 'color' | 'outline';
  setMeshRenderMode: (mode: 'color' | 'outline') => void;
}

export const useAnnotationStore = create<AnnotationStore>((set, get) => ({
  // ...
  meshRenderMode: 'outline',  // 默认使用快速的轮廓模式
  setMeshRenderMode: (mode) => set({ meshRenderMode: mode }),
}));
```

### 2. 预提取顶点数据

为了提高性能，在模型加载时一次性提取所有顶点数据：

```typescript
interface ExtractedMeshData {
  vertices: Float32Array;      // [x0,y0,z0, x1,y1,z1, ...] 已应用meshMatrix和MESH_SCALE
  triangleIndices: Uint32Array; // [a0,b0,c0, a1,b1,c1, ...]
  colors: Uint8Array;          // [r0,g0,b0, r1,g1,b1, ...] 0-255
  avgColor: { r: number; g: number; b: number };
}

// 在模型加载时提取
useEffect(() => {
  const loader = new GLTFLoader();
  loader.load(modelUrl, (gltf) => {
    const MESH_SCALE = 2.0;
    const allVertices: number[] = [];
    const allTriangleIndices: number[] = [];
    const allColors: number[] = [];
    let vertexOffset = 0;
    
    gltf.scene.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        const geometry = obj.geometry;
        const positions = geometry.attributes.position;
        const colors = geometry.attributes.color;
        obj.updateMatrixWorld(true);
        const meshMatrix = obj.matrixWorld;
        
        // 提取顶点（应用变换）
        for (let i = 0; i < positions.count; i++) {
          const vec = new THREE.Vector3(
            positions.getX(i),
            positions.getY(i),
            positions.getZ(i)
          );
          vec.applyMatrix4(meshMatrix);
          allVertices.push(vec.x * MESH_SCALE, vec.y * MESH_SCALE, vec.z * MESH_SCALE);
          
          // 提取颜色
          if (colors) {
            allColors.push(
              Math.round(colors.getX(i) * 255),
              Math.round(colors.getY(i) * 255),
              Math.round(colors.getZ(i) * 255)
            );
          } else {
            allColors.push(180, 180, 180); // 默认灰色
          }
        }
        
        // 提取三角形索引
        const indices = geometry.index;
        for (let ti = 0; ti < triCount; ti++) {
          const a = indices ? indices.getX(ti * 3) : ti * 3;
          const b = indices ? indices.getX(ti * 3 + 1) : ti * 3 + 1;
          const c = indices ? indices.getX(ti * 3 + 2) : ti * 3 + 2;
          allTriangleIndices.push(vertexOffset + a, vertexOffset + b, vertexOffset + c);
        }
        
        vertexOffset += positions.count;
      }
    });
    
    setExtractedMeshData({
      vertices: new Float32Array(allVertices),
      triangleIndices: new Uint32Array(allTriangleIndices),
      colors: new Uint8Array(allColors),
      avgColor: { r: avgR, g: avgG, b: avgB }
    });
  });
}, [modelUrl]);
```

### 3. 投影计算

使用预提取的数据进行投影，只投影需要的顶点：

```typescript
const projectedData = useMemo(() => {
  // 根据渲染模式决定采样数量
  const maxTris = renderMode === 'outline' ? 2000 : 15000;
  const step = Math.max(1, Math.ceil(triCount / maxTris));
  
  // 收集需要投影的顶点索引
  const neededVertices = new Set<number>();
  for (let ti = 0; ti < triCount; ti += step) {
    neededVertices.add(triangleIndices[ti * 3]);
    neededVertices.add(triangleIndices[ti * 3 + 1]);
    neededVertices.add(triangleIndices[ti * 3 + 2]);
  }
  
  // 只投影需要的顶点
  const projectedVertices = new Map<number, { u: number; v: number; depth: number }>();
  
  for (const i of neededVertices) {
    const sx = vertices[i * 3];
    const sy = vertices[i * 3 + 1];
    const sz = vertices[i * 3 + 2];
    
    // 应用pose变换 -> 世界坐标 -> 相机坐标 -> 像素坐标
    // ... 投影计算
    
    projectedVertices.set(i, { u, v, depth: camZ });
  }
  
  // 构建三角形数据
  const allTriangles: ProjectedTriangle[] = [];
  for (let ti = 0; ti < triCount; ti += step) {
    const a = triangleIndices[ti * 3];
    const b = triangleIndices[ti * 3 + 1];
    const c = triangleIndices[ti * 3 + 2];
    
    const p0 = projectedVertices.get(a);
    const p1 = projectedVertices.get(b);
    const p2 = projectedVertices.get(c);
    
    if (!p0 || !p1 || !p2) continue;
    
    if (renderMode === 'color') {
      allTriangles.push({
        p0: { u: p0.u, v: p0.v },
        p1: { u: p1.u, v: p1.v },
        p2: { u: p2.u, v: p2.v },
        c0: { r: colors[a * 3], g: colors[a * 3 + 1], b: colors[a * 3 + 2] },
        c1: { r: colors[b * 3], g: colors[b * 3 + 1], b: colors[b * 3 + 2] },
        c2: { r: colors[c * 3], g: colors[c * 3 + 1], b: colors[c * 3 + 2] },
        depth: (p0.depth + p1.depth + p2.depth) / 3
      });
    }
  }
  
  return { triangles: allTriangles, vertices: allVertices, avgColor };
}, [extractedMeshData, pose, intrinsics, extrinsics, renderMode]);
```

### 4. Canvas 渲染

使用 Canvas 替代 SVG 渲染，性能更好：

```typescript
const canvasRef = useRef<HTMLCanvasElement>(null);

useEffect(() => {
  const canvas = canvasRef.current;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, imageWidth, imageHeight);
  
  // 计算凸包
  const points = projectedData.vertices.map(v => ({ x: v.u, y: v.v }));
  const hull = computeConvexHull(points);
  
  if (renderMode === 'color' && projectedData.triangles.length > 0) {
    // 颜色模式：渲染所有三角形
    for (const tri of projectedData.triangles) {
      const avgR = Math.round((tri.c0.r + tri.c1.r + tri.c2.r) / 3);
      const avgG = Math.round((tri.c0.g + tri.c1.g + tri.c2.g) / 3);
      const avgB = Math.round((tri.c0.b + tri.c1.b + tri.c2.b) / 3);
      
      ctx.beginPath();
      ctx.moveTo(tri.p0.u, tri.p0.v);
      ctx.lineTo(tri.p1.u, tri.p1.v);
      ctx.lineTo(tri.p2.u, tri.p2.v);
      ctx.closePath();
      ctx.fillStyle = `rgb(${avgR}, ${avgG}, ${avgB})`;
      ctx.fill();
    }
    
    // 绘制凸包边框
    if (hull.length >= 3) {
      ctx.beginPath();
      ctx.moveTo(hull[0].x, hull[0].y);
      for (let i = 1; i < hull.length; i++) {
        ctx.lineTo(hull[i].x, hull[i].y);
      }
      ctx.closePath();
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  } else {
    // 轮廓模式：只绘制凸包
    if (hull.length >= 3) {
      const { avgColor } = projectedData;
      ctx.beginPath();
      ctx.moveTo(hull[0].x, hull[0].y);
      for (let i = 1; i < hull.length; i++) {
        ctx.lineTo(hull[i].x, hull[i].y);
      }
      ctx.closePath();
      ctx.fillStyle = `rgba(${avgColor.r}, ${avgColor.g}, ${avgColor.b}, 0.5)`;
      ctx.fill();
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 3;
      ctx.stroke();
    }
  }
}, [projectedData, imageWidth, imageHeight, renderMode]);

return (
  <canvas
    ref={canvasRef}
    width={imageWidth}
    height={imageHeight}
    className="absolute inset-0 w-full h-full pointer-events-none"
    style={{ zIndex: 10, objectFit: 'contain', objectPosition: 'center' }}
  />
);
```

### 5. UI 切换按钮

在标题栏添加渲染模式切换按钮：

```tsx
{showGhostWireframe && (
  <div className="flex items-center gap-1 bg-gray-700 rounded px-1">
    <button
      onClick={() => setMeshRenderMode('outline')}
      className={`px-2 py-1 rounded text-xs ${
        meshRenderMode === 'outline' ? 'bg-green-600 text-white' : 'text-gray-300 hover:bg-gray-600'
      }`}
      title="轮廓模式（快速）"
    >
      轮廓
    </button>
    <button
      onClick={() => setMeshRenderMode('color')}
      className={`px-2 py-1 rounded text-xs ${
        meshRenderMode === 'color' ? 'bg-green-600 text-white' : 'text-gray-300 hover:bg-gray-600'
      }`}
      title="颜色模式（详细）"
    >
      颜色
    </button>
  </div>
)}
```

## 性能优化要点

1. **预提取顶点数据**：模型加载时一次性提取，避免每次 pose 变化时重新遍历 mesh
2. **使用 TypedArray**：`Float32Array`, `Uint32Array`, `Uint8Array` 比普通数组更快
3. **限制采样数量**：outline 模式 2000 个三角形，color 模式 15000 个三角形
4. **只投影需要的顶点**：使用 `Set` 收集需要的顶点索引，避免投影所有顶点
5. **移除深度排序**：Canvas 渲染顺序对视觉效果影响不大，移除 O(n log n) 排序
6. **Canvas 替代 SVG**：Canvas 批量渲染比 SVG DOM 元素更快

## 性能对比

| 模式 | 原始线框 | 颜色渲染（优化后） |
|------|----------|-------------------|
| 采样数量 | 10000 | 15000 |
| 投影顶点数 | ~30000 | ~45000 |
| 排序 | 无 | 无 |
| 渲染方式 | SVG line | Canvas fill |

## 注意事项

1. 颜色渲染需要 mesh 有 `geometry.attributes.color` 属性
2. 如果没有顶点颜色，会使用默认灰色 `rgb(180, 180, 180)`
3. 颜色模式比轮廓模式慢，但仍然可以实时交互
4. 凸包计算使用 Graham Scan 算法，复杂度 O(n log n)
