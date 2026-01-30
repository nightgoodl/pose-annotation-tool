/**
 * 左侧视图 - 世界空间 2D+3D 叠加视图
 * 
 * 功能：
 * - 渲染 RGB 图像作为背景
 * - 叠加幽灵线框 (Ghost Wireframe)
 * - 用户点击图像获取像素坐标，反投影到世界坐标系
 * - Mask 区域外亮度压暗
 * 
 * 坐标系约定 (基于 COORDINATE_SYSTEM_SUMMARY.md):
 * - 相机坐标系: OpenCV标准, +X右, +Y下, +Z前
 * - RT矩阵: camera-to-world (P_world = R @ P_cam + t)
 */

import { useRef, useCallback, useMemo, useState, useEffect } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { useAnnotationStore } from '../stores/annotationStore';
import { unprojectPoint } from '../utils/math';
import type { Matrix4, CameraIntrinsics } from '../types';

// ========== 工具函数 ==========

function getDepthValue(
  depthMap: Float32Array,
  u: number,
  v: number,
  width: number,
  height: number
): number | null {
  const x = Math.round(u);
  const y = Math.round(v);
  
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return null;
  }
  
  const index = y * width + x;
  const depth = depthMap[index];
  
  if (!isFinite(depth) || depth <= 0) {
    return null;
  }
  
  return depth;
}

// ========== 凸包和几何工具函数 ==========

interface Point2D {
  x: number;
  y: number;
}

// 计算凸包（Graham扫描算法）
function computeConvexHull(points: Point2D[]): Point2D[] {
  if (points.length < 3) return points;
  
  // 找到最下方的点（y最大，如果相同则x最小）
  let start = 0;
  for (let i = 1; i < points.length; i++) {
    if (points[i].y > points[start].y || 
        (points[i].y === points[start].y && points[i].x < points[start].x)) {
      start = i;
    }
  }
  
  const startPoint = points[start];
  
  // 按极角排序
  const sorted = points
    .filter((_, i) => i !== start)
    .map(p => ({
      point: p,
      angle: Math.atan2(p.y - startPoint.y, p.x - startPoint.x)
    }))
    .sort((a, b) => a.angle - b.angle)
    .map(item => item.point);
  
  // Graham扫描
  const hull: Point2D[] = [startPoint];
  
  for (const p of sorted) {
    while (hull.length >= 2) {
      const a = hull[hull.length - 2];
      const b = hull[hull.length - 1];
      const cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
      if (cross <= 0) {
        hull.pop();
      } else {
        break;
      }
    }
    hull.push(p);
  }
  
  return hull;
}

// 判断点是否在多边形内（射线法）
function isPointInPolygon(point: Point2D, polygon: Point2D[]): boolean {
  let inside = false;
  const n = polygon.length;
  
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    
    if (((yi > point.y) !== (yj > point.y)) &&
        (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  
  return inside;
}

// ========== 2D 投影轮廓组件（使用模型轮廓而不是边界框） ==========

interface ProjectedMeshOutlineProps {
  modelUrl: string;
  pose: Matrix4;
  intrinsics: CameraIntrinsics;
  extrinsics: Matrix4;
  imageWidth: number;
  imageHeight: number;
  maskUrl?: string;
  onIoUCalculated?: (iou: number) => void;
}

function ProjectedMeshOutline({ modelUrl, pose, intrinsics, extrinsics, imageWidth, imageHeight, maskUrl, onIoUCalculated }: ProjectedMeshOutlineProps) {
  const [gltfScene, setGltfScene] = useState<THREE.Group | null>(null);
  const [loading, setLoading] = useState(true);
  const [maskData, setMaskData] = useState<ImageData | null>(null);
  
  // 使用GLTFLoader加载模型
  useEffect(() => {
    if (!modelUrl) {
      setLoading(false);
      return;
    }
    
    setLoading(true);
    
    const loader = new GLTFLoader();
    loader.load(
      modelUrl,
      (gltf) => {
        console.log('GLTF loaded successfully:', modelUrl);
        setGltfScene(gltf.scene);
        setLoading(false);
      },
      undefined,
      (err) => {
        console.error('Error loading GLTF:', err);
        setLoading(false);
      }
    );
  }, [modelUrl]);
  
  // 加载mask图像
  useEffect(() => {
    if (!maskUrl) {
      console.log('[IoU] No maskUrl provided');
      setMaskData(null);
      return;
    }
    
    console.log('[IoU] Loading mask from:', maskUrl);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      console.log('[IoU] Mask loaded:', img.width, 'x', img.height);
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0);
        const data = ctx.getImageData(0, 0, img.width, img.height);
        console.log('[IoU] Mask data extracted, size:', data.width, 'x', data.height);
        setMaskData(data);
      }
    };
    img.onerror = (err) => {
      console.error('[IoU] Failed to load mask:', err);
    };
    img.src = maskUrl;
  }, [maskUrl]);
  
  // 提取模型顶点和边，应用正确的坐标系转换
  const projectedData = useMemo<{ edges: { p1: { u: number; v: number }; p2: { u: number; v: number } }[]; vertices: { u: number; v: number }[] }>(() => {
    try {
      // 验证必要参数
      if (!gltfScene || !pose || !intrinsics || !extrinsics) {
        return { edges: [], vertices: [] };
      }
      
      if (intrinsics.fx === undefined || intrinsics.fy === undefined ||
          intrinsics.cx === undefined || intrinsics.cy === undefined) {
        console.warn('ProjectedMeshOutline: invalid intrinsics', intrinsics);
        return { edges: [], vertices: [] };
      }
      
      // 验证pose和extrinsics是有效的4x4矩阵
      if (!Array.isArray(pose) || pose.length !== 4 || 
          !pose.every(row => Array.isArray(row) && row.length === 4)) {
        console.warn('ProjectedMeshOutline: invalid pose matrix');
        return { edges: [], vertices: [] };
      }
      
      if (!Array.isArray(extrinsics) || extrinsics.length !== 4 || 
          !extrinsics.every(row => Array.isArray(row) && row.length === 4)) {
        console.warn('ProjectedMeshOutline: invalid extrinsics matrix');
        return { edges: [], vertices: [] };
      }
      
      // 使用队列遍历mesh
      const meshes: THREE.Mesh[] = [];
      const queue: THREE.Object3D[] = [gltfScene];
      const visited = new Set<THREE.Object3D>();
      
      while (queue.length > 0 && meshes.length < 100) {
        const obj = queue.shift()!;
        if (visited.has(obj)) continue;
        visited.add(obj);
        
        if (obj instanceof THREE.Mesh) {
          meshes.push(obj);
        }
        
        if (obj.children) {
          for (const child of obj.children) {
            if (!visited.has(child)) {
              queue.push(child);
            }
          }
        }
      }
      
      console.log(`Found ${meshes.length} meshes in model`);
      
      // 收集所有世界坐标系顶点
      const worldVertices: { x: number; y: number; z: number }[] = [];
      const allEdges: [number, number][] = [];
      let vertexOffset = 0;
      
      // 提取RT矩阵的R和t
      const R = [
        [extrinsics[0][0], extrinsics[0][1], extrinsics[0][2]],
        [extrinsics[1][0], extrinsics[1][1], extrinsics[1][2]],
        [extrinsics[2][0], extrinsics[2][1], extrinsics[2][2]]
      ];
      const t = [extrinsics[0][3], extrinsics[1][3], extrinsics[2][3]];
      
      // 提取world_pose的R和t
      const wpR = [
        [pose[0][0], pose[0][1], pose[0][2]],
        [pose[1][0], pose[1][1], pose[1][2]],
        [pose[2][0], pose[2][1], pose[2][2]]
      ];
      const wpT = [pose[0][3], pose[1][3], pose[2][3]];
      
      // 处理每个mesh
      for (const mesh of meshes) {
        const geometry = mesh.geometry;
        if (!geometry || !geometry.attributes) continue;
        
        const positions = geometry.attributes.position;
        if (!positions) continue;
        
        mesh.updateMatrixWorld(true);
        const meshMatrix = mesh.matrixWorld;
        
        const vertCount = positions.count;
        const maxTris = 10000;  // 增加三角形数量使线框更密集
        const meshVertices: { x: number; y: number; z: number }[] = [];
        
        // 提取所有顶点
        for (let i = 0; i < vertCount; i++) {
          const localX = positions.getX(i);
          const localY = positions.getY(i);
          const localZ = positions.getZ(i);
          
          // 应用mesh自身的变换
          const vec = new THREE.Vector3(localX, localY, localZ);
          vec.applyMatrix4(meshMatrix);
          
          // decoder mesh范围是(-0.5, 0.5)，而ultrashape mesh范围是(-1, 1)
          // 需要将decoder mesh缩放2倍以匹配原始world_pose
          const MESH_SCALE = 2.0;
          const scaledX = vec.x * MESH_SCALE;
          const scaledY = vec.y * MESH_SCALE;
          const scaledZ = vec.z * MESH_SCALE;
          
          // 应用world_pose: P_world = wpR @ P_scaled + wpT
          const worldX = wpR[0][0] * scaledX + wpR[0][1] * scaledY + wpR[0][2] * scaledZ + wpT[0];
          const worldY = wpR[1][0] * scaledX + wpR[1][1] * scaledY + wpR[1][2] * scaledZ + wpT[1];
          const worldZ = wpR[2][0] * scaledX + wpR[2][1] * scaledY + wpR[2][2] * scaledZ + wpT[2];
          
          meshVertices.push({ x: worldX, y: worldY, z: worldZ });
        }
        
        // 提取边 - 均匀采样
        const indices = geometry.index;
        
        if (indices) {
          const triCount = Math.floor(indices.count / 3);
          const step = Math.max(1, Math.ceil(triCount / maxTris));
          
          for (let ti = 0; ti < triCount && allEdges.length < 50000; ti += step) {
            const i = ti * 3;
            const a = indices.getX(i);
            const b = indices.getX(i + 1);
            const c = indices.getX(i + 2);
            allEdges.push([a + vertexOffset, b + vertexOffset]);
            allEdges.push([b + vertexOffset, c + vertexOffset]);
            allEdges.push([c + vertexOffset, a + vertexOffset]);
          }
        } else {
          const triCount = Math.floor(vertCount / 3);
          const step = Math.max(1, Math.ceil(triCount / maxTris));
          
          for (let ti = 0; ti < triCount && allEdges.length < 50000; ti += step) {
            const i = ti * 3;
            allEdges.push([i + vertexOffset, i + 1 + vertexOffset]);
            allEdges.push([i + 1 + vertexOffset, i + 2 + vertexOffset]);
            allEdges.push([i + 2 + vertexOffset, i + vertexOffset]);
          }
        }
        
        // 添加顶点
        for (let i = 0; i < meshVertices.length; i++) {
          worldVertices.push(meshVertices[i]);
        }
        vertexOffset += meshVertices.length;
      }
      
      if (worldVertices.length === 0) {
        console.log('No vertices found in model');
        return { edges: [], vertices: [] };
      }
      
      console.log(`Extracted ${worldVertices.length} vertices and ${allEdges.length} edges`);
      
      // 投影到图像平面
      // world → camera: P_cam = R.T @ (P_world - t)
      // camera → pixel: u = x*fx/z + cx, v = y*fy/z + cy
      const { fx, fy, cx, cy } = intrinsics;
      
      const projectedVerts = worldVertices.map(v => {
        // P_world - t
        const dx = v.x - t[0];
        const dy = v.y - t[1];
        const dz = v.z - t[2];
        
        // R.T @ (P_world - t)
        const camX = R[0][0] * dx + R[1][0] * dy + R[2][0] * dz;
        const camY = R[0][1] * dx + R[1][1] * dy + R[2][1] * dz;
        const camZ = R[0][2] * dx + R[1][2] * dy + R[2][2] * dz;
        
        // 检查是否在相机前方
        if (camZ <= 0.01) {
          return null;
        }
        
        // 投影到像素
        const u = camX * fx / camZ + cx;
        const v_coord = camY * fy / camZ + cy;
        
        return { u, v: v_coord };
      });
      
      // 过滤有效的边
      const validEdges = allEdges.filter(([a, b]) => {
        return a < projectedVerts.length && b < projectedVerts.length &&
               projectedVerts[a] !== null && projectedVerts[b] !== null;
      });
      
      console.log(`Valid edges: ${validEdges.length}`);
      
      // 收集所有有效的投影顶点（用于IoU计算）
      const validVerts = projectedVerts.filter((v): v is { u: number; v: number } => v !== null);
      
      const edges = validEdges.map(([a, b]) => ({
        p1: projectedVerts[a]!,
        p2: projectedVerts[b]!
      }));
      
      return { edges, vertices: validVerts };
    } catch (err) {
      console.error('Error in projectedEdges useMemo:', err instanceof Error ? err.message : String(err));
      return { edges: [], vertices: [] };
    }
  }, [gltfScene, pose, intrinsics, extrinsics]);
  
  // 计算IoU
  useEffect(() => {
    console.log('[IoU] useEffect triggered:', {
      hasMaskData: !!maskData,
      hasCallback: !!onIoUCalculated,
      verticesCount: projectedData.vertices.length
    });
    
    if (!maskData || !onIoUCalculated || projectedData.vertices.length === 0) {
      return;
    }
    
    const { vertices } = projectedData;
    const maskWidth = maskData.width;
    const maskHeight = maskData.height;
    
    // 计算投影点覆盖的像素集合（使用凸包填充）
    // 简化方法：使用投影点的边界框内的所有点，然后用射线法判断是否在凸包内
    
    // 1. 计算凸包
    const points = vertices.map((v: { u: number; v: number }) => ({ x: v.u, y: v.v }));
    const hull = computeConvexHull(points);
    
    if (hull.length < 3) {
      onIoUCalculated(0);
      return;
    }
    
    // 2. 计算边界框
    const minU = Math.max(0, Math.floor(Math.min(...hull.map(p => p.x))));
    const maxU = Math.min(imageWidth - 1, Math.ceil(Math.max(...hull.map(p => p.x))));
    const minV = Math.max(0, Math.floor(Math.min(...hull.map(p => p.y))));
    const maxV = Math.min(imageHeight - 1, Math.ceil(Math.max(...hull.map(p => p.y))));
    
    // 3. 遍历边界框内的像素，计算IoU
    let intersection = 0;
    let projectionArea = 0;
    let maskArea = 0;
    
    // 缩放因子（mask可能与图像尺寸不同）
    const scaleX = maskWidth / imageWidth;
    const scaleY = maskHeight / imageHeight;
    
    for (let v = minV; v <= maxV; v++) {
      for (let u = minU; u <= maxU; u++) {
        const inProjection = isPointInPolygon({ x: u, y: v }, hull);
        
        // 获取mask值（缩放坐标）
        const maskU = Math.round(u * scaleX);
        const maskV = Math.round(v * scaleY);
        const maskIdx = (maskV * maskWidth + maskU) * 4;
        const maskValue = maskData.data[maskIdx]; // R通道
        const inMask = maskValue > 127;
        
        if (inProjection) projectionArea++;
        if (inMask) maskArea++;
        if (inProjection && inMask) intersection++;
      }
    }
    
    // 也需要统计边界框外的mask区域
    for (let v = 0; v < imageHeight; v++) {
      for (let u = 0; u < imageWidth; u++) {
        if (u >= minU && u <= maxU && v >= minV && v <= maxV) continue; // 已统计
        
        const maskU = Math.round(u * scaleX);
        const maskV = Math.round(v * scaleY);
        const maskIdx = (maskV * maskWidth + maskU) * 4;
        const maskValue = maskData.data[maskIdx];
        if (maskValue > 127) maskArea++;
      }
    }
    
    const union = projectionArea + maskArea - intersection;
    const iou = union > 0 ? intersection / union : 0;
    
    console.log(`IoU计算: intersection=${intersection}, projectionArea=${projectionArea}, maskArea=${maskArea}, union=${union}, IoU=${iou.toFixed(4)}`);
    onIoUCalculated(iou);
  }, [maskData, projectedData, onIoUCalculated, imageWidth, imageHeight]);
  
  if (loading) {
    return null;
  }
  
  if (projectedData.edges.length === 0) {
    return null;
  }
  
  return (
    <svg 
      className="absolute inset-0 w-full h-full pointer-events-none" 
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* 绘制模型边缘 */}
      {projectedData.edges.map((edge: { p1: { u: number; v: number }; p2: { u: number; v: number } }, idx: number) => (
        <line
          key={idx}
          x1={edge.p1.u}
          y1={edge.p1.v}
          x2={edge.p2.u}
          y2={edge.p2.v}
          stroke="#00aa00"
          strokeWidth="1"
          opacity="0.6"
        />
      ))}
    </svg>
  );
}

// ========== 2D 点标记叠加层 ==========

interface PixelMarkerOverlayProps {
  width: number;
  height: number;
}

function PixelMarkerOverlay({ width, height }: PixelMarkerOverlayProps) {
  const pointPairs = useAnnotationStore((state) => state.pointPairs);
  const pendingWorldPoint = useAnnotationStore((state) => state.pendingWorldPoint);
  const selectedPairId = useAnnotationStore((state) => state.selectedPairId);
  
  // 使用SVG确保坐标精确对应
  // 注意：必须使用xMidYMid meet保持与图像相同的宽高比
  return (
    <svg 
      className="absolute inset-0 w-full h-full pointer-events-none" 
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {pointPairs.map((pair, index) => (
        <g key={pair.id}>
          <circle
            cx={pair.pixelCoord.u}
            cy={pair.pixelCoord.v}
            r="8"
            fill={pair.id === selectedPairId ? '#facc15' : '#4ade80'}
            stroke={pair.id === selectedPairId ? '#ca8a04' : '#16a34a'}
            strokeWidth="2"
          />
          <text
            x={pair.pixelCoord.u}
            y={pair.pixelCoord.v - 15}
            textAnchor="middle"
            fill="white"
            fontSize="14"
            fontWeight="bold"
            style={{ textShadow: '1px 1px 2px black' }}
          >
            {index + 1}
          </text>
        </g>
      ))}
      {pendingWorldPoint && (
        <circle
          cx={pendingWorldPoint.pixel.u}
          cy={pendingWorldPoint.pixel.v}
          r="10"
          fill="#f97316"
          stroke="#c2410c"
          strokeWidth="2"
        />
      )}
    </svg>
  );
}

// ========== 主组件 ==========

interface WorldViewProps {
  imageUrl?: string;
  maskUrl?: string;
  modelUrl?: string;
  depthMap?: Float32Array;
  depthWidth?: number;
  depthHeight?: number;
  cameraIntrinsics?: CameraIntrinsics;
  cameraExtrinsics?: Matrix4;
  initialPose?: Matrix4;
}

export function WorldView({
  imageUrl,
  maskUrl,
  modelUrl,
  depthMap,
  depthWidth = 512,
  depthHeight = 384,
  cameraIntrinsics,
  cameraExtrinsics,
  initialPose
}: WorldViewProps) {
  const isEnabled = useAnnotationStore((state) => state.isLeftViewEnabled);
  const showGhostWireframe = useAnnotationStore((state) => state.showGhostWireframe);
  const maskOpacity = useAnnotationStore((state) => state.maskOpacity);
  const calculatedPose = useAnnotationStore((state) => state.calculatedPose);
  const setPendingWorldPoint = useAnnotationStore((state) => state.setPendingWorldPoint);
  const pendingLocalPoint = useAnnotationStore((state) => state.pendingLocalPoint);
  const currentIoU = useAnnotationStore((state) => state.currentIoU);
  const setCurrentIoU = useAnnotationStore((state) => state.setCurrentIoU);
  
  // 追踪实际图像尺寸
  const [actualImageSize, setActualImageSize] = useState<{width: number, height: number} | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  
  // 缩放状态
  const [zoomLevel, setZoomLevel] = useState(1);
  const zoomLevels = [1, 1.5, 2, 3, 4];
  
  const handleZoomIn = () => {
    const currentIndex = zoomLevels.indexOf(zoomLevel);
    if (currentIndex < zoomLevels.length - 1) {
      setZoomLevel(zoomLevels[currentIndex + 1]);
    }
  };
  
  const handleZoomOut = () => {
    const currentIndex = zoomLevels.indexOf(zoomLevel);
    if (currentIndex > 0) {
      setZoomLevel(zoomLevels[currentIndex - 1]);
    }
  };
  
  const handleZoomReset = () => {
    setZoomLevel(1);
  };
  
  // 使用计算后的位姿或初始位姿
  const currentPose = calculatedPose ?? initialPose;
  
  const handleImageClick = useCallback((u: number, v: number, naturalWidth: number, naturalHeight: number) => {
    if (!depthMap || !cameraIntrinsics || !cameraExtrinsics) {
      console.warn('Missing depth map or camera parameters');
      return;
    }
    
    // 深度图尺寸与图像尺寸不同，需要缩放坐标
    const depthU = u * (depthWidth / naturalWidth);
    const depthV = v * (depthHeight / naturalHeight);
    
    // 获取深度值（使用缩放后的坐标）
    const depth = getDepthValue(depthMap, depthU, depthV, depthWidth, depthHeight);
    if (depth === null) {
      console.warn('Invalid depth at pixel:', u, v, '-> depth coord:', depthU, depthV);
      return;
    }
    
    console.log('Click at image:', u, v, '-> depth coord:', depthU, depthV, 'depth:', depth);
    
    // 反投影到世界坐标系
    // 注意：cameraExtrinsics 在输入中定义为 World-to-Camera
    // 但根据 COORDINATE_SYSTEM_SUMMARY.md，RT 是 camera-to-world
    // 这里假设传入的是 camera-to-world，如果是 world-to-camera 需要取逆
    const worldPoint = unprojectPoint(u, v, depth, cameraIntrinsics, cameraExtrinsics);
    
    console.log('World point:', worldPoint, 'from pixel:', u, v, 'depth:', depth);
    
    setPendingWorldPoint({
      point: worldPoint,
      pixel: { u, v },
      depth
    });
  }, [depthMap, depthWidth, depthHeight, cameraIntrinsics, cameraExtrinsics, setPendingWorldPoint]);
  
  // 使用实际图像尺寸，如果还没加载则使用cameraIntrinsics的值
  const imageWidth = actualImageSize?.width ?? cameraIntrinsics?.width ?? 512;
  const imageHeight = actualImageSize?.height ?? cameraIntrinsics?.height ?? 384;
  
  return (
    <div className="relative w-full h-full bg-gray-900 rounded-lg overflow-hidden">
      {/* 标题栏 */}
      <div className="absolute top-0 left-0 right-0 z-10 bg-gray-800/80 backdrop-blur px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-white font-medium">世界空间</span>
          <span className="text-gray-400 text-sm">(World Space)</span>
        </div>
        <div className="flex items-center gap-2">
          {/* 缩放控制按钮 */}
          <div className="flex items-center gap-1 bg-gray-700 rounded px-1">
            <button
              onClick={handleZoomOut}
              disabled={zoomLevel <= 1}
              className="px-2 py-1 text-white hover:bg-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              title="缩小"
            >
              −
            </button>
            <button
              onClick={handleZoomReset}
              className="px-2 py-1 text-white hover:bg-gray-600 rounded text-xs min-w-[50px]"
              title="重置缩放"
            >
              {Math.round(zoomLevel * 100)}%
            </button>
            <button
              onClick={handleZoomIn}
              disabled={zoomLevel >= 4}
              className="px-2 py-1 text-white hover:bg-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              title="放大"
            >
              +
            </button>
          </div>
          <div className={`px-2 py-1 rounded text-xs ${isEnabled ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'}`}>
            {isEnabled ? '可交互' : '已禁用'}
          </div>
        </div>
      </div>
      
      {/* 提示信息 */}
      {isEnabled && pendingLocalPoint && (
        <div className="absolute top-12 left-4 right-4 z-10 bg-blue-600/80 backdrop-blur text-white px-3 py-2 rounded text-sm">
          请在图像上点击对应的特征点位置
        </div>
      )}
      
      {/* 禁用遮罩 */}
      {!isEnabled && (
        <div className="absolute inset-0 z-20 bg-black/50 flex items-center justify-center">
          <div className="text-gray-400 text-center">
            <p className="text-lg mb-2">交互已禁用</p>
            <p className="text-sm">请先完成物体分类</p>
          </div>
        </div>
      )}
      
      {/* 图像 + 3D 叠加 */}
      <div className="relative w-full h-full pt-12">
        {imageUrl ? (
          <div 
            className="absolute inset-0 pt-12 bg-black"
            style={{ overflow: zoomLevel > 1 ? 'auto' : 'hidden' }}
          >
            {/* 图像容器 */}
            <div 
              className="w-full h-full flex items-center justify-center"
              style={{ 
                minWidth: zoomLevel > 1 ? 'fit-content' : '100%',
                minHeight: zoomLevel > 1 ? 'fit-content' : '100%',
              }}
            >
              {/* 图像容器 - 使用相对定位让叠加层正确对齐 */}
              <div className="relative" style={{ display: 'inline-block' }}>
                <img 
                  ref={imgRef}
                  src={imageUrl} 
                  alt="RGB" 
                  className="block cursor-crosshair"
                  style={{ 
                    width: zoomLevel > 1 ? `${(actualImageSize?.width ?? 512) * zoomLevel}px` : 'auto',
                    height: zoomLevel > 1 ? `${(actualImageSize?.height ?? 384) * zoomLevel}px` : 'auto',
                    maxWidth: zoomLevel === 1 ? '100%' : 'none',
                    maxHeight: zoomLevel === 1 ? 'calc(100vh - 120px)' : 'none',
                  }}
                onLoad={(e) => {
                  const img = e.currentTarget;
                  setActualImageSize({ width: img.naturalWidth, height: img.naturalHeight });
                }}
                onClick={(e) => {
                  if (!isEnabled) return;
                  const img = e.currentTarget;
                  const rect = img.getBoundingClientRect();
                  
                  // 使用图像的实际尺寸
                  const naturalWidth = img.naturalWidth;
                  const naturalHeight = img.naturalHeight;
                  
                  // 图像现在直接显示，rect就是实际显示尺寸
                  const displayWidth = rect.width;
                  const displayHeight = rect.height;
                  
                  // 计算点击位置相对于图像的坐标
                  const clickX = e.clientX - rect.left;
                  const clickY = e.clientY - rect.top;
                  
                  // 转换为图像原始坐标
                  const u = (clickX / displayWidth) * naturalWidth;
                  const v = (clickY / displayHeight) * naturalHeight;
                  
                  console.log('Image click:', { 
                    clientX: e.clientX, clientY: e.clientY,
                    rect: { left: rect.left, top: rect.top, width: rect.width, height: rect.height },
                    natural: { width: naturalWidth, height: naturalHeight },
                    click: { x: clickX, y: clickY },
                    imageCoord: { u, v }
                  });
                  
                  handleImageClick(u, v, naturalWidth, naturalHeight);
                }}
              />
              {/* Mask叠加层 - 与图像完全重叠 */}
              {maskUrl && (
                <img 
                  src={maskUrl} 
                  alt="Mask"
                  className="absolute top-0 left-0 w-full h-full pointer-events-none"
                  style={{ 
                    mixBlendMode: 'multiply',
                    opacity: maskOpacity * 0.5
                  }}
                />
              )}
              {/* 幽灵线框叠加层 - mesh投影轮廓 */}
              {showGhostWireframe && modelUrl && currentPose && cameraIntrinsics && cameraExtrinsics && (
                <ProjectedMeshOutline 
                  modelUrl={modelUrl}
                  pose={currentPose}
                  intrinsics={cameraIntrinsics}
                  extrinsics={cameraExtrinsics}
                  imageWidth={imageWidth}
                  imageHeight={imageHeight}
                  maskUrl={maskUrl}
                  onIoUCalculated={setCurrentIoU}
                />
              )}
              {/* 像素坐标点标记叠加层 */}
              <PixelMarkerOverlay width={imageWidth} height={imageHeight} />
            </div>
            </div>
          </div>
        ) : (
          <div className="absolute inset-0 pt-12 flex items-center justify-center text-gray-500">
            无图像数据
          </div>
        )}
      </div>
      
      {/* 状态信息 */}
      <div className="absolute bottom-4 left-4 text-xs text-gray-400">
        <div>图像尺寸: {imageWidth}×{imageHeight}</div>
        <div>点击图像获取世界坐标</div>
        {currentIoU !== null && (
          <div className={`mt-1 font-bold ${currentIoU > 0.5 ? 'text-green-400' : currentIoU > 0.3 ? 'text-yellow-400' : 'text-red-400'}`}>
            IoU: {(currentIoU * 100).toFixed(1)}%
          </div>
        )}
      </div>
    </div>
  );
}

export default WorldView;
