/**
 * 多视角标注 - 单帧视图组件
 * 
 * 显示单帧图像，支持点击标注关键点，显示mesh投影
 */

import { useCallback, useMemo, useState, useEffect } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { useMVAnnotationStore } from '../stores/mvAnnotationStore';
import { unprojectPoint } from '../utils/math';
import type { Matrix4, CameraIntrinsics } from '../types';
import type { FrameData, MVPointPair } from '../types/multiview';

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

// ========== 2D 投影轮廓组件 ==========

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

// 计算凸包 (Graham Scan算法)
function computeConvexHull(points: { x: number; y: number }[]): { x: number; y: number }[] {
  if (points.length < 3) return points;
  
  // 找到最下最左的点
  let start = 0;
  for (let i = 1; i < points.length; i++) {
    if (points[i].y < points[start].y || 
        (points[i].y === points[start].y && points[i].x < points[start].x)) {
      start = i;
    }
  }
  
  const pivot = points[start];
  
  // 按极角排序
  const sorted = points
    .filter((_, i) => i !== start)
    .map(p => ({
      point: p,
      angle: Math.atan2(p.y - pivot.y, p.x - pivot.x)
    }))
    .sort((a, b) => a.angle - b.angle)
    .map(p => p.point);
  
  const hull: { x: number; y: number }[] = [pivot];
  
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

// 缓存加载的模型
const modelCache = new Map<string, THREE.Group>();

function ProjectedMeshOutline({ modelUrl, pose, intrinsics, extrinsics, imageWidth, imageHeight, maskUrl, onIoUCalculated }: ProjectedMeshOutlineProps) {
  const [gltfScene, setGltfScene] = useState<THREE.Group | null>(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    if (!modelUrl) {
      setLoading(false);
      return;
    }
    
    // 检查缓存
    if (modelCache.has(modelUrl)) {
      setGltfScene(modelCache.get(modelUrl)!.clone());
      setLoading(false);
      return;
    }
    
    setLoading(true);
    const loader = new GLTFLoader();
    loader.load(
      modelUrl,
      (gltf) => {
        modelCache.set(modelUrl, gltf.scene);
        setGltfScene(gltf.scene.clone());
        setLoading(false);
      },
      undefined,
      (err) => {
        console.error('Error loading GLTF:', err);
        setLoading(false);
      }
    );
  }, [modelUrl]);
  
  // 计算投影凸包和边界框
  const projectedData = useMemo<{ hull: { x: number; y: number }[]; bbox: { minU: number; minV: number; maxU: number; maxV: number } | null }>(() => {
    if (!gltfScene || !pose || !intrinsics || !extrinsics) {
      return { hull: [], bbox: null };
    }
    
    if (intrinsics.fx === undefined || intrinsics.fy === undefined ||
        intrinsics.cx === undefined || intrinsics.cy === undefined) {
      return { hull: [], bbox: null };
    }
    
    if (!Array.isArray(pose) || pose.length !== 4 || 
        !pose.every(row => Array.isArray(row) && row.length === 4)) {
      return { hull: [], bbox: null };
    }
    
    if (!Array.isArray(extrinsics) || extrinsics.length !== 4 || 
        !extrinsics.every(row => Array.isArray(row) && row.length === 4)) {
      return { hull: [], bbox: null };
    }
    
    // 收集mesh
    const meshes: THREE.Mesh[] = [];
    const queue: THREE.Object3D[] = [gltfScene];
    const visited = new Set<THREE.Object3D>();
    
    while (queue.length > 0 && meshes.length < 10) {
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
    
    const R = [
      [extrinsics[0][0], extrinsics[0][1], extrinsics[0][2]],
      [extrinsics[1][0], extrinsics[1][1], extrinsics[1][2]],
      [extrinsics[2][0], extrinsics[2][1], extrinsics[2][2]]
    ];
    const t = [extrinsics[0][3], extrinsics[1][3], extrinsics[2][3]];
    
    const wpR = [
      [pose[0][0], pose[0][1], pose[0][2]],
      [pose[1][0], pose[1][1], pose[1][2]],
      [pose[2][0], pose[2][1], pose[2][2]]
    ];
    const wpT = [pose[0][3], pose[1][3], pose[2][3]];
    
    const { fx, fy, cx, cy } = intrinsics;
    const projectedPoints: { x: number; y: number }[] = [];
    
    // 采样顶点（大幅减少计算量）
    for (const mesh of meshes) {
      const geometry = mesh.geometry;
      if (!geometry || !geometry.attributes) continue;
      
      const positions = geometry.attributes.position;
      if (!positions) continue;
      
      mesh.updateMatrixWorld(true);
      const meshMatrix = mesh.matrixWorld;
      
      const vertCount = positions.count;
      // 最多采样500个顶点
      const step = Math.max(1, Math.ceil(vertCount / 500));
      
      for (let i = 0; i < vertCount; i += step) {
        const localX = positions.getX(i);
        const localY = positions.getY(i);
        const localZ = positions.getZ(i);
        
        const vec = new THREE.Vector3(localX, localY, localZ);
        vec.applyMatrix4(meshMatrix);
        
        const MESH_SCALE = 2.0;
        const scaledX = vec.x * MESH_SCALE;
        const scaledY = vec.y * MESH_SCALE;
        const scaledZ = vec.z * MESH_SCALE;
        
        const worldX = wpR[0][0] * scaledX + wpR[0][1] * scaledY + wpR[0][2] * scaledZ + wpT[0];
        const worldY = wpR[1][0] * scaledX + wpR[1][1] * scaledY + wpR[1][2] * scaledZ + wpT[1];
        const worldZ = wpR[2][0] * scaledX + wpR[2][1] * scaledY + wpR[2][2] * scaledZ + wpT[2];
        
        // 投影到图像
        const dx = worldX - t[0];
        const dy = worldY - t[1];
        const dz = worldZ - t[2];
        
        const camX = R[0][0] * dx + R[1][0] * dy + R[2][0] * dz;
        const camY = R[0][1] * dx + R[1][1] * dy + R[2][1] * dz;
        const camZ = R[0][2] * dx + R[1][2] * dy + R[2][2] * dz;
        
        if (camZ > 0.01) {
          const u = camX * fx / camZ + cx;
          const v = camY * fy / camZ + cy;
          projectedPoints.push({ x: u, y: v });
        }
      }
    }
    
    if (projectedPoints.length < 3) {
      return { hull: [], bbox: null };
    }
    
    // 计算凸包
    const hull = computeConvexHull(projectedPoints);
    
    // 计算边界框
    const minU = Math.min(...projectedPoints.map(p => p.x));
    const maxU = Math.max(...projectedPoints.map(p => p.x));
    const minV = Math.min(...projectedPoints.map(p => p.y));
    const maxV = Math.max(...projectedPoints.map(p => p.y));
    
    return { hull, bbox: { minU, minV, maxU, maxV } };
  }, [gltfScene, pose, intrinsics, extrinsics]);
  
  // IoU计算（基于凸包采样）
  useEffect(() => {
    if (!maskUrl || !onIoUCalculated || !projectedData.bbox || projectedData.hull.length < 3) return;
    
    const { hull, bbox } = projectedData;
    
    // 加载mask图像
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      ctx.drawImage(img, 0, 0);
      const maskData = ctx.getImageData(0, 0, img.width, img.height);
      
      // 缩放因子
      const scaleX = img.width / imageWidth;
      const scaleY = img.height / imageHeight;
      
      // 采样计算IoU
      const sampleStep = Math.max(4, Math.floor(Math.max(imageWidth, imageHeight) / 100));
      
      let intersection = 0;
      let projectionArea = 0;
      let maskArea = 0;
      
      // 点是否在凸包内
      const isInHull = (px: number, py: number): boolean => {
        for (let i = 0; i < hull.length; i++) {
          const a = hull[i];
          const b = hull[(i + 1) % hull.length];
          const cross = (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
          if (cross < 0) return false;
        }
        return true;
      };
      
      // 在边界框范围内采样
      const minU = Math.max(0, Math.floor(bbox.minU));
      const maxU = Math.min(imageWidth - 1, Math.ceil(bbox.maxU));
      const minV = Math.max(0, Math.floor(bbox.minV));
      const maxV = Math.min(imageHeight - 1, Math.ceil(bbox.maxV));
      
      for (let v = minV; v <= maxV; v += sampleStep) {
        for (let u = minU; u <= maxU; u += sampleStep) {
          const inProjection = isInHull(u, v);
          
          const maskU = Math.round(u * scaleX);
          const maskV = Math.round(v * scaleY);
          const maskIdx = (maskV * img.width + maskU) * 4;
          const inMask = maskData.data[maskIdx] > 127;
          
          if (inProjection) projectionArea++;
          if (inMask) maskArea++;
          if (inProjection && inMask) intersection++;
        }
      }
      
      // 边界框外的mask区域
      for (let v = 0; v < imageHeight; v += sampleStep) {
        for (let u = 0; u < imageWidth; u += sampleStep) {
          if (u >= minU && u <= maxU && v >= minV && v <= maxV) continue;
          
          const maskU = Math.round(u * scaleX);
          const maskV = Math.round(v * scaleY);
          const maskIdx = (maskV * img.width + maskU) * 4;
          if (maskData.data[maskIdx] > 127) maskArea++;
        }
      }
      
      const union = projectionArea + maskArea - intersection;
      const iou = union > 0 ? intersection / union : 0;
      
      onIoUCalculated(iou);
    };
    img.src = maskUrl;
  }, [maskUrl, onIoUCalculated, projectedData, imageWidth, imageHeight]);
  
  if (loading || projectedData.hull.length < 3) {
    return null;
  }
  
  const { hull, bbox } = projectedData;
  const hullPath = hull.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z';
  
  return (
    <svg 
      className="absolute inset-0 w-full h-full pointer-events-none" 
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* 黄色边界框 */}
      {bbox && (
        <rect
          x={bbox.minU}
          y={bbox.minV}
          width={bbox.maxU - bbox.minU}
          height={bbox.maxV - bbox.minV}
          fill="none"
          stroke="#ffcc00"
          strokeWidth="2"
          strokeDasharray="6,4"
          opacity="0.8"
        />
      )}
      {/* 凸包轮廓 - 显示mesh投影形状 */}
      <path
        d={hullPath}
        fill="rgba(0, 200, 0, 0.1)"
        stroke="#00ff00"
        strokeWidth="2"
        opacity="0.8"
      />
    </svg>
  );
}

// ========== 像素坐标点标记叠加层 ==========

interface PixelMarkerOverlayProps {
  width: number;
  height: number;
  frameId: string;
  pointPairs: MVPointPair[];
  pendingWorldPoint: { pixel: { u: number; v: number }; frameId: string } | null;
}

function PixelMarkerOverlay({ width, height, frameId, pointPairs, pendingWorldPoint }: PixelMarkerOverlayProps) {
  const framePairs = pointPairs.filter(p => p.frame_id === frameId);
  const selectedPairId = useMVAnnotationStore((state) => state.selectedPairId);
  
  return (
    <svg 
      className="absolute inset-0 w-full h-full pointer-events-none" 
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {framePairs.map((pair) => (
        <g key={pair.id}>
          <circle
            cx={pair.pixelCoord.u}
            cy={pair.pixelCoord.v}
            r="6"
            fill={pair.id === selectedPairId ? '#facc15' : '#4ade80'}
            stroke={pair.id === selectedPairId ? '#ca8a04' : '#16a34a'}
            strokeWidth="2"
          />
          <text
            x={pair.pixelCoord.u}
            y={pair.pixelCoord.v - 10}
            textAnchor="middle"
            fill="white"
            fontSize="12"
            fontWeight="bold"
            style={{ textShadow: '1px 1px 2px black' }}
          >
            {pointPairs.indexOf(pair) + 1}
          </text>
        </g>
      ))}
      {pendingWorldPoint && pendingWorldPoint.frameId === frameId && (
        <circle
          cx={pendingWorldPoint.pixel.u}
          cy={pendingWorldPoint.pixel.v}
          r="8"
          fill="#f97316"
          stroke="#c2410c"
          strokeWidth="2"
        />
      )}
    </svg>
  );
}

// ========== 主组件 ==========

interface MVFrameViewProps {
  frame: FrameData;
  modelUrl: string;
  pose: Matrix4 | null;
  isActive: boolean;
  onSelect: () => void;
  serverUrl: string;
  enlarged?: boolean;
}

export function MVFrameView({ frame, modelUrl, pose, isActive, onSelect, serverUrl, enlarged = false }: MVFrameViewProps) {
  const isEnabled = useMVAnnotationStore((state) => state.isAnnotationEnabled);
  const showGhostWireframe = useMVAnnotationStore((state) => state.showGhostWireframe);
  const maskOpacity = useMVAnnotationStore((state) => state.maskOpacity);
  const pointPairs = useMVAnnotationStore((state) => state.pointPairs);
  const pendingWorldPoint = useMVAnnotationStore((state) => state.pendingWorldPoint);
  const pendingLocalPoint = useMVAnnotationStore((state) => state.pendingLocalPoint);
  const setPendingWorldPoint = useMVAnnotationStore((state) => state.setPendingWorldPoint);
  const updateFrameDepth = useMVAnnotationStore((state) => state.updateFrameDepth);
  const setFrameIoU = useMVAnnotationStore((state) => state.setFrameIoU);
  
  const [actualImageSize, setActualImageSize] = useState<{width: number, height: number} | null>(null);
  const [depthLoading, setDepthLoading] = useState(false);
  
  // IoU计算回调
  const handleIoUCalculated = useCallback((iou: number) => {
    setFrameIoU(frame.frame_id, iou);
  }, [frame.frame_id, setFrameIoU]);
  
  // 加载深度图
  useEffect(() => {
    if (frame.depth_url && !frame.depthMap && !depthLoading) {
      setDepthLoading(true);
      const depthUrl = `${serverUrl}${frame.depth_url}`;
      console.log(`[MVFrameView] 加载深度图: ${depthUrl}`);
      fetch(depthUrl)
        .then(res => {
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
          }
          return res.json();
        })
        .then(data => {
          if (!data.data || !data.width || !data.height) {
            throw new Error('Invalid depth data format');
          }
          const depthMap = new Float32Array(data.data);
          console.log(`[MVFrameView] 深度图加载成功: ${data.width}x${data.height}`);
          updateFrameDepth(frame.frame_id, depthMap, data.width, data.height);
          setDepthLoading(false);
        })
        .catch(err => {
          console.error(`[MVFrameView] 深度图加载失败 (${frame.frame_id}):`, err);
          setDepthLoading(false);
        });
    }
  }, [frame.depth_url, frame.depthMap, frame.frame_id, serverUrl, updateFrameDepth, depthLoading]);
  
  const handleImageClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.stopPropagation(); // 阻止事件冒泡到父元素
    
    console.log('[MVFrameView] 图像点击事件触发', { isEnabled, isActive, hasDepth: !!frame.depthMap });
    
    if (!isEnabled) {
      alert('请先点击右侧"需要对齐"按钮启用标注');
      return;
    }
    if (!isActive) {
      alert('请先点击选中此帧（点击帧的标题栏）');
      return;
    }
    
    const clickLayer = e.currentTarget;
    const rect = clickLayer.getBoundingClientRect();
    
    // 使用actualImageSize或默认值
    const naturalWidth = actualImageSize?.width ?? frame.camera_intrinsics.width ?? 1024;
    const naturalHeight = actualImageSize?.height ?? frame.camera_intrinsics.height ?? 768;
    const displayWidth = rect.width;
    const displayHeight = rect.height;
    
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    const u = (clickX / displayWidth) * naturalWidth;
    const v = (clickY / displayHeight) * naturalHeight;
    
    console.log(`[MVFrameView] 点击坐标: display=(${clickX.toFixed(1)}, ${clickY.toFixed(1)}), image=(${u.toFixed(1)}, ${v.toFixed(1)})`);
    
    // 获取深度
    if (!frame.depthMap) {
      console.warn('[MVFrameView] 深度图未加载，尝试使用默认深度');
      // 使用默认深度值继续（允许用户在深度图加载前标注）
      const defaultDepth = 1.5; // 默认1.5米
      
      const K = frame.camera_intrinsics.K;
      if (!K) {
        alert('相机参数缺失，无法标注');
        return;
      }
      
      const intrinsics: CameraIntrinsics = {
        fx: K[0][0],
        fy: K[1][1],
        cx: K[0][2],
        cy: K[1][2],
        width: frame.camera_intrinsics.width,
        height: frame.camera_intrinsics.height
      };
      
      const worldPoint = unprojectPoint(u, v, defaultDepth, intrinsics, frame.camera_extrinsics);
      console.log(`[MVFrameView] 使用默认深度 ${defaultDepth}m, world=`, worldPoint);
      
      setPendingWorldPoint({
        point: worldPoint,
        pixel: { u, v },
        depth: defaultDepth,
        frameId: frame.frame_id
      });
      return;
    }
    
    const depthU = u * ((frame.depthWidth ?? 512) / naturalWidth);
    const depthV = v * ((frame.depthHeight ?? 384) / naturalHeight);
    
    let depth = getDepthValue(
      frame.depthMap, 
      depthU, 
      depthV, 
      frame.depthWidth ?? 512, 
      frame.depthHeight ?? 384
    );
    
    // 如果深度无效，直接提示用户
    if (depth === null) {
      console.warn('Invalid depth at pixel:', u, v);
      alert('该位置没有有效深度值，请选择其他位置');
      return;
    }
    
    // 构建相机内参
    const K = frame.camera_intrinsics.K;
    if (!K) {
      console.warn('No camera intrinsics for frame:', frame.frame_id);
      return;
    }
    
    const intrinsics: CameraIntrinsics = {
      fx: K[0][0],
      fy: K[1][1],
      cx: K[0][2],
      cy: K[1][2],
      width: frame.camera_intrinsics.width,
      height: frame.camera_intrinsics.height
    };
    
    // 反投影到世界坐标系
    const worldPoint = unprojectPoint(u, v, depth, intrinsics, frame.camera_extrinsics);
    
    console.log(`[Frame ${frame.frame_id}] Click at (${u.toFixed(1)}, ${v.toFixed(1)}), depth=${depth.toFixed(3)}, world=`, worldPoint);
    
    setPendingWorldPoint({
      point: worldPoint,
      pixel: { u, v },
      depth,
      frameId: frame.frame_id
    });
  }, [isEnabled, isActive, frame, setPendingWorldPoint, actualImageSize]);
  
  const imageWidth = actualImageSize?.width ?? frame.camera_intrinsics.width ?? 512;
  const imageHeight = actualImageSize?.height ?? frame.camera_intrinsics.height ?? 384;
  
  // 构建相机内参用于投影
  const intrinsics: CameraIntrinsics | null = frame.camera_intrinsics.K ? {
    fx: frame.camera_intrinsics.K[0][0],
    fy: frame.camera_intrinsics.K[1][1],
    cx: frame.camera_intrinsics.K[0][2],
    cy: frame.camera_intrinsics.K[1][2],
    width: frame.camera_intrinsics.width,
    height: frame.camera_intrinsics.height
  } : null;
  
  const framePointCount = pointPairs.filter(p => p.frame_id === frame.frame_id).length;
  
  // 放大视图时的样式
  const containerClass = enlarged 
    ? 'relative bg-gray-800 rounded overflow-hidden'
    : `relative bg-gray-800 rounded overflow-hidden cursor-pointer transition-all ${
        isActive ? 'ring-2 ring-blue-500' : 'ring-1 ring-gray-600 hover:ring-gray-400'
      }`;
  
  return (
    <div 
      className={containerClass}
      onClick={enlarged ? undefined : onSelect}
    >
      {/* 缩略图视图 - 标题和提示 */}
      {!enlarged && (
        <div className="flex flex-col">
          <div className="bg-gray-900/80 px-2 py-1 flex items-center justify-between">
            <span className="text-xs text-gray-300 truncate">
              #{frame.frame_index} - {frame.frame_id.slice(-8)}
            </span>
            <div className="flex items-center gap-1">
              {framePointCount > 0 && (
                <span className="px-1 bg-green-600 text-white text-xs rounded">
                  {framePointCount}点
                </span>
              )}
              {isActive && (
                <span className="px-1 bg-blue-600 text-white text-xs rounded">
                  活动
                </span>
              )}
            </div>
          </div>
          
          {/* 状态提示条 */}
          {!isEnabled && (
            <div className="bg-gray-600/90 text-gray-300 px-2 py-1 text-xs text-center">
              请先点击"需要对齐"启用标注
            </div>
          )}
          {isEnabled && !isActive && (
            <div className="bg-yellow-600/90 text-white px-2 py-1 text-xs text-center">
              点击选中此帧
            </div>
          )}
        </div>
      )}
      
      {/* 放大视图 - 标题和提示（flex布局，不覆盖图像） */}
      {enlarged && (
        <div className="w-full shrink-0">
          <div className="bg-gray-900 px-2 py-1 flex items-center justify-between">
            <span className="text-sm text-gray-300 truncate">
              #{frame.frame_index} - {frame.frame_id.slice(-8)}
              <span className="ml-2 text-blue-400">(放大视图)</span>
            </span>
            <div className="flex items-center gap-1">
              {framePointCount > 0 && (
                <span className="px-1 bg-green-600 text-white text-sm rounded">
                  {framePointCount}点
                </span>
              )}
            </div>
          </div>
          
          {/* 状态提示条 - 放大视图 */}
          {isEnabled && (
            <div className={`${pendingLocalPoint ? 'bg-blue-600' : 'bg-green-600'} text-white px-2 py-1.5 text-xs text-center`}>
              {pendingLocalPoint ? '✓ 已选择模型点 → 请点击图像上的对应位置' : '请先在右侧模型上点击，或直接在图像上点击标注'}
            </div>
          )}
        </div>
      )}
      
      {/* 图像区域 */}
      <div className={`${enlarged ? 'flex items-center justify-center p-2' : ''}`}>
        {/* 图像包装器 - 确保叠加层与图像对齐 */}
        <div className="relative" style={{ display: 'inline-block' }}>
          {/* 主图像 - 底层 */}
          <img
            src={`${serverUrl}${frame.rgb_url}`}
            alt={`Frame ${frame.frame_id}`}
            className={enlarged ? 'max-w-full h-auto block' : 'w-full h-auto block'}
            style={enlarged ? { maxHeight: '60vh' } : undefined}
            onLoad={(e) => {
              const img = e.currentTarget;
              setActualImageSize({ width: img.naturalWidth, height: img.naturalHeight });
            }}
          />
          
          {/* Mask叠加 - 与图像完全对齐 */}
          {frame.mask_url && maskOpacity > 0 && (
            <img 
              src={`${serverUrl}${frame.mask_url}`}
              alt="Mask"
              className="absolute inset-0 w-full h-full pointer-events-none object-contain"
              style={{ opacity: maskOpacity * 0.6 }}
            />
          )}
          
          {/* Mesh投影 */}
          {showGhostWireframe && pose && intrinsics && (
            <div className="absolute inset-0 w-full h-full pointer-events-none">
              <ProjectedMeshOutline
                modelUrl={`${serverUrl}${modelUrl}`}
                pose={pose}
                intrinsics={intrinsics}
                extrinsics={frame.camera_extrinsics}
                imageWidth={imageWidth}
                imageHeight={imageHeight}
                maskUrl={frame.mask_url ? `${serverUrl}${frame.mask_url}` : undefined}
                onIoUCalculated={handleIoUCalculated}
              />
            </div>
          )}
          
          {/* 点标记 */}
          <div className="absolute inset-0 w-full h-full pointer-events-none">
            <PixelMarkerOverlay
              width={imageWidth}
              height={imageHeight}
              frameId={frame.frame_id}
              pointPairs={pointPairs}
              pendingWorldPoint={pendingWorldPoint}
            />
          </div>
          
          {/* 透明点击层 - 直接绑定到图像上方 */}
          <div 
            className={`absolute top-0 left-0 right-0 bottom-0 ${isEnabled && isActive ? 'cursor-crosshair' : 'cursor-pointer'}`}
            style={{ 
              zIndex: 100,
              touchAction: 'none'
            }}
            onClick={handleImageClick}
          />
        </div>
      </div>
      
      {/* 禁用遮罩 */}
      {!isEnabled && (
        <div className="absolute inset-0 bg-black/30 flex items-center justify-center">
          <span className="text-gray-400 text-xs">请先分类</span>
        </div>
      )}
    </div>
  );
}

export default MVFrameView;
