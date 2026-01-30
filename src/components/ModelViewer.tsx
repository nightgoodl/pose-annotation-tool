/**
 * 右侧视图 - 模型空间 CAD 模型查看器
 * 
 * 功能：
 * - 渲染纯净的 CAD 模型（位于原点，无旋转，缩放为1）
 * - 用户点击模型表面获取局部坐标 P_local
 * - 禁用状态时不允许交互
 */

import React, { useRef, useCallback, Suspense, useMemo } from 'react';
import { Canvas, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, useGLTF, Center, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { useAnnotationStore } from '../stores/annotationStore';
import type { Point3D } from '../types';

interface ModelMeshProps {
  url: string;
  onPointClick: (point: Point3D) => void;
  enabled: boolean;
}

function ModelMesh({ url, onPointClick, enabled }: ModelMeshProps) {
  const { scene } = useGLTF(url);
  const meshRef = useRef<THREE.Group>(null);
  
  const clonedScene = useMemo(() => {
    const clone = scene.clone(true);
    let hasTexture = false;
    
    // 检查模型是否有纹理或顶点颜色
    clone.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        const material = child.material as THREE.MeshStandardMaterial;
        if (material.map || (child.geometry.attributes.color)) {
          hasTexture = true;
        }
      }
    });
    
    // 如果有纹理/顶点颜色，保留原始材质；否则使用默认蓝色
    clone.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        const originalMaterial = child.material as THREE.MeshStandardMaterial;
        
        if (hasTexture) {
          // 保留原始材质，但确保双面渲染
          if (originalMaterial.clone) {
            const newMaterial = originalMaterial.clone();
            newMaterial.side = THREE.DoubleSide;
            child.material = newMaterial;
          }
        } else {
          // 无纹理时使用默认蓝色材质
          child.material = new THREE.MeshStandardMaterial({
            color: 0x4488ff,
            metalness: 0.2,
            roughness: 0.5,
            side: THREE.DoubleSide,
            emissive: 0x112244,
            emissiveIntensity: 0.3
          });
        }
      }
    });
    return clone;
  }, [scene]);
  
  const handleClick = useCallback((e: ThreeEvent<MouseEvent>) => {
    if (!enabled) return;
    e.stopPropagation();
    
    // e.point 是射线与模型表面的第一个交点（最近的表面点）
    // e.face 包含法线信息，可以确认是否点击在外表面
    if (e.point && e.face) {
      // 检查法线方向，确保点击在面向相机的表面
      const normal = e.face.normal;
      const cameraDir = e.ray.direction;
      const dotProduct = normal.dot(cameraDir);
      
      // 如果法线与射线方向的点积为正，说明是背面
      if (dotProduct > 0) {
        console.log('Clicked on back face, ignoring');
        return;
      }
      
      // decoder mesh范围是(-0.5, 0.5)，而ultrashape mesh范围是(-1, 1)
      // 需要将点击坐标缩放2倍以匹配原始world_pose的尺度
      const MESH_SCALE = 2.0;
      const localPoint: Point3D = {
        x: e.point.x * MESH_SCALE,
        y: e.point.y * MESH_SCALE,
        z: e.point.z * MESH_SCALE
      };
      console.log('Surface point selected (scaled):', localPoint, 'normal:', { x: normal.x, y: normal.y, z: normal.z });
      onPointClick(localPoint);
    }
  }, [enabled, onPointClick]);
  
  return (
    <Center>
      <group ref={meshRef}>
        <primitive 
          object={clonedScene} 
          onClick={handleClick}
          onPointerOver={(e: ThreeEvent<PointerEvent>) => {
            if (enabled) {
              e.stopPropagation();
              document.body.style.cursor = 'crosshair';
            }
          }}
          onPointerOut={() => {
            document.body.style.cursor = 'default';
          }}
        />
      </group>
    </Center>
  );
}

interface PointMarkerProps {
  point: Point3D;
  color: string;
  size?: number;
}

function PointMarker({ point, color, size = 0.02 }: PointMarkerProps) {
  return (
    <mesh position={[point.x, point.y, point.z]}>
      <sphereGeometry args={[size, 16, 16]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

function PointMarkers() {
  const pointPairs = useAnnotationStore((state) => state.pointPairs);
  const pendingLocalPoint = useAnnotationStore((state) => state.pendingLocalPoint);
  const selectedPairId = useAnnotationStore((state) => state.selectedPairId);
  
  // localPoint存储的是缩放后的坐标(-1,1)，但mesh显示的是(-0.5,0.5)
  // 需要将点位置除以2才能正确显示在mesh上
  const MESH_SCALE = 2.0;
  const scalePoint = (p: { x: number; y: number; z: number }) => ({
    x: p.x / MESH_SCALE,
    y: p.y / MESH_SCALE,
    z: p.z / MESH_SCALE
  });
  
  return (
    <group>
      {pointPairs.map((pair) => (
        <PointMarker
          key={pair.id}
          point={scalePoint(pair.localPoint)}
          color={pair.id === selectedPairId ? '#ffff00' : '#00ff00'}
          size={pair.id === selectedPairId ? 0.025 : 0.02}
        />
      ))}
      {pendingLocalPoint && (
        <PointMarker
          point={scalePoint(pendingLocalPoint)}
          color="#ff6600"
          size={0.025}
        />
      )}
    </group>
  );
}

function SceneSetup() {
  return (
    <>
      <ambientLight intensity={0.8} />
      <directionalLight position={[5, 5, 5]} intensity={1.5} />
      <directionalLight position={[-5, 5, -5]} intensity={0.8} />
      <directionalLight position={[0, -5, 0]} intensity={0.4} />
      <Grid 
        infiniteGrid 
        cellSize={0.1} 
        sectionSize={1} 
        fadeDistance={10}
        cellColor="#555555"
        sectionColor="#777777"
      />
    </>
  );
}

interface ModelViewerProps {
  modelUrl?: string;
}

export function ModelViewer({ modelUrl }: ModelViewerProps) {
  const isEnabled = useAnnotationStore((state) => state.isRightViewEnabled);
  const setPendingLocalPoint = useAnnotationStore((state) => state.setPendingLocalPoint);
  const pendingWorldPoint = useAnnotationStore((state) => state.pendingWorldPoint);
  
  const handlePointClick = useCallback((point: Point3D) => {
    console.log('Model point clicked:', point);
    setPendingLocalPoint(point);
  }, [setPendingLocalPoint]);
  
  return (
    <div className="relative w-full h-full bg-gray-900 rounded-lg overflow-hidden">
      {/* 标题栏 */}
      <div className="absolute top-0 left-0 right-0 z-10 bg-gray-800/80 backdrop-blur px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-white font-medium">模型空间</span>
          <span className="text-gray-400 text-sm">(Model Space)</span>
        </div>
        <div className={`px-2 py-1 rounded text-xs ${isEnabled ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'}`}>
          {isEnabled ? '可交互' : '已禁用'}
        </div>
      </div>
      
      {/* 提示信息 */}
      {isEnabled && pendingWorldPoint && (
        <div className="absolute top-12 left-4 right-4 z-10 bg-blue-600/80 backdrop-blur text-white px-3 py-2 rounded text-sm">
          请在模型上点击对应的特征点
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
      
      {/* 无模型提示 */}
      {!modelUrl && (
        <div className="absolute inset-0 z-10 flex items-center justify-center">
          <div className="text-gray-400 text-center">
            <p className="text-lg mb-2">无可用模型</p>
            <p className="text-sm">模型URL: {modelUrl || '(空)'}</p>
          </div>
        </div>
      )}
      
      {/* 3D 画布 */}
      <Canvas
        camera={{ position: [2, 2, 2], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
      >
        <SceneSetup />
        <Suspense fallback={null}>
          {modelUrl && (
            <ModelMesh
              url={modelUrl}
              onPointClick={handlePointClick}
              enabled={isEnabled}
            />
          )}
        </Suspense>
        <PointMarkers />
        <OrbitControls 
          enabled={isEnabled}
          makeDefault
          minDistance={0.5}
          maxDistance={10}
        />
      </Canvas>
      
      {/* 坐标轴提示 */}
      <div className="absolute bottom-4 left-4 text-xs text-gray-400">
        <div>X: 红 | Y: 绿 | Z: 蓝</div>
        <div>模型位于原点，无变换</div>
      </div>
    </div>
  );
}

export default ModelViewer;
