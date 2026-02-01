/**
 * 多视角标注 - 模型空间 CAD 模型查看器
 * 
 * 使用 mvAnnotationStore 代替 annotationStore
 */

import { useRef, useCallback, Suspense, useMemo } from 'react';
import { Canvas, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, useGLTF, Center, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { useMVAnnotationStore } from '../stores/mvAnnotationStore';
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
    
    clone.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        const material = child.material as THREE.MeshStandardMaterial;
        if (material.map || (child.geometry.attributes.color)) {
          hasTexture = true;
        }
      }
    });
    
    clone.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        const originalMaterial = child.material as THREE.MeshStandardMaterial;
        
        if (hasTexture) {
          if (originalMaterial.clone) {
            const newMaterial = originalMaterial.clone();
            newMaterial.side = THREE.DoubleSide;
            child.material = newMaterial;
          }
        } else {
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
    
    if (e.point && e.face) {
      const normal = e.face.normal;
      const cameraDir = e.ray.direction;
      const dotProduct = normal.dot(cameraDir);
      
      if (dotProduct > 0) {
        console.log('Clicked on back face, ignoring');
        return;
      }
      
      const MESH_SCALE = 2.0;
      const localPoint: Point3D = {
        x: e.point.x * MESH_SCALE,
        y: e.point.y * MESH_SCALE,
        z: e.point.z * MESH_SCALE
      };
      console.log('MV Surface point selected (scaled):', localPoint);
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

function MVPointMarkers() {
  const pointPairs = useMVAnnotationStore((state) => state.pointPairs);
  const pendingLocalPoint = useMVAnnotationStore((state) => state.pendingLocalPoint);
  const selectedPairId = useMVAnnotationStore((state) => state.selectedPairId);
  
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

interface MVModelViewerProps {
  modelUrl?: string;
}

export function MVModelViewer({ modelUrl }: MVModelViewerProps) {
  const isEnabled = useMVAnnotationStore((state) => state.isAnnotationEnabled);
  const setPendingLocalPoint = useMVAnnotationStore((state) => state.setPendingLocalPoint);
  const pendingWorldPoint = useMVAnnotationStore((state) => state.pendingWorldPoint);
  
  const handlePointClick = useCallback((point: Point3D) => {
    console.log('MV Model point clicked:', point);
    setPendingLocalPoint(point);
  }, [setPendingLocalPoint]);
  
  return (
    <div className="relative w-full h-full bg-gray-900 rounded-lg overflow-hidden">
      {/* 标题栏 */}
      <div className="absolute top-0 left-0 right-0 z-10 bg-gray-800/80 backdrop-blur px-3 py-1 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-white font-medium text-sm">模型空间</span>
        </div>
        <div className={`px-2 py-0.5 rounded text-xs ${isEnabled ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'}`}>
          {isEnabled ? '可交互' : '已禁用'}
        </div>
      </div>
      
      {/* 提示信息 */}
      {isEnabled && pendingWorldPoint && (
        <div className="absolute top-8 left-2 right-2 z-10 bg-blue-600/90 text-white px-2 py-1 rounded text-xs text-center">
          点击模型标注对应点
        </div>
      )}
      
      {/* 禁用遮罩 */}
      {!isEnabled && (
        <div className="absolute inset-0 z-20 bg-black/50 flex items-center justify-center">
          <div className="text-gray-400 text-center text-sm">
            <p>请先分类</p>
          </div>
        </div>
      )}
      
      {/* 无模型提示 */}
      {!modelUrl && (
        <div className="absolute inset-0 z-10 flex items-center justify-center">
          <div className="text-gray-400 text-center text-sm">
            <p>无可用模型</p>
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
        <MVPointMarkers />
        <OrbitControls 
          enabled={isEnabled}
          makeDefault
          minDistance={0.5}
          maxDistance={10}
        />
      </Canvas>
      
      {/* 坐标轴提示 */}
      <div className="absolute bottom-2 left-2 text-xs text-gray-500">
        点击模型表面标注特征点
      </div>
    </div>
  );
}

export default MVModelViewer;
