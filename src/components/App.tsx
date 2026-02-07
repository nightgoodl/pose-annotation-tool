/**
 * 主应用组件 - 6D Pose Annotation Tool
 * 
 * 布局：
 * - 左侧：世界空间视图 (2D图像 + 3D幽灵线框)
 * - 右侧：模型空间视图 (CAD模型)
 * - 右侧边栏：控制面板
 * - 弹出层：分类面板
 */

import { useEffect, useState, useCallback } from 'react';
import { useAnnotationStore } from '../stores/annotationStore';
import { WorldView } from './WorldView';
import { ModelViewer } from './ModelViewer';
import { ClassificationPanel } from './ClassificationPanel';
import { ControlPanel } from './ControlPanel';
import { showGlobalToast } from '../hooks/useToast';
import type { AnnotationInput, Matrix4 } from '../types';

// 数据服务器地址 - 使用base路径以支持tunnel
// import.meta.env.BASE_URL includes trailing slash, so we need to remove it
const BASE_PATH = ((import.meta as any).env?.BASE_URL || '/').replace(/\/$/, '');
const DATA_SERVER = BASE_PATH;
// 解码服务通过数据服务器代理访问

// 已对齐物体的类型
interface AlignedObject {
  scene_id: string;
  object_id: string;
  frame_id: string;
  iou: number | null;
}

// 下一个任务的类型
interface NextTaskResponse {
  success: boolean;
  has_next: boolean;
  data: {
    scene_id: string;
    object_id: string;
    frame_id: string;
    iou: number | null;
  } | null;
  remaining_count: number;
  message?: string;
}

// 获取下一个待标注任务
async function fetchNextTask(currentSceneId?: string, excludeObjectId?: string, excludeFrameId?: string): Promise<NextTaskResponse> {
  const params = new URLSearchParams();
  if (currentSceneId) params.set('current_scene_id', currentSceneId);
  if (excludeObjectId) params.set('exclude_object_id', excludeObjectId);
  if (excludeFrameId) params.set('exclude_frame_id', excludeFrameId);
  
  const response = await fetch(`${DATA_SERVER}/api/next_task?${params.toString()}`);
  return await response.json();
}

// 从服务器加载数据
async function loadObjectData(sceneId: string, objectId: string, frameId: string): Promise<AnnotationInput | null> {
  try {
    const response = await fetch(
      `${DATA_SERVER}/api/object_data?scene_id=${sceneId}&object_id=${objectId}&frame_id=${frameId}`
    );
    const data = await response.json();
    
    if (data.error) {
      console.error('Error loading data:', data.error);
      return null;
    }
    
    // 转换相机内参格式
    const K = data.camera_intrinsics.K;
    const cameraIntrinsics = {
      fx: K[0][0],
      fy: K[1][1],
      cx: K[0][2],
      cy: K[1][2],
      width: data.camera_intrinsics.width,
      height: data.camera_intrinsics.height
    };
    
    // camera_extrinsics 是 camera-to-world (RT)
    const cameraExtrinsics = data.camera_extrinsics as Matrix4;
    
    // world_pose 是 Model-to-World
    const initialCoarsePose = data.world_pose as Matrix4 || [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ];
    
    // 加载深度图
    let depthMap = new Float32Array(512 * 384).fill(1.5);
    let depthWidth = 512;
    let depthHeight = 384;
    
    if (data.depth_url) {
      try {
        const depthResponse = await fetch(`${DATA_SERVER}${data.depth_url}`);
        const depthData = await depthResponse.json();
        depthMap = new Float32Array(depthData.data);
        depthWidth = depthData.width;
        depthHeight = depthData.height;
      } catch (e) {
        console.warn('Failed to load depth:', e);
      }
    }
    
    // 尝试从解码服务获取带颜色的模型
    let cadModelUrl = data.mesh_url ? `${DATA_SERVER}${data.mesh_url}` : '';
    
    console.log('[loadObjectData] 初始状态:', {
      mesh_url: data.mesh_url,
      latent_path: data.latent_path ? '存在' : '不存在',
      cadModelUrl
    });
    
    // 必须从解码服务获取带颜色的模型（通过数据服务器代理）
    if (data.latent_path) {
      try {
        console.log('[loadObjectData] 从解码服务获取带颜色的模型...');
        // 通过数据服务器代理访问解码服务
        const decodeUrl = `${DATA_SERVER}/api/decode?scene_id=${sceneId}&object_id=${objectId}`;
        console.log('[loadObjectData] 请求URL:', decodeUrl);
        
        const decodeResponse = await fetch(decodeUrl);
        console.log('[loadObjectData] 响应状态:', decodeResponse.status, decodeResponse.statusText);
        
        if (!decodeResponse.ok) {
          throw new Error(`HTTP ${decodeResponse.status}: ${decodeResponse.statusText}`);
        }
        
        const decodeData = await decodeResponse.json();
        console.log('[loadObjectData] 解码服务响应:', decodeData);
        
        if (decodeData.success && decodeData.mesh_url) {
          // mesh_url是/cache/xxx.glb，通过数据服务器代理访问
          cadModelUrl = `${DATA_SERVER}${decodeData.mesh_url}`;
          console.log('[loadObjectData] ✓ 使用解码后的带颜色模型:', cadModelUrl);
        } else {
          console.warn('[loadObjectData] 解码失败:', decodeData);
        }
      } catch (e) {
        console.error('[loadObjectData] 解码服务错误:', e);
        if (e instanceof Error) {
          console.error('[loadObjectData] 错误详情:', {
            name: e.name,
            message: e.message,
            stack: e.stack
          });
        }
      }
    } else {
      console.log('[loadObjectData] 无latent_path，无法加载模型');
    }
    
    console.log('[loadObjectData] 最终cadModelUrl:', cadModelUrl);
    
    return {
      objectId: `${sceneId}_${objectId}_${frameId}`,
      rgbImage: data.rgb_url ? `${DATA_SERVER}${data.rgb_url}` : '',
      depthMap,
      depthWidth,
      depthHeight,
      maskImage: data.mask_url ? `${DATA_SERVER}${data.mask_url}` : '',
      cadModel: cadModelUrl,
      cameraIntrinsics,
      cameraExtrinsics,
      initialCoarsePose
    };
  } catch (e) {
    console.error('Failed to load object data:', e);
    return null;
  }
}

// 数据选择器组件
interface DataSelectorProps {
  onSelect: (data: AnnotationInput) => void;
}

function DataSelector({ onSelect }: DataSelectorProps) {
  const [alignedObjects, setAlignedObjects] = useState<AlignedObject[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingData, setLoadingData] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // 加载已对齐物体列表
  const loadAlignedObjects = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${DATA_SERVER}/api/aligned_objects`);
      const data = await response.json();
      setAlignedObjects(data.aligned_objects || []);
    } catch (e) {
      setError(`无法连接数据服务器 (${DATA_SERVER})。请确保服务器已启动。`);
      console.error('Failed to load aligned objects:', e);
    }
    setLoading(false);
  }, []);
  
  useEffect(() => {
    loadAlignedObjects();
  }, [loadAlignedObjects]);
  
  const handleSelect = async (obj: AlignedObject) => {
    setLoadingData(true);
    const data = await loadObjectData(obj.scene_id, obj.object_id, obj.frame_id);
    setLoadingData(false);
    if (data) {
      onSelect(data);
    } else {
      setError('加载数据失败');
    }
  };
  
  return (
    <div className="w-full h-screen bg-gray-900 flex flex-col">
      <header className="h-12 bg-gray-800 border-b border-gray-700 flex items-center px-4">
        <h1 className="text-white font-semibold">6D Pose Annotation Tool - 数据选择</h1>
      </header>
      
      <div className="flex-1 p-4 overflow-auto">
        {error && (
          <div className="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded mb-4">
            {error}
            <button 
              onClick={loadAlignedObjects}
              className="ml-4 px-2 py-1 bg-red-600 rounded text-sm hover:bg-red-500"
            >
              重试
            </button>
          </div>
        )}
        
        {loading ? (
          <div className="text-gray-400 text-center py-8">加载中...</div>
        ) : (
          <div className="space-y-2">
            <div className="text-gray-400 mb-4">
              找到 {alignedObjects.length} 个已对齐物体 (按IOU排序)
            </div>
            
            <div className="grid gap-2">
              {alignedObjects.map((obj, index) => (
                <button
                  key={`${obj.scene_id}-${obj.object_id}-${obj.frame_id}`}
                  onClick={() => handleSelect(obj)}
                  disabled={loadingData}
                  className="flex items-center gap-4 p-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-left transition-colors disabled:opacity-50"
                >
                  <span className="text-gray-500 w-8">{index + 1}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-white truncate">
                      {obj.object_id}
                    </div>
                    <div className="text-gray-500 text-sm">
                      Scene: {obj.scene_id} | Frame: {obj.frame_id}
                    </div>
                  </div>
                  <div className={`px-2 py-1 rounded text-sm ${
                    (obj.iou || 0) > 0.7 ? 'bg-green-600' :
                    (obj.iou || 0) > 0.5 ? 'bg-yellow-600' : 'bg-red-600'
                  } text-white`}>
                    IOU: {obj.iou?.toFixed(3) || 'N/A'}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
        
        {loadingData && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center">
            <div className="bg-gray-800 px-6 py-4 rounded-lg text-white">
              加载数据中...
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface PoseAnnotationToolProps {
  input?: AnnotationInput;
  onSave?: (output: any) => void;
  onSkip?: (reason: string) => void;
}

export function PoseAnnotationTool({ input, onSave, onSkip }: PoseAnnotationToolProps) {
  const setCurrentInput = useAnnotationStore((state) => state.setCurrentInput);
  const currentInput = useAnnotationStore((state) => state.currentInput);
  const workflowState = useAnnotationStore((state) => state.workflowState);
  const category = useAnnotationStore((state) => state.category);
  const isSavingNext = useAnnotationStore((state) => state.isSavingNext);
  const reset = useAnnotationStore((state) => state.reset);
  
  // 初始化输入数据
  useEffect(() => {
    if (input) {
      setCurrentInput(input);
    }
  }, [input, setCurrentInput]);
  
  // 处理分类后的回调
  useEffect(() => {
    if (category === 'fixed' && onSave) {
      const result = useAnnotationStore.getState().exportAnnotation();
      if (result) {
        onSave(result);
      }
    } else if (category === 'invalid' && onSkip) {
      onSkip('invalid');
    }
  }, [category, onSave, onSkip]);
  
  const handleDataSelect = useCallback((data: AnnotationInput) => {
    setCurrentInput(data);
  }, [setCurrentInput]);
  
  const handleBack = useCallback(() => {
    reset();
  }, [reset]);
  
  // 保存并处理下一个
  const handleSaveAndNext = useCallback(async () => {
    const store = useAnnotationStore.getState();
    if (store.isSavingNext || !store.currentInput) return;
    
    store.setIsSavingNext(true);
    
    try {
      // 步骤1：保存当前标注
      const saveResult = await store.savePose();
      if (!saveResult.success) {
        showGlobalToast(`保存失败: ${saveResult.error}`, 'error', 4000);
        store.setIsSavingNext(false);
        return;
      }
      showGlobalToast('已保存当前标注', 'success', 2000);
      
      // 解析当前objectId获取scene_id, object_id, frame_id
      const parts = store.currentInput!.objectId.split('_');
      const currentSceneId = parts[0];
      const currentFrameId = parts[parts.length - 1];
      const currentObjectId = parts.slice(1, -1).join('_');
      
      // 步骤2：获取下一个待标注任务
      const nextTaskResponse = await fetchNextTask(currentSceneId, currentObjectId, currentFrameId);
      
      if (!nextTaskResponse.success || !nextTaskResponse.has_next || !nextTaskResponse.data) {
        showGlobalToast('所有物体已标注完成！', 'info', 5000);
        store.setIsSavingNext(false);
        reset();
        return;
      }
      
      store.setRemainingCount(nextTaskResponse.remaining_count - 1);
      
      // 步骤3：加载下一个任务的数据
      const nextData = nextTaskResponse.data;
      const objectData = await loadObjectData(
        nextData.scene_id,
        nextData.object_id,
        nextData.frame_id
      );
      
      if (!objectData) {
        showGlobalToast('加载下一个物体数据失败', 'error', 4000);
        store.setIsSavingNext(false);
        return;
      }
      
      // 步骤4：切换到新的标注对象
      setCurrentInput(objectData);
      showGlobalToast(
        `已切换到下一个物体 (剩余 ${nextTaskResponse.remaining_count - 1} 个)`,
        'success',
        3000
      );
      
    } catch (error) {
      console.error('[handleSaveAndNext] error:', error);
      showGlobalToast(`操作失败: ${error}`, 'error', 4000);
    } finally {
      useAnnotationStore.getState().setIsSavingNext(false);
    }
  }, [reset, setCurrentInput]);
  
  // 快捷键支持
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+S / Cmd+S: 保存
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        const store = useAnnotationStore.getState();
        if (store.workflowState === 'annotation') {
          store.savePose().then(result => {
            if (result.success) {
              showGlobalToast(`Pose已保存到: ${result.pose_path}`, 'success', 3000);
            } else {
              showGlobalToast(`保存失败: ${result.error}`, 'error', 4000);
            }
          });
        }
      }
      
      // Ctrl+Enter / Cmd+Enter: 保存并处理下一个
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        handleSaveAndNext();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleSaveAndNext]);
  
  // 没有数据时显示数据选择器
  if (!currentInput) {
    return <DataSelector onSelect={handleDataSelect} />;
  }
  
  return (
    <div className="w-full h-screen bg-gray-900 flex flex-col">
      {/* 顶部标题栏 */}
      <header className="h-12 bg-gray-800 border-b border-gray-700 flex items-center px-4">
        <button
          onClick={handleBack}
          className="mr-4 px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-white text-sm"
        >
          ← 返回
        </button>
        <h1 className="text-white font-semibold">6D Pose Annotation Tool</h1>
        <div className="ml-4 text-sm text-gray-400 truncate max-w-md">
          {currentInput.objectId}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className={`px-2 py-1 rounded text-xs ${
            workflowState === 'classification' ? 'bg-yellow-600' :
            workflowState === 'annotation' ? 'bg-green-600' :
            'bg-blue-600'
          } text-white`}>
            {workflowState === 'classification' ? '分类中' :
             workflowState === 'annotation' ? '标注中' : '审核中'}
          </span>
        </div>
      </header>
      
      {/* 主内容区 */}
      <div className="flex-1 flex min-h-0">
        {/* 左侧：世界空间视图 */}
        <div className="flex-1 p-2">
          <WorldView
            imageUrl={currentInput.rgbImage}
            maskUrl={currentInput.maskImage}
            modelUrl={currentInput.cadModel}
            depthMap={currentInput.depthMap}
            depthWidth={currentInput.depthWidth}
            depthHeight={currentInput.depthHeight}
            cameraIntrinsics={currentInput.cameraIntrinsics}
            cameraExtrinsics={currentInput.cameraExtrinsics}
            initialPose={currentInput.initialCoarsePose}
          />
        </div>
        
        {/* 右侧：模型空间视图 */}
        <div className="flex-1 p-2">
          <ModelViewer modelUrl={currentInput.cadModel} />
        </div>
        
        {/* 右侧边栏：控制面板 */}
        <div className="w-72 p-2">
          <ControlPanel onSaveAndNext={handleSaveAndNext} />
        </div>
      </div>
      
      {/* 分类弹出层 */}
      <ClassificationPanel />
      
      {/* 保存并处理下一个的加载遮罩 */}
      {isSavingNext && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-[9998]">
          <div className="bg-gray-800 rounded-xl px-8 py-6 text-center shadow-2xl border border-gray-700">
            <div className="w-12 h-12 border-4 border-gray-600 border-t-blue-500 rounded-full animate-spin mx-auto mb-4" />
            <div className="space-y-2">
              <div className="text-green-400 text-sm">✓ 已保存当前标注</div>
              <div className="text-white text-sm font-semibold">正在加载下一个物体...</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default PoseAnnotationTool;
