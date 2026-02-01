/**
 * 多视角位姿标注工具 - 主应用组件
 * 
 * 布局：
 * - 左侧：多帧图像网格视图 (2D图像 + mesh投影)
 * - 右侧上：模型空间视图 (CAD模型)
 * - 右侧下：控制面板
 */

import { useEffect, useState, useCallback } from 'react';
import { useMVAnnotationStore } from '../stores/mvAnnotationStore';
import { MVFrameView } from './MVFrameView';
import { MVModelViewer } from './MVModelViewer';
import { MVControlPanel } from './MVControlPanel';
import type { MVAnnotationInput, MVObjectItem, MVObjectData } from '../types/multiview';
import type { Matrix4 } from '../types';

// 数据服务器地址 - 使用相对路径，通过Vite代理转发
const MV_DATA_SERVER = '';

// 从服务器加载多视角物体数据
async function loadMVObjectData(sceneId: string, objectId: string, numFrames: number = 8): Promise<MVAnnotationInput | null> {
  try {
    const response = await fetch(
      `${MV_DATA_SERVER}/api/mv_object_data?scene_id=${sceneId}&object_id=${objectId}&num_frames=${numFrames}`
    );
    const data: MVObjectData = await response.json();
    
    if ((data as any).error) {
      console.error('Error loading MV data:', (data as any).error);
      return null;
    }
    
    return {
      objectId: `${data.scene_id}_${data.object_id}`,
      meshUrl: data.mesh_url,
      frames: data.frames,
      initialPose: data.world_pose as Matrix4 | null
    };
  } catch (e) {
    console.error('Failed to load MV object data:', e);
    return null;
  }
}

// 分页响应类型
interface MVObjectsResponse {
  mv_objects: MVObjectItem[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  scenes: string[];
}

// 数据选择器组件
interface MVDataSelectorProps {
  onSelect: (data: MVAnnotationInput) => void;
  savedObjectInfo?: {objectId: string; category: string} | null;
}

function MVDataSelector({ onSelect, savedObjectInfo }: MVDataSelectorProps) {
  const [mvObjects, setMVObjects] = useState<MVObjectItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingData, setLoadingData] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [numFrames, setNumFrames] = useState(8);
  
  // 分页状态
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [scenes, setScenes] = useState<string[]>([]);
  const [selectedScene, setSelectedScene] = useState<string>('');
  
  // 当有保存的对象信息时，直接更新本地列表
  useEffect(() => {
    if (savedObjectInfo) {
      console.log('[MVDataSelector] 检测到保存的对象:', savedObjectInfo);
      
      // 解析scene_id和object_id
      const parts = savedObjectInfo.objectId.split('_');
      const sceneId = parts[0];
      const objectId = parts.slice(1).join('_');
      
      // 立即更新本地列表（乐观更新）
      setMVObjects(prev => prev.map(obj => {
        if (obj.scene_id === sceneId && obj.object_id === objectId) {
          console.log('[MVDataSelector] 更新对象状态:', obj.object_id);
          return {
            ...obj,
            category: savedObjectInfo.category as 'valid' | 'fixed' | 'invalid',
            has_alignment: true
          };
        }
        return obj;
      }));
      
      // 异步增量刷新后端缓存（不阻塞UI，只更新单个对象）
      fetch(`/api/refresh_cache?scene_id=${sceneId}&object_id=${objectId}`).catch(() => {});
    }
  }, [savedObjectInfo]);
  
  // 加载多视角物体列表（带分页）
  const loadMVObjects = useCallback(async (p: number = 1, scene: string = '') => {
    setLoading(true);
    setError(null);
    try {
      let url = `${MV_DATA_SERVER}/api/mv_objects?page=${p}&page_size=${pageSize}`;
      if (scene) {
        url += `&scene=${scene}`;
      }
      const response = await fetch(url);
      const data: MVObjectsResponse = await response.json();
      setMVObjects(data.mv_objects || []);
      setTotal(data.total);
      setTotalPages(data.total_pages);
      setPage(data.page);
      if (data.scenes && data.scenes.length > 0) {
        setScenes(data.scenes);
      }
    } catch (e) {
      setError(`无法连接数据服务器 (${MV_DATA_SERVER})。请确保服务器已启动。`);
      console.error('Failed to load MV objects:', e);
    }
    setLoading(false);
  }, [pageSize]);
  
  // 初始加载
  useEffect(() => {
    loadMVObjects(1, '');
  }, [loadMVObjects]);
  
  const handleSceneChange = (scene: string) => {
    setSelectedScene(scene);
    loadMVObjects(1, scene);
  };
  
  const handlePageChange = (newPage: number) => {
    loadMVObjects(newPage, selectedScene);
  };
  
  const handleSelect = async (obj: MVObjectItem) => {
    setLoadingData(true);
    const data = await loadMVObjectData(obj.scene_id, obj.object_id, numFrames);
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
        <h1 className="text-white font-semibold">多视角位姿标注工具 - 数据选择</h1>
        <span className="ml-4 text-gray-400 text-sm">共 {total} 个物体</span>
      </header>
      
      <div className="flex-1 p-4 overflow-auto">
        {error && (
          <div className="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded mb-4">
            {error}
            <button 
              onClick={() => loadMVObjects(page, selectedScene)}
              className="ml-4 px-2 py-1 bg-red-600 rounded text-sm hover:bg-red-500"
            >
              重试
            </button>
          </div>
        )}
        
        {/* 筛选和设置 */}
        <div className="mb-4 flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-sm">场景:</span>
            <select
              value={selectedScene}
              onChange={(e) => handleSceneChange(e.target.value)}
              className="bg-gray-700 text-white px-3 py-1 rounded text-sm"
            >
              <option value="">全部场景</option>
              {scenes.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-sm">采样帧数:</span>
            <select
              value={numFrames}
              onChange={(e) => setNumFrames(parseInt(e.target.value))}
              className="bg-gray-700 text-white px-3 py-1 rounded text-sm"
            >
              <option value={4}>4帧</option>
              <option value={6}>6帧</option>
              <option value={8}>8帧</option>
              <option value={12}>12帧</option>
              <option value={16}>16帧</option>
            </select>
          </div>
        </div>
        
        {loading ? (
          <div className="text-gray-400 text-center py-8">加载中...</div>
        ) : (
          <>
            {/* 物体列表 */}
            <div className="grid gap-2 mb-4">
              {mvObjects.map((obj) => (
                <button
                  key={`${obj.scene_id}-${obj.object_id}`}
                  onClick={() => handleSelect(obj)}
                  disabled={loadingData}
                  className="flex items-center gap-4 p-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-left transition-colors disabled:opacity-50"
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-white truncate text-sm">
                      {obj.object_id}
                    </div>
                    <div className="text-gray-500 text-xs">
                      Scene: {obj.scene_id}
                    </div>
                  </div>
                  <div className={`px-2 py-1 rounded text-xs ${
                    obj.category === 'invalid' ? 'bg-red-600' :
                    obj.category === 'fixed' || obj.has_alignment ? 'bg-green-600' : 'bg-gray-600'
                  } text-white`}>
                    {obj.category === 'invalid' ? '无效数据' :
                     obj.category === 'fixed' || obj.has_alignment ? '已对齐' : '待标注'}
                  </div>
                </button>
              ))}
            </div>
            
            {/* 分页控制 */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2">
                <button
                  onClick={() => handlePageChange(1)}
                  disabled={page === 1}
                  className="px-3 py-1 bg-gray-700 text-white rounded text-sm disabled:opacity-50"
                >
                  首页
                </button>
                <button
                  onClick={() => handlePageChange(page - 1)}
                  disabled={page === 1}
                  className="px-3 py-1 bg-gray-700 text-white rounded text-sm disabled:opacity-50"
                >
                  上一页
                </button>
                <span className="text-gray-400 text-sm px-4">
                  {page} / {totalPages}
                </span>
                <button
                  onClick={() => handlePageChange(page + 1)}
                  disabled={page === totalPages}
                  className="px-3 py-1 bg-gray-700 text-white rounded text-sm disabled:opacity-50"
                >
                  下一页
                </button>
                <button
                  onClick={() => handlePageChange(totalPages)}
                  disabled={page === totalPages}
                  className="px-3 py-1 bg-gray-700 text-white rounded text-sm disabled:opacity-50"
                >
                  末页
                </button>
              </div>
            )}
          </>
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

// 主标注界面
export function MVPoseAnnotationTool() {
  const setCurrentInput = useMVAnnotationStore((state) => state.setCurrentInput);
  const currentInput = useMVAnnotationStore((state) => state.currentInput);
  const activeFrameId = useMVAnnotationStore((state) => state.activeFrameId);
  const setActiveFrameId = useMVAnnotationStore((state) => state.setActiveFrameId);
  const calculatedPose = useMVAnnotationStore((state) => state.calculatedPose);
  const reset = useMVAnnotationStore((state) => state.reset);
  const category = useMVAnnotationStore((state) => state.category);
  
  // 用于通知数据选择器更新对象状态
  const [savedObjectInfo, setSavedObjectInfo] = useState<{objectId: string; category: string} | null>(null);
  
  const handleDataSelect = useCallback((data: MVAnnotationInput) => {
    setCurrentInput(data);
  }, [setCurrentInput]);
  
  const handleBack = useCallback(() => {
    // 如果有保存的分类信息，传递给数据选择器
    // 注意：保存时会自动将非invalid的category设置为fixed
    if (category && currentInput) {
      const finalCategory = category === 'invalid' ? 'invalid' : 'fixed';
      setSavedObjectInfo({
        objectId: currentInput.objectId,
        category: finalCategory
      });
    }
    reset();
  }, [reset, category, currentInput]);
  
  // 使用计算后的位姿或初始位姿
  const currentPose = calculatedPose ?? currentInput?.initialPose;
  
  // 没有数据时显示数据选择器
  if (!currentInput) {
    return <MVDataSelector onSelect={handleDataSelect} savedObjectInfo={savedObjectInfo} />;
  }
  
  return (
    <div className="w-full h-screen bg-gray-900 flex flex-col">
      {/* 顶部标题栏 */}
      <header className="h-12 bg-gray-800 border-b border-gray-700 flex items-center px-4 shrink-0">
        <button
          onClick={handleBack}
          className="mr-4 px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-white text-sm"
        >
          ← 返回
        </button>
        <h1 className="text-white font-semibold">多视角位姿标注</h1>
        <div className="ml-4 text-sm text-gray-400 truncate max-w-md">
          {currentInput.objectId}
        </div>
        <div className="ml-auto text-sm text-gray-400">
          {currentInput.frames.length} 帧
        </div>
      </header>
      
      {/* 主内容区 */}
      <div className="flex-1 flex min-h-0">
        {/* 左侧区域 - 可滚动 */}
        <div className="flex-1 overflow-auto p-2">
          {/* 选中帧放大显示 */}
          {activeFrameId && (
            <div className="mb-4">
              <div className="bg-gray-800 rounded-lg overflow-hidden">
                <MVFrameView
                  key={`active-${activeFrameId}`}
                  frame={currentInput.frames.find(f => f.frame_id === activeFrameId)!}
                  modelUrl={currentInput.meshUrl}
                  pose={currentPose ?? null}
                  isActive={true}
                  onSelect={() => {}}
                  serverUrl={MV_DATA_SERVER}
                  enlarged={true}
                />
              </div>
            </div>
          )}
          
          {/* 多帧缩略图网格 */}
          <div className="grid grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-1">
            {currentInput.frames.map((frame) => (
              <MVFrameView
                key={frame.frame_id}
                frame={frame}
                modelUrl={currentInput.meshUrl}
                pose={currentPose ?? null}
                isActive={frame.frame_id === activeFrameId}
                onSelect={() => setActiveFrameId(frame.frame_id)}
                serverUrl={MV_DATA_SERVER}
                enlarged={false}
              />
            ))}
          </div>
        </div>
        
        {/* 右侧面板 */}
        <div className="w-80 flex flex-col p-2 gap-2 shrink-0">
          {/* 模型视图 */}
          <div className="h-1/2">
            <MVModelViewer modelUrl={`${MV_DATA_SERVER}${currentInput.meshUrl}`} />
          </div>
          
          {/* 控制面板 */}
          <div className="h-1/2">
            <MVControlPanel />
          </div>
        </div>
      </div>
    </div>
  );
}

export default MVPoseAnnotationTool;
