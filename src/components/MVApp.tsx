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
import { ToastContainer } from './ToastContainer';
import { showGlobalToast } from '../hooks/useToast';
import type { MVAnnotationInput, MVObjectItem, MVObjectData, BboxInfo, MeshInfo } from '../types/multiview';
import type { Matrix4 } from '../types';

// 数据服务器地址 - 使用base路径以支持tunnel
// import.meta.env.BASE_URL includes trailing slash, so we need to remove it
const BASE_PATH = ((import.meta as any).env?.BASE_URL || '/').replace(/\/$/, '');
const MV_DATA_SERVER = BASE_PATH;

// 下一个MV任务的响应类型
interface NextMVTaskResponse {
  success: boolean;
  has_next: boolean;
  data: {
    scene_id: string;
    object_id: string;
  } | null;
  remaining_count: number;
  message?: string;
}

// 获取下一个待标注MV任务
async function fetchNextMVTask(currentSceneId?: string, excludeObjectId?: string): Promise<NextMVTaskResponse> {
  const params = new URLSearchParams();
  if (currentSceneId) params.set('current_scene_id', currentSceneId);
  if (excludeObjectId) params.set('exclude_object_id', excludeObjectId);
  
  const url = `${MV_DATA_SERVER}/api/next_mv_task?${params.toString()}`;
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error(`获取下一个任务失败: HTTP ${response.status} (请确认后端服务已重启)`);
  }
  
  const contentType = response.headers.get('content-type') || '';
  if (!contentType.includes('application/json')) {
    throw new Error(`获取下一个任务失败: 返回非JSON响应 (请确认后端服务已重启)`);
  }
  
  return await response.json();
}

// 从服务器加载多视角物体数据
async function loadMVObjectData(sceneId: string, objectId: string, numFrames: number = 4): Promise<MVAnnotationInput | null> {
  try {
    const response = await fetch(
      `${MV_DATA_SERVER}/api/mv_object_data?scene_id=${sceneId}&object_id=${objectId}&num_frames=${numFrames}`
    );
    const data: MVObjectData = await response.json();
    
    if ((data as any).error) {
      console.error('Error loading MV data:', (data as any).error);
      return null;
    }
    
    console.log('[loadMVObjectData] gt_bbox:', data.gt_bbox);
    console.log('[loadMVObjectData] mesh_info:', data.mesh_info);
    
    return {
      objectId: `${data.scene_id}_${data.object_id}`,
      meshUrl: data.mesh_url,
      meshPath: data.mesh_path,
      frames: data.frames,
      initialPose: data.world_pose as Matrix4 | null,
      gtBbox: data.gt_bbox as BboxInfo | null,
      meshInfo: data.mesh_info as MeshInfo | null
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
  stats?: {
    all: number;
    pending: number;
    aligned: number;
    invalid: number;
  };
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
  const [numFrames, setNumFrames] = useState(4);
  
  // 分页状态
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [scenes, setScenes] = useState<string[]>([]);
  const [selectedScene, setSelectedScene] = useState<string>('');
  
  // 排序状态
  const [sortBy, setSortBy] = useState<string | null>(null);
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  
  // 状态过滤
  const [statusFilter, setStatusFilter] = useState<string>('');
  
  // 统计
  const [stats, setStats] = useState<{all: number; pending: number; aligned: number; invalid: number}>({all: 0, pending: 0, aligned: 0, invalid: 0});
  
  // 当有保存的对象信息时，直接更新本地列表
  useEffect(() => {
    if (savedObjectInfo) {
      const parts = savedObjectInfo.objectId.split('_');
      const sceneId = parts[0];
      const objectId = parts.slice(1).join('_');
      
      setMVObjects(prev => prev.map(obj => {
        if (obj.scene_id === sceneId && obj.object_id === objectId) {
          return { ...obj, category: savedObjectInfo.category as 'valid' | 'fixed' | 'invalid', has_alignment: true };
        }
        return obj;
      }));
      
      fetch(`${BASE_PATH}/api/refresh_cache?scene_id=${sceneId}&object_id=${objectId}`).catch(() => {});
    }
  }, [savedObjectInfo]);
  
  // 加载多视角物体列表
  const loadMVObjects = useCallback(async (p: number = 1, scene: string = '', sort?: string | null, order?: string, status?: string) => {
    setLoading(true);
    setError(null);
    try {
      let url = `${MV_DATA_SERVER}/api/mv_objects?page=${p}&page_size=${pageSize}`;
      if (scene) url += `&scene=${scene}`;
      if (sort) url += `&sort_by=${sort}&sort_order=${order || 'desc'}`;
      if (status) url += `&status=${status}`;
      
      const response = await fetch(url);
      const data: MVObjectsResponse = await response.json();
      setMVObjects(data.mv_objects || []);
      setTotal(data.total);
      setTotalPages(data.total_pages);
      setPage(data.page);
      if (data.scenes && data.scenes.length > 0) setScenes(data.scenes);
      if (data.stats) setStats(data.stats);
    } catch (e) {
      setError(`无法连接数据服务器 (${MV_DATA_SERVER})。请确保服务器已启动。`);
    }
    setLoading(false);
  }, [pageSize]);
  
  // 初始加载
  useEffect(() => { loadMVObjects(1, ''); }, [loadMVObjects]);
  
  const reload = (p?: number, scene?: string, sort?: string | null, order?: 'asc' | 'desc', status?: string) => {
    loadMVObjects(p ?? page, scene ?? selectedScene, sort ?? sortBy, order ?? sortOrder, status ?? statusFilter);
  };
  
  const handleSceneChange = (scene: string) => { setSelectedScene(scene); loadMVObjects(1, scene, sortBy, sortOrder, statusFilter); };
  const handlePageChange = (p: number) => { loadMVObjects(p, selectedScene, sortBy, sortOrder, statusFilter); };
  const handleStatusFilter = (s: string) => { setStatusFilter(s); loadMVObjects(1, selectedScene, sortBy, sortOrder, s); };
  
  const handleSort = (key: string) => {
    const newOrder = sortBy === key && sortOrder === 'desc' ? 'asc' : 'desc';
    setSortBy(key);
    setSortOrder(newOrder);
    loadMVObjects(1, selectedScene, key, newOrder, statusFilter);
  };
  
  const handleSelect = async (obj: MVObjectItem) => {
    setLoadingData(true);
    const data = await loadMVObjectData(obj.scene_id, obj.object_id, numFrames);
    setLoadingData(false);
    if (data) { onSelect(data); } else { setError('加载数据失败'); }
  };
  
  const getStatusInfo = (obj: MVObjectItem) => {
    if (obj.category === 'invalid') return { text: '无效', cls: 'bg-red-600' };
    if (obj.category === 'fixed' || obj.has_alignment) return { text: '已对齐', cls: 'bg-green-600' };
    return { text: '待标注', cls: 'bg-gray-600' };
  };
  
  const getIoUColor = (iou: number) => {
    if (iou >= 0.5) return 'text-green-400';
    if (iou >= 0.3) return 'text-yellow-400';
    if (iou > 0) return 'text-red-400';
    return 'text-gray-500';
  };
  
  const SortIcon = ({ field }: { field: string }) => (
    <span className="ml-1 text-xs">
      {sortBy === field ? (sortOrder === 'desc' ? '▼' : '▲') : '↕'}
    </span>
  );
  
  return (
    <div className="w-full h-screen bg-gray-900 flex flex-col">
      <header className="h-12 bg-gray-800 border-b border-gray-700 flex items-center px-4">
        <h1 className="text-white font-semibold">多视角位姿标注工具</h1>
        <span className="ml-4 text-gray-400 text-sm">共 {stats.all} 个物体</span>
      </header>
      
      <div className="flex-1 flex flex-col p-4 overflow-hidden">
        {error && (
          <div className="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded mb-4 shrink-0">
            {error}
            <button onClick={() => reload(page)} className="ml-4 px-2 py-1 bg-red-600 rounded text-sm hover:bg-red-500">重试</button>
          </div>
        )}
        
        {/* 状态统计标签 */}
        <div className="flex items-center gap-2 mb-3 shrink-0">
          {[
            { key: '', label: '全部', count: stats.all, cls: 'bg-gray-700' },
            { key: 'pending', label: '待标注', count: stats.pending, cls: 'bg-gray-600' },
            { key: 'aligned', label: '已对齐', count: stats.aligned, cls: 'bg-green-700' },
            { key: 'invalid', label: '无效', count: stats.invalid, cls: 'bg-red-700' },
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => handleStatusFilter(tab.key)}
              className={`px-3 py-1.5 rounded text-sm transition-colors ${
                statusFilter === tab.key
                  ? `${tab.cls} text-white ring-2 ring-white/30`
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {tab.label} <span className="ml-1 text-xs opacity-75">{tab.count}</span>
            </button>
          ))}
          
          <div className="ml-auto flex items-center gap-3">
            <div className="flex items-center gap-2">
              <span className="text-gray-400 text-sm">场景:</span>
              <select value={selectedScene} onChange={(e) => handleSceneChange(e.target.value)}
                className="bg-gray-700 text-white px-2 py-1 rounded text-sm">
                <option value="">全部</option>
                {scenes.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-gray-400 text-sm">帧数:</span>
              <select value={numFrames} onChange={(e) => setNumFrames(parseInt(e.target.value))}
                className="bg-gray-700 text-white px-2 py-1 rounded text-sm">
                {[4, 6, 8, 12, 16].map(n => <option key={n} value={n}>{n}帧</option>)}
              </select>
            </div>
          </div>
        </div>
        
        {loading ? (
          <div className="text-gray-400 text-center py-8">加载中...</div>
        ) : (
          <>
            {/* 表格 */}
            <div className="flex-1 overflow-auto rounded-lg border border-gray-700">
              <table className="w-full text-sm">
                <thead className="bg-gray-800 sticky top-0 z-10">
                  <tr>
                    <th className="text-left text-gray-400 font-medium px-3 py-2.5 w-20">预览</th>
                    <th className="text-left text-gray-400 font-medium px-3 py-2.5">物体 ID</th>
                    <th className="text-left text-gray-400 font-medium px-3 py-2.5 w-24">场景</th>
                    <th className="text-left text-gray-400 font-medium px-3 py-2.5 w-24 cursor-pointer hover:text-white select-none"
                        onClick={() => handleSort('num_point_pairs')}>
                      标注点数<SortIcon field="num_point_pairs" />
                    </th>
                    <th className="text-left text-gray-400 font-medium px-3 py-2.5 w-28 cursor-pointer hover:text-white select-none"
                        onClick={() => handleSort('average_iou')}>
                      对齐质量<SortIcon field="average_iou" />
                    </th>
                    <th className="text-left text-gray-400 font-medium px-3 py-2.5 w-20">状态</th>
                    <th className="text-right text-gray-400 font-medium px-3 py-2.5 w-20">操作</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {mvObjects.map((obj) => {
                    const status = getStatusInfo(obj);
                    const iou = obj.average_iou || 0;
                    const points = obj.num_point_pairs || 0;
                    return (
                      <tr key={`${obj.scene_id}-${obj.object_id}`}
                          className="bg-gray-900 hover:bg-gray-800/70 transition-colors">
                        <td className="px-3 py-2">
                          {obj.thumbnail_url ? (
                            <img
                              src={`${MV_DATA_SERVER}${obj.thumbnail_url}`}
                              alt=""
                              className="w-16 h-12 object-cover rounded bg-gray-700"
                              loading="lazy"
                              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                            />
                          ) : (
                            <div className="w-16 h-12 bg-gray-700 rounded flex items-center justify-center text-gray-500 text-xs">N/A</div>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          <div className="text-white truncate max-w-[240px] text-xs font-mono" title={obj.object_id}>
                            {obj.object_id}
                          </div>
                        </td>
                        <td className="px-3 py-2 text-gray-400 text-xs">{obj.scene_id}</td>
                        <td className="px-3 py-2">
                          {points > 0 ? (
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                                <div className="h-full bg-blue-500 rounded-full" style={{ width: `${Math.min(points / 10, 1) * 100}%` }} />
                              </div>
                              <span className="text-gray-300 text-xs">{points}</span>
                            </div>
                          ) : (
                            <span className="text-gray-600 text-xs">-</span>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          {iou > 0 ? (
                            <span className={`text-xs font-bold ${getIoUColor(iou)}`}>
                              {(iou * 100).toFixed(1)}%
                            </span>
                          ) : (
                            <span className="text-gray-600 text-xs">-</span>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          <span className={`px-2 py-0.5 rounded text-xs text-white ${status.cls}`}>
                            {status.text}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-right">
                          <button
                            onClick={() => handleSelect(obj)}
                            disabled={loadingData}
                            className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs disabled:opacity-50"
                          >
                            标注
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            
            {/* 分页 */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-3 shrink-0">
                <button onClick={() => handlePageChange(1)} disabled={page === 1}
                  className="px-3 py-1 bg-gray-700 text-white rounded text-sm disabled:opacity-50">首页</button>
                <button onClick={() => handlePageChange(page - 1)} disabled={page === 1}
                  className="px-3 py-1 bg-gray-700 text-white rounded text-sm disabled:opacity-50">上一页</button>
                <span className="text-gray-400 text-sm px-4">{page} / {totalPages}</span>
                <button onClick={() => handlePageChange(page + 1)} disabled={page === totalPages}
                  className="px-3 py-1 bg-gray-700 text-white rounded text-sm disabled:opacity-50">下一页</button>
                <button onClick={() => handlePageChange(totalPages)} disabled={page === totalPages}
                  className="px-3 py-1 bg-gray-700 text-white rounded text-sm disabled:opacity-50">末页</button>
              </div>
            )}
          </>
        )}
        
        {loadingData && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-gray-800 px-6 py-4 rounded-lg text-white">加载数据中...</div>
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
  const maskOpacity = useMVAnnotationStore((state) => state.maskOpacity);
  const setMaskOpacity = useMVAnnotationStore((state) => state.setMaskOpacity);
  const isSavingNext = useMVAnnotationStore((state) => state.isSavingNext);
  
  // 用于通知数据选择器更新对象状态
  const [savedObjectInfo, setSavedObjectInfo] = useState<{objectId: string; category: string} | null>(null);
  // 记录当前使用的帧数设置
  const [numFrames, setNumFrames] = useState(4);
  
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
  
  // 保存并处理下一个
  const handleSaveAndNext = useCallback(async () => {
    const store = useMVAnnotationStore.getState();
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
      
      // 通知数据选择器更新（乐观更新）
      const finalCategory = store.category === 'invalid' ? 'invalid' : 'fixed';
      setSavedObjectInfo({
        objectId: store.currentInput!.objectId,
        category: finalCategory
      });
      
      // 解析当前objectId获取scene_id, object_id
      const parts = store.currentInput!.objectId.split('_');
      const currentSceneId = parts[0];
      const currentObjectId = parts.slice(1).join('_');
      
      // 步骤2：获取下一个待标注任务
      const nextTaskResponse = await fetchNextMVTask(currentSceneId, currentObjectId);
      
      if (!nextTaskResponse.success || !nextTaskResponse.has_next || !nextTaskResponse.data) {
        showGlobalToast('所有物体已标注完成！', 'info', 5000);
        store.setIsSavingNext(false);
        reset();
        return;
      }
      
      store.setRemainingCount(nextTaskResponse.remaining_count - 1);
      
      // 步骤3：加载下一个任务的数据
      const nextData = nextTaskResponse.data;
      const objectData = await loadMVObjectData(
        nextData.scene_id,
        nextData.object_id,
        numFrames
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
      useMVAnnotationStore.getState().setIsSavingNext(false);
    }
  }, [reset, setCurrentInput, numFrames]);
  
  // 放弃并处理下一个（标记为无效，保存后跳到下一个）
  const handleSkipAndNext = useCallback(async () => {
    const store = useMVAnnotationStore.getState();
    if (store.isSavingNext || !store.currentInput) return;
    
    store.setIsSavingNext(true);
    
    try {
      // 标记为无效并保存
      await store.classifyAsInvalid();
      showGlobalToast('已标记为无效数据', 'warning', 1500);
      
      // 通知列表更新
      setSavedObjectInfo({
        objectId: store.currentInput!.objectId,
        category: 'invalid'
      });
      
      // 解析当前objectId获取scene_id, object_id
      const parts = store.currentInput!.objectId.split('_');
      const currentSceneId = parts[0];
      const currentObjectId = parts.slice(1).join('_');
      
      // 获取下一个待标注任务（后端会自动跳过无效数据）
      const nextTaskResponse = await fetchNextMVTask(currentSceneId, currentObjectId);
      
      if (!nextTaskResponse.success || !nextTaskResponse.has_next || !nextTaskResponse.data) {
        showGlobalToast('没有更多待标注物体了', 'info', 5000);
        store.setIsSavingNext(false);
        reset();
        return;
      }
      
      store.setRemainingCount(nextTaskResponse.remaining_count - 1);
      
      // 加载下一个任务的数据
      const nextData = nextTaskResponse.data;
      const objectData = await loadMVObjectData(
        nextData.scene_id,
        nextData.object_id,
        numFrames
      );
      
      if (!objectData) {
        showGlobalToast('加载下一个物体数据失败', 'error', 4000);
        store.setIsSavingNext(false);
        return;
      }
      
      // 切换到新的标注对象
      setCurrentInput(objectData);
      showGlobalToast(
        `已跳转下一个物体 (剩余 ${nextTaskResponse.remaining_count - 1} 个)`,
        'success',
        3000
      );
      
    } catch (error) {
      console.error('[handleSkipAndNext] error:', error);
      showGlobalToast(`操作失败: ${error}`, 'error', 4000);
    } finally {
      useMVAnnotationStore.getState().setIsSavingNext(false);
    }
  }, [reset, setCurrentInput, numFrames]);
  
  // 快捷键支持
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+Enter / Cmd+Enter: 保存并处理下一个
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key === 'Enter') {
        e.preventDefault();
        handleSaveAndNext();
      }
      
      // Ctrl+Shift+Enter / Cmd+Shift+Enter: 放弃并处理下一个
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'Enter') {
        e.preventDefault();
        handleSkipAndNext();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleSaveAndNext, handleSkipAndNext]);
  
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
        
        {/* Mask透明度控制 - 放在顶部便于快速访问 */}
        <div className="ml-auto flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <span>Mask</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={maskOpacity}
              onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
              className="w-24"
            />
            <span className="w-8">{(maskOpacity * 100).toFixed(0)}%</span>
          </div>
          <div className="text-sm text-gray-400">
            {currentInput.frames.length} 帧
          </div>
          <button
            onClick={async () => {
              const store = useMVAnnotationStore.getState();
              if (store.isSavingNext) return;
              
              store.setIsSavingNext(true);
              try {
                // 标记为无效并保存
                await store.classifyAsInvalid();
                showGlobalToast('已标记为无效数据', 'warning', 1500);
                
                // 通知列表更新
                if (store.currentInput) {
                  setSavedObjectInfo({
                    objectId: store.currentInput.objectId,
                    category: 'invalid'
                  });
                }
                
                // 获取并跳转到下一个
                const parts = store.currentInput!.objectId.split('_');
                const nextTaskResponse = await fetchNextMVTask(parts[0], parts.slice(1).join('_'));
                
                if (!nextTaskResponse.success || !nextTaskResponse.has_next || !nextTaskResponse.data) {
                  showGlobalToast('没有更多待标注物体了', 'info', 5000);
                  store.setIsSavingNext(false);
                  reset();
                  return;
                }
                
                store.setRemainingCount(nextTaskResponse.remaining_count - 1);
                const objectData = await loadMVObjectData(nextTaskResponse.data.scene_id, nextTaskResponse.data.object_id, numFrames);
                
                if (objectData) {
                  setCurrentInput(objectData);
                  showGlobalToast(`已跳转下一个 (剩余 ${nextTaskResponse.remaining_count - 1} 个)`, 'success', 2000);
                } else {
                  showGlobalToast('加载下一个物体失败', 'error', 4000);
                }
              } catch (err) {
                showGlobalToast(`操作失败: ${err}`, 'error', 4000);
              } finally {
                useMVAnnotationStore.getState().setIsSavingNext(false);
              }
            }}
            disabled={isSavingNext}
            className={`px-3 py-1 rounded text-white text-sm ${isSavingNext ? 'bg-gray-600 cursor-not-allowed' : 'bg-red-600 hover:bg-red-500'}`}
          >
            ✗ 无效数据
          </button>
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
                  meshPath={currentInput.meshPath}
                  pose={currentPose ?? null}
                  isActive={true}
                  onSelect={() => {}}
                  serverUrl={MV_DATA_SERVER}
                  enlarged={true}
                  splitView={true}
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
                meshPath={currentInput.meshPath}
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
            <MVControlPanel onSaveAndNext={handleSaveAndNext} onSkipAndNext={handleSkipAndNext} />
          </div>
        </div>
      </div>
      
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
      
      {/* Toast 容器 */}
      <ToastContainer />
    </div>
  );
}

export default MVPoseAnnotationTool;
