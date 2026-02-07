/**
 * Zustand 状态管理 - 位姿标注工具
 */

import { create } from 'zustand';
import type { 
  AnnotationInput, 
  AnnotationOutput, 
  ObjectCategory, 
  WorkflowState, 
  PointPair, 
  Matrix4, 
  Point3D 
} from '../types';
import { solveUmeyama } from '../utils/math';

// Get API base path from Vite's BASE_URL
const getApiBasePath = () => {
  // @ts-ignore - import.meta.env is available in Vite
  const basePath = import.meta.env.BASE_URL || '/';
  return basePath.replace(/\/$/, ''); // Remove trailing slash
};

interface AnnotationStore {
  // ========== 输入数据 ==========
  currentInput: AnnotationInput | null;
  setCurrentInput: (input: AnnotationInput | null) => void;
  
  // ========== 工作流状态 ==========
  workflowState: WorkflowState;
  category: ObjectCategory;
  setWorkflowState: (state: WorkflowState) => void;
  setCategory: (category: ObjectCategory) => void;
  
  // ========== 点对匹配 ==========
  pointPairs: PointPair[];
  selectedPairId: string | null;
  addPointPair: (pair: Omit<PointPair, 'id'>) => void;
  removePointPair: (id: string) => void;
  updatePointPair: (id: string, updates: Partial<PointPair>) => void;
  selectPointPair: (id: string | null) => void;
  clearPointPairs: () => void;
  
  // ========== 临时标记状态 (用于两侧视图同步) ==========
  pendingLocalPoint: Point3D | null;
  pendingWorldPoint: { point: Point3D; pixel: { u: number; v: number }; depth: number } | null;
  setPendingLocalPoint: (point: Point3D | null) => void;
  setPendingWorldPoint: (data: { point: Point3D; pixel: { u: number; v: number }; depth: number } | null) => void;
  
  // ========== 计算结果 ==========
  calculatedPose: Matrix4 | null;
  calculatedScale: number;
  alignmentError: number;
  currentIoU: number | null;
  setCurrentIoU: (iou: number | null) => void;
  runAlignment: () => void;
  resetAlignment: () => void;
  
  // ========== UI 状态 ==========
  isLeftViewEnabled: boolean;
  isRightViewEnabled: boolean;
  showGhostWireframe: boolean;
  maskOpacity: number;
  setLeftViewEnabled: (enabled: boolean) => void;
  setRightViewEnabled: (enabled: boolean) => void;
  setShowGhostWireframe: (show: boolean) => void;
  setMaskOpacity: (opacity: number) => void;
  
  // ========== 分类操作 ==========
  classifyAsValid: () => void;
  classifyAsFixed: () => void;
  classifyAsInvalid: () => void;
  
  // ========== 导出 ==========
  exportAnnotation: () => AnnotationOutput | null;
  savePose: () => Promise<{ success: boolean; pose_path?: string; error?: string }>;
  
  // ========== 重置 ==========
  reset: () => void;
}

let pairIdCounter = 0;

export const useAnnotationStore = create<AnnotationStore>((set, get) => ({
  // ========== 输入数据 ==========
  currentInput: null,
  setCurrentInput: (input) => {
    set({ 
      currentInput: input,
      // 重置相关状态
      workflowState: 'classification',
      category: 'unclassified',
      pointPairs: [],
      selectedPairId: null,
      pendingLocalPoint: null,
      pendingWorldPoint: null,
      calculatedPose: input?.initialCoarsePose ?? null,
      calculatedScale: 1,
      alignmentError: 0,
      currentIoU: null,
      isLeftViewEnabled: false,
      isRightViewEnabled: false
    });
  },
  
  // ========== 工作流状态 ==========
  workflowState: 'classification',
  category: 'unclassified',
  setWorkflowState: (state) => set({ workflowState: state }),
  setCategory: (category) => set({ category }),
  
  // ========== 点对匹配 ==========
  pointPairs: [],
  selectedPairId: null,
  
  addPointPair: (pair) => {
    const id = `pair_${++pairIdCounter}`;
    set((state) => ({
      pointPairs: [...state.pointPairs, { ...pair, id }],
      pendingLocalPoint: null,
      pendingWorldPoint: null
    }));
  },
  
  removePointPair: (id) => {
    set((state) => ({
      pointPairs: state.pointPairs.filter(p => p.id !== id),
      selectedPairId: state.selectedPairId === id ? null : state.selectedPairId
    }));
  },
  
  updatePointPair: (id, updates) => {
    set((state) => ({
      pointPairs: state.pointPairs.map(p => 
        p.id === id ? { ...p, ...updates } : p
      )
    }));
  },
  
  selectPointPair: (id) => set({ selectedPairId: id }),
  
  clearPointPairs: () => set({ 
    pointPairs: [], 
    selectedPairId: null,
    pendingLocalPoint: null,
    pendingWorldPoint: null
  }),
  
  // ========== 临时标记状态 ==========
  pendingLocalPoint: null,
  pendingWorldPoint: null,
  
  setPendingLocalPoint: (point) => {
    set({ pendingLocalPoint: point });
    // 如果两边都有点，自动创建点对
    const state = get();
    if (point && state.pendingWorldPoint) {
      state.addPointPair({
        localPoint: point,
        worldPoint: state.pendingWorldPoint.point,
        pixelCoord: state.pendingWorldPoint.pixel,
        depth: state.pendingWorldPoint.depth
      });
    }
  },
  
  setPendingWorldPoint: (data) => {
    set({ pendingWorldPoint: data });
    // 如果两边都有点，自动创建点对
    const state = get();
    if (data && state.pendingLocalPoint) {
      state.addPointPair({
        localPoint: state.pendingLocalPoint,
        worldPoint: data.point,
        pixelCoord: data.pixel,
        depth: data.depth
      });
    }
  },
  
  // ========== 计算结果 ==========
  calculatedPose: null,
  calculatedScale: 1,
  alignmentError: 0,
  currentIoU: null,
  setCurrentIoU: (iou) => set({ currentIoU: iou }),
  
  runAlignment: () => {
    const state = get();
    if (state.pointPairs.length < 3) {
      console.warn('需要至少3对点才能运行对齐');
      return;
    }
    
    const srcPoints = state.pointPairs.map(p => p.localPoint);
    const dstPoints = state.pointPairs.map(p => p.worldPoint);
    
    const result = solveUmeyama(srcPoints, dstPoints);
    
    set({
      calculatedPose: result.transformMatrix,
      calculatedScale: result.scale,
      alignmentError: result.error
    });
    
    console.log('Alignment result:', {
      scale: result.scale,
      error: result.error,
      rotation: result.rotation,
      translation: result.translation
    });
  },
  
  resetAlignment: () => {
    const state = get();
    set({
      calculatedPose: state.currentInput?.initialCoarsePose ?? null,
      calculatedScale: 1,
      alignmentError: 0,
      pointPairs: [],
      selectedPairId: null,
      pendingLocalPoint: null,
      pendingWorldPoint: null
    });
    console.log('Reset alignment: cleared all point pairs and reset pose');
  },
  
  // ========== UI 状态 ==========
  isLeftViewEnabled: false,
  isRightViewEnabled: false,
  showGhostWireframe: true,
  maskOpacity: 0.5,
  
  setLeftViewEnabled: (enabled) => set({ isLeftViewEnabled: enabled }),
  setRightViewEnabled: (enabled) => set({ isRightViewEnabled: enabled }),
  setShowGhostWireframe: (show) => set({ showGhostWireframe: show }),
  setMaskOpacity: (opacity) => set({ maskOpacity: opacity }),
  
  // ========== 分类操作 ==========
  classifyAsValid: () => {
    set({
      category: 'valid',
      workflowState: 'annotation',
      isLeftViewEnabled: true,
      isRightViewEnabled: true
    });
  },
  
  classifyAsFixed: () => {
    set({
      category: 'fixed',
      workflowState: 'review'
    });
    // TODO: 触发自动保存并跳转下一张
  },
  
  classifyAsInvalid: () => {
    set({
      category: 'invalid',
      workflowState: 'review'
    });
    // TODO: 触发跳转下一张
  },
  
  // ========== 导出 ==========
  exportAnnotation: () => {
    const state = get();
    if (!state.currentInput) return null;
    
    const pose = state.calculatedPose ?? state.currentInput.initialCoarsePose;
    
    // 将 Matrix4 转为 16 元素数组 (行优先)
    const worldPoseArray: number[] = [];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        worldPoseArray.push(pose[i][j]);
      }
    }
    
    return {
      objectId: state.currentInput.objectId,
      category: state.category,
      worldPose: worldPoseArray,
      scale: state.calculatedScale,
      points: state.pointPairs,
      timestamp: Date.now()
    };
  },
  
  // ========== 保存Pose到服务器 ==========
  savePose: async () => {
    const state = get();
    if (!state.currentInput) {
      console.error('No current input to save');
      return { success: false, error: 'No current input' };
    }
    
    const pose = state.calculatedPose ?? state.currentInput.initialCoarsePose;
    
    // 从objectId解析scene_id, object_id, frame_id
    // objectId格式: "scene_id_object_id_frame_id"
    const parts = state.currentInput.objectId.split('_');
    if (parts.length < 3) {
      console.error('Invalid objectId format:', state.currentInput.objectId);
      return { success: false, error: 'Invalid objectId format' };
    }
    
    const scene_id = parts[0];
    const frame_id = parts[parts.length - 1];
    // object_id可能包含下划线，所以取中间部分
    const object_id = parts.slice(1, -1).join('_');
    
    const requestData = {
      scene_id,
      object_id,
      frame_id,
      pose,
      scale: state.calculatedScale,
      error: state.alignmentError,
      point_pairs: state.pointPairs.map(p => ({
        localPoint: p.localPoint,
        worldPoint: p.worldPoint,
        pixelCoord: p.pixelCoord
      }))
    };
    
    console.log('[savePose] 保存数据:', requestData);
    
    try {
      const apiPath = `${getApiBasePath()}/api/save_pose`;
      const response = await fetch(apiPath, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      });
      
      const result = await response.json();
      console.log('[savePose] 服务器响应:', result);
      
      if (result.success) {
        return { success: true, pose_path: result.pose_path };
      } else {
        return { success: false, error: result.error };
      }
    } catch (e) {
      console.error('[savePose] 保存失败:', e);
      return { success: false, error: String(e) };
    }
  },
  
  // ========== 重置 ==========
  reset: () => {
    set({
      currentInput: null,
      workflowState: 'classification',
      category: 'unclassified',
      pointPairs: [],
      selectedPairId: null,
      pendingLocalPoint: null,
      pendingWorldPoint: null,
      calculatedPose: null,
      calculatedScale: 1,
      alignmentError: 0,
      currentIoU: null,
      isLeftViewEnabled: false,
      isRightViewEnabled: false,
      showGhostWireframe: true,
      maskOpacity: 0.5
    });
  }
}));
