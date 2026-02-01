/**
 * Zustand 状态管理 - 多视角位姿标注工具
 */

import { create } from 'zustand';
import type { 
  Matrix4, 
  Point3D,
  ObjectCategory,
  WorkflowState
} from '../types';
import type { 
  MVAnnotationInput, 
  MVPointPair, 
  FrameData 
} from '../types/multiview';
import { solveUmeyama, identity4 } from '../utils/math';

interface MVAnnotationStore {
  // ========== 输入数据 ==========
  currentInput: MVAnnotationInput | null;
  setCurrentInput: (input: MVAnnotationInput | null) => void;
  
  // ========== 帧管理 ==========
  activeFrameId: string | null;  // 当前活动帧
  setActiveFrameId: (frameId: string | null) => void;
  getFrameData: (frameId: string) => FrameData | undefined;
  updateFrameDepth: (frameId: string, depthMap: Float32Array, width: number, height: number) => void;
  
  // ========== 工作流状态 ==========
  workflowState: WorkflowState;
  category: ObjectCategory;
  setWorkflowState: (state: WorkflowState) => void;
  setCategory: (category: ObjectCategory) => void;
  
  // ========== 多帧点对匹配 ==========
  pointPairs: MVPointPair[];
  selectedPairId: string | null;
  addPointPair: (pair: Omit<MVPointPair, 'id'>) => void;
  removePointPair: (id: string) => void;
  selectPointPair: (id: string | null) => void;
  clearPointPairs: () => void;
  getPointPairsForFrame: (frameId: string) => MVPointPair[];
  
  // ========== 临时标记状态 ==========
  pendingLocalPoint: Point3D | null;
  pendingWorldPoint: { 
    point: Point3D; 
    pixel: { u: number; v: number }; 
    depth: number;
    frameId: string;
  } | null;
  setPendingLocalPoint: (point: Point3D | null) => void;
  setPendingWorldPoint: (data: { 
    point: Point3D; 
    pixel: { u: number; v: number }; 
    depth: number;
    frameId: string;
  } | null) => void;
  
  // ========== 计算结果 ==========
  calculatedPose: Matrix4 | null;
  calculatedScale: number;
  alignmentError: number;
  frameIoUs: Map<string, number>; // 每帧的IoU
  averageIoU: number;
  setFrameIoU: (frameId: string, iou: number) => void;
  runAlignment: () => void;
  resetAlignment: () => void;
  
  // ========== UI 状态 ==========
  isAnnotationEnabled: boolean;
  showGhostWireframe: boolean;
  maskOpacity: number;
  setAnnotationEnabled: (enabled: boolean) => void;
  setShowGhostWireframe: (show: boolean) => void;
  setMaskOpacity: (opacity: number) => void;
  
  // ========== 分类操作 ==========
  classifyAsValid: () => void;
  classifyAsFixed: () => void;
  classifyAsInvalid: () => void;
  
  // ========== 保存 ==========
  savePose: () => Promise<{ success: boolean; pose_path?: string; error?: string }>;
  
  // ========== 重置 ==========
  reset: () => void;
}

let pairIdCounter = 0;

export const useMVAnnotationStore = create<MVAnnotationStore>((set, get) => ({
  // ========== 输入数据 ==========
  currentInput: null,
  setCurrentInput: (input) => {
    set({ 
      currentInput: input,
      activeFrameId: input?.frames[0]?.frame_id ?? null,
      workflowState: 'classification',
      category: 'unclassified',
      pointPairs: [],
      selectedPairId: null,
      pendingLocalPoint: null,
      pendingWorldPoint: null,
      calculatedPose: input?.initialPose ?? null,
      calculatedScale: 1,
      alignmentError: 0,
      frameIoUs: new Map(),
      averageIoU: 0,
      isAnnotationEnabled: false
    });
  },
  
  // ========== 帧管理 ==========
  activeFrameId: null,
  setActiveFrameId: (frameId) => set({ activeFrameId: frameId }),
  
  getFrameData: (frameId) => {
    const state = get();
    return state.currentInput?.frames.find(f => f.frame_id === frameId);
  },
  
  updateFrameDepth: (frameId, depthMap, width, height) => {
    set((state) => {
      if (!state.currentInput) return state;
      const newFrames = state.currentInput.frames.map(f => {
        if (f.frame_id === frameId) {
          return { ...f, depthMap, depthWidth: width, depthHeight: height };
        }
        return f;
      });
      return {
        currentInput: {
          ...state.currentInput,
          frames: newFrames
        }
      };
    });
  },
  
  // ========== 工作流状态 ==========
  workflowState: 'classification',
  category: 'unclassified',
  setWorkflowState: (state) => set({ workflowState: state }),
  setCategory: (category) => set({ category }),
  
  // ========== 多帧点对匹配 ==========
  pointPairs: [],
  selectedPairId: null,
  
  addPointPair: (pair) => {
    const id = `mvpair_${++pairIdCounter}`;
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
  
  selectPointPair: (id) => set({ selectedPairId: id }),
  
  clearPointPairs: () => set({ 
    pointPairs: [], 
    selectedPairId: null,
    pendingLocalPoint: null,
    pendingWorldPoint: null
  }),
  
  getPointPairsForFrame: (frameId) => {
    const state = get();
    return state.pointPairs.filter(p => p.frame_id === frameId);
  },
  
  // ========== 临时标记状态 ==========
  pendingLocalPoint: null,
  pendingWorldPoint: null,
  
  setPendingLocalPoint: (point) => {
    set({ pendingLocalPoint: point });
    // 如果两边都有点，自动创建点对
    const state = get();
    if (point && state.pendingWorldPoint) {
      state.addPointPair({
        frame_id: state.pendingWorldPoint.frameId,
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
        frame_id: data.frameId,
        localPoint: state.pendingLocalPoint,
        worldPoint: data.point,
        pixelCoord: data.pixel,
        depth: data.depth
      });
    }
  },
  
  // ========== 计算结果 ==========
  calculatedPose: identity4(),
  calculatedScale: 1,
  alignmentError: 0,
  frameIoUs: new Map(),
  averageIoU: 0,
  
  setFrameIoU: (frameId, iou) => {
    set((state) => {
      const newFrameIoUs = new Map(state.frameIoUs);
      newFrameIoUs.set(frameId, iou);
      
      // 计算平均IoU
      const ious = Array.from(newFrameIoUs.values());
      const avgIoU = ious.length > 0 ? ious.reduce((a, b) => a + b, 0) / ious.length : 0;
      
      return {
        frameIoUs: newFrameIoUs,
        averageIoU: avgIoU
      };
    });
  },
  
  runAlignment: () => {
    const state = get();
    if (state.pointPairs.length < 3) {
      console.warn('需要至少3对点才能运行对齐');
      return;
    }
    
    // 使用所有帧的点对进行对齐
    const srcPoints = state.pointPairs.map(p => p.localPoint);
    const dstPoints = state.pointPairs.map(p => p.worldPoint);
    
    const result = solveUmeyama(srcPoints, dstPoints);
    
    set({
      calculatedPose: result.transformMatrix,
      calculatedScale: result.scale,
      alignmentError: result.error
    });
    
    console.log('MV Alignment result:', {
      numPoints: state.pointPairs.length,
      numFrames: new Set(state.pointPairs.map(p => p.frame_id)).size,
      scale: result.scale,
      error: result.error,
      rotation: result.rotation,
      translation: result.translation
    });
  },
  
  resetAlignment: () => {
    const state = get();
    set({
      calculatedPose: state.currentInput?.initialPose ?? null,
      calculatedScale: 1,
      alignmentError: 0,
      frameIoUs: new Map(),
      averageIoU: 0,
      pointPairs: [],
      selectedPairId: null,
      pendingLocalPoint: null,
      pendingWorldPoint: null
    });
    console.log('Reset MV alignment: cleared all point pairs and reset pose');
  },
  
  // ========== UI 状态 ==========
  isAnnotationEnabled: false,
  showGhostWireframe: true,
  maskOpacity: 0.5,
  
  setAnnotationEnabled: (enabled) => set({ isAnnotationEnabled: enabled }),
  setShowGhostWireframe: (show) => set({ showGhostWireframe: show }),
  setMaskOpacity: (opacity) => set({ maskOpacity: opacity }),
  
  // ========== 分类操作 ==========
  classifyAsValid: () => {
    set({
      category: 'valid',
      workflowState: 'annotation',
      isAnnotationEnabled: true
    });
  },
  
  classifyAsFixed: () => {
    set({
      category: 'fixed',
      workflowState: 'review'
    });
  },
  
  classifyAsInvalid: async () => {
    set({
      category: 'invalid',
      workflowState: 'review'
    });
    
    // 自动保存无效数据分类
    const state = get();
    if (state.currentInput) {
      await state.savePose();
    }
  },
  
  // ========== 保存Pose到服务器 ==========
  savePose: async () => {
    const state = get();
    if (!state.currentInput) {
      console.error('No current input to save');
      return { success: false, error: 'No current input' };
    }
    
    const pose = state.calculatedPose ?? state.currentInput.initialPose ?? identity4();
    
    // 从objectId解析scene_id, object_id
    const parts = state.currentInput.objectId.split('_');
    if (parts.length < 2) {
      console.error('Invalid objectId format:', state.currentInput.objectId);
      return { success: false, error: 'Invalid objectId format' };
    }
    
    const scene_id = parts[0];
    const object_id = parts.slice(1).join('_');
    
    // 如果不是invalid，保存时自动设置为fixed（已对齐）
    const finalCategory = state.category === 'invalid' ? 'invalid' : 'fixed';
    
    const requestData = {
      scene_id,
      object_id,
      pose,
      scale: state.calculatedScale,
      error: state.alignmentError,
      category: finalCategory,
      point_pairs: state.pointPairs.map(p => ({
        frame_id: p.frame_id,
        localPoint: p.localPoint,
        worldPoint: p.worldPoint,
        pixelCoord: p.pixelCoord
      }))
    };
    
    console.log('[saveMVPose] 保存数据:', requestData);
    
    try {
      const response = await fetch('/api/save_mv_pose', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      });
      
      const result = await response.json();
      console.log('[saveMVPose] 服务器响应:', result);
      
      if (result.success) {
        // 保存成功后，更新store中的category为实际保存的值
        set({ category: finalCategory });
        return { success: true, pose_path: result.pose_path };
      } else {
        return { success: false, error: result.error };
      }
    } catch (e) {
      console.error('[saveMVPose] 保存失败:', e);
      return { success: false, error: String(e) };
    }
  },
  
  // ========== 重置 ==========
  reset: () => {
    set({
      currentInput: null,
      activeFrameId: null,
      workflowState: 'classification',
      category: 'unclassified',
      pointPairs: [],
      selectedPairId: null,
      pendingLocalPoint: null,
      pendingWorldPoint: null,
      calculatedPose: null,
      calculatedScale: 1,
      alignmentError: 0,
      isAnnotationEnabled: false,
      showGhostWireframe: true,
      maskOpacity: 0.5
    });
  }
}));
