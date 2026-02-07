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
  FrameData,
  MVPointPair
} from '../types/multiview';
import { 
  computeBboxAlignTransform, 
  solveUmeyamaConstrained, 
  solveUmeyamaConstrainedRANSAC, 
  identity3,
  identity4,
  multiply4x4
} from '../utils/math';

// Get API base path from Vite's BASE_URL
const getApiBasePath = () => {
  // @ts-ignore - import.meta.env is available in Vite
  const basePath = import.meta.env.BASE_URL || '/';
  return basePath.replace(/\/$/, ''); // Remove trailing slash
};

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
  runAlignment: (useRANSAC?: boolean) => void;
  applyBboxAlign: () => void;  // 应用纯 Bbox Align（不使用点对）
  resetAlignment: () => void;
  
  // ========== UI 状态 ==========
  isAnnotationEnabled: boolean;
  showGhostWireframe: boolean;
  maskOpacity: number;
  useNvdiffrastRender: boolean;  // 使用 nvdiffrast GPU 渲染
  setAnnotationEnabled: (enabled: boolean) => void;
  setShowGhostWireframe: (show: boolean) => void;
  setMaskOpacity: (opacity: number) => void;
  setUseNvdiffrastRender: (use: boolean) => void;
  
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
    console.log('[setCurrentInput] gtBbox:', input?.gtBbox);
    console.log('[setCurrentInput] meshInfo:', input?.meshInfo);
    set({ 
      currentInput: input,
      activeFrameId: input?.frames[0]?.frame_id ?? null,
      workflowState: 'annotation',
      category: 'valid',
      pointPairs: [],
      selectedPairId: null,
      pendingLocalPoint: null,
      pendingWorldPoint: null,
      calculatedPose: input?.initialPose ?? null,
      calculatedScale: 1,
      alignmentError: 0,
      frameIoUs: new Map(),
      averageIoU: 0,
      isAnnotationEnabled: true
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
    
    // 自动对齐：≥3 点时运行带约束 Umeyama
    const state = get();
    if (state.pointPairs.length >= 3) {
      state.runAlignment(false);
    }
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
  
  runAlignment: (useRANSAC: boolean = false) => {
    const state = get();
    if (state.pointPairs.length < 3) {
      console.warn('需要至少3对点才能运行对齐');
      return;
    }
    
    // 使用所有帧的点对进行对齐
    // 注意：mesh 是 Y-up 的（Three.js 默认），世界坐标系是 Z-up 的
    // 需要将 localPoint 从 Y-up 变换到 Z-up
    // Y-up 到 Z-up 变换: x' = x, y' = -z, z' = y
    const srcPoints = state.pointPairs.map(p => ({
      x: p.localPoint.x,
      y: -p.localPoint.z,  // Y-up 的 Z 变成 Z-up 的 -Y
      z: p.localPoint.y    // Y-up 的 Y 变成 Z-up 的 Z
    }));
    const dstPoints = state.pointPairs.map(p => p.worldPoint);
    
    // 调试：输出所有点对（变换后）
    console.log('[runAlignment] Point pairs (after Y-up to Z-up transform):');
    state.pointPairs.forEach((p, i) => {
      const src = srcPoints[i];
      console.log(`  Pair ${i}: local_orig=(${p.localPoint.x.toFixed(3)}, ${p.localPoint.y.toFixed(3)}, ${p.localPoint.z.toFixed(3)}) -> local_zup=(${src.x.toFixed(3)}, ${src.y.toFixed(3)}, ${src.z.toFixed(3)}) -> world=(${p.worldPoint.x.toFixed(3)}, ${p.worldPoint.y.toFixed(3)}, ${p.worldPoint.z.toFixed(3)})`);
    });
    
    // 调试：输出 GT bbox 信息
    if (state.currentInput?.gtBbox) {
      const gtBbox = state.currentInput.gtBbox;
      console.log('[runAlignment] GT Bbox position:', gtBbox.position);
      console.log('[runAlignment] GT Bbox scale:', gtBbox.scale);
      if (gtBbox.corners) {
        console.log('[runAlignment] GT Bbox corners (world coordinates):');
        gtBbox.corners.forEach((c: number[], i: number) => {
          console.log(`  Corner ${i}: (${c[0].toFixed(3)}, ${c[1].toFixed(3)}, ${c[2].toFixed(3)})`);
        });
      }
    }
    
    // 计算 baseRotation：直接使用 GT bbox 的 R 矩阵
    // 这保证了最终结果是 GT bbox R + 4 个旋转角度之一（0°, 90°, 180°, 270°）
    let baseRotation = identity3();
    console.log('[runAlignment] gtBbox:', state.currentInput?.gtBbox);
    
    if (state.currentInput?.gtBbox?.R) {
      // 直接使用 GT bbox 的 R 作为 baseRotation
      const R = state.currentInput.gtBbox.R;
      baseRotation = [
        [R[0][0], R[0][1], R[0][2]],
        [R[1][0], R[1][1], R[1][2]],
        [R[2][0], R[2][1], R[2][2]]
      ];
      console.log('[runAlignment] 使用 gtBbox.R 作为 baseRotation');
    } else {
      console.log('[runAlignment] 无 gtBbox.R，使用单位旋转');
    }
    
    // Y-up 到 Z-up 变换矩阵（用于最终变换矩阵）
    // 因为 mesh 渲染时是 Y-up 的，所以最终变换矩阵需要包含这个变换
    // dst = R * S * R_yup_to_zup * src_yup + t
    const R_yup_to_zup: Matrix4 = [
      [1, 0, 0, 0],
      [0, 0, -1, 0],
      [0, 1, 0, 0],
      [0, 0, 0, 1]
    ];
    
    if (useRANSAC && state.pointPairs.length >= 5) {
      // RANSAC + 带约束 Umeyama（手动触发）
      const result = solveUmeyamaConstrainedRANSAC(srcPoints, dstPoints, baseRotation, {
        maxIterations: 100,
        inlierThreshold: 0.05  // 5cm 阈值
      });
      
      // 最终变换矩阵 = result.transformMatrix * R_yup_to_zup
      const finalTransform = multiply4x4(result.transformMatrix, R_yup_to_zup);
      
      set({
        calculatedPose: finalTransform,
        calculatedScale: result.scale,
        alignmentError: result.error
      });
      
      console.log('MV Alignment (Constrained RANSAC) result:', {
        numPoints: state.pointPairs.length,
        numFrames: new Set(state.pointPairs.map(p => p.frame_id)).size,
        inliers: result.inlierIndices.length,
        outliers: result.outlierIndices.length,
        scale: result.scale,
        error: result.error
      });
    } else {
      // 普通带约束 Umeyama（自动触发）
      const result = solveUmeyamaConstrained(srcPoints, dstPoints, baseRotation);
      
      // 最终变换矩阵 = result.transformMatrix * R_yup_to_zup
      const finalTransform = multiply4x4(result.transformMatrix, R_yup_to_zup);
      
      set({
        calculatedPose: finalTransform,
        calculatedScale: result.scale,
        alignmentError: result.error
      });
      
      console.log('MV Alignment (Constrained) result:', {
        numPoints: state.pointPairs.length,
        numFrames: new Set(state.pointPairs.map(p => p.frame_id)).size,
        scale: result.scale,
        error: result.error
      });
    }
  },
  
  applyBboxAlign: () => {
    const state = get();
    if (!state.currentInput?.meshInfo || !state.currentInput?.gtBbox) {
      console.warn('[applyBboxAlign] 需要 meshInfo 和 gtBbox');
      return;
    }
    
    const bboxAlignPose = computeBboxAlignTransform(
      state.currentInput.meshInfo,
      state.currentInput.gtBbox
    );
    
    set({
      calculatedPose: bboxAlignPose,
      calculatedScale: 1,  // scale 已经包含在 transform 中
      alignmentError: 0
    });
    
    console.log('[applyBboxAlign] 应用纯 Bbox Align 变换');
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
  useNvdiffrastRender: true,  // 默认使用 nvdiffrast 渲染
  
  setAnnotationEnabled: (enabled) => set({ isAnnotationEnabled: enabled }),
  setUseNvdiffrastRender: (use) => set({ useNvdiffrastRender: use }),
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
      const apiPath = `${getApiBasePath()}/api/save_mv_pose`;
      const response = await fetch(apiPath, {
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
      maskOpacity: 0.5,
      useNvdiffrastRender: true
    });
  }
}));
