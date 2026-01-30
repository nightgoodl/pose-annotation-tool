/**
 * 6D Pose Annotation Tool - Type Definitions
 */

// 3x3 矩阵类型
export type Matrix3 = [
  [number, number, number],
  [number, number, number],
  [number, number, number]
];

// 4x4 矩阵类型
export type Matrix4 = [
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number]
];

// 3D 向量
export type Vector3 = [number, number, number];

// 2D 像素坐标
export type PixelCoord = { u: number; v: number };

// 3D 点
export interface Point3D {
  x: number;
  y: number;
  z: number;
}

// 点对匹配
export interface PointPair {
  id: string;
  // 模型空间局部坐标 (右侧视图)
  localPoint: Point3D;
  // 世界空间坐标 (左侧视图反投影得到)
  worldPoint: Point3D;
  // 原始像素坐标
  pixelCoord: PixelCoord;
  // 深度值
  depth: number;
}

// 物体分类状态
export type ObjectCategory = 'valid' | 'fixed' | 'invalid' | 'unclassified';

// 工作流状态
export type WorkflowState = 
  | 'classification'  // 分类阶段
  | 'annotation'      // 标注阶段
  | 'review';         // 审核阶段

// 相机内参
export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

// 标注输入数据
export interface AnnotationInput {
  objectId: string;
  rgbImage: string;          // 图像 URL
  depthMap: Float32Array;    // 深度图数据
  depthWidth: number;
  depthHeight: number;
  maskImage: string;         // 分割掩码 URL
  cadModel: string;          // 3D 模型文件路径 (.glb/.obj)
  cameraIntrinsics: CameraIntrinsics;
  cameraExtrinsics: Matrix4; // World-to-Camera 矩阵 (注意：文档中RT是camera-to-world)
  initialCoarsePose: Matrix4; // Model-to-World 初始粗配准
}

// 标注输出结果
export interface AnnotationOutput {
  objectId: string;
  category: ObjectCategory;
  worldPose: number[];       // 16位数组 (Model-to-World Matrix)
  scale: number;
  points: PointPair[];
  timestamp: number;
}

// Umeyama 算法结果
export interface UmeyamaResult {
  rotation: Matrix3;         // 3x3 旋转矩阵
  translation: Vector3;      // 平移向量
  scale: number;             // 缩放因子
  transformMatrix: Matrix4;  // 完整 4x4 变换矩阵
  error: number;             // 配准误差
}

// 应用状态
export interface AppState {
  // 当前输入数据
  currentInput: AnnotationInput | null;
  
  // 工作流状态
  workflowState: WorkflowState;
  category: ObjectCategory;
  
  // 点对匹配
  pointPairs: PointPair[];
  selectedPairId: string | null;
  
  // 计算结果
  calculatedPose: Matrix4 | null;
  calculatedScale: number;
  
  // UI 状态
  isLeftViewEnabled: boolean;
  isRightViewEnabled: boolean;
  showGhostWireframe: boolean;
  maskOpacity: number;
}
