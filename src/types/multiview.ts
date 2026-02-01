/**
 * 多视角标注工具 - 类型定义
 */

import type { Matrix4, Point3D, PixelCoord } from './index';

// 单帧数据
export interface FrameData {
  frame_id: string;
  frame_index: number;
  rgb_url: string;
  mask_url: string | null;
  depth_url: string | null;
  camera_intrinsics: {
    K: number[][] | null;
    K_original: number[][] | null;
    width: number;
    height: number;
  };
  camera_extrinsics: Matrix4;  // camera-to-world
  // 运行时加载的深度数据
  depthMap?: Float32Array;
  depthWidth?: number;
  depthHeight?: number;
}

// 多视角物体数据
export interface MVObjectData {
  scene_id: string;
  object_id: string;
  mesh_url: string;
  mesh_path: string;
  world_pose: Matrix4 | null;
  frames: FrameData[];
  total_frames: number;
}

// 多视角点对匹配 (包含帧信息)
export interface MVPointPair {
  id: string;
  frame_id: string;           // 所属帧ID
  localPoint: Point3D;        // 模型空间局部坐标
  worldPoint: Point3D;        // 世界空间坐标
  pixelCoord: PixelCoord;     // 像素坐标
  depth: number;              // 深度值
}

// 多视角物体列表项
export interface MVObjectItem {
  scene_id: string;
  object_id: string;
  num_frames: number;
  has_alignment: boolean;
  category?: 'valid' | 'fixed' | 'invalid' | null;
}

// 多视角标注输入数据
export interface MVAnnotationInput {
  objectId: string;           // scene_id_object_id
  meshUrl: string;            // mesh文件URL
  frames: FrameData[];        // 多帧数据
  initialPose: Matrix4 | null;  // 初始pose
}

// 多视角标注输出结果
export interface MVAnnotationOutput {
  scene_id: string;
  object_id: string;
  worldPose: number[];        // 16位数组 (Model-to-World Matrix)
  scale: number;
  points: MVPointPair[];
  timestamp: number;
}
