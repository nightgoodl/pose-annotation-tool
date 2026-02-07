#!/usr/bin/env python3
"""调试脚本：检查 bbox 和深度图反投影点云在世界坐标系的对齐情况"""

import numpy as np
import json
import os
import cv2
from PIL import Image
import glob

# 测试数据路径
SCENE_ID = "42444908"
OBJECT_ID = "01ae51db-310c-41fe-b110-ea5212c6b2e1"

# 数据目录
DATA_DIR = "/root/csz/data_partcrafter/LASA1M"
SCENE_DIR = os.path.join(DATA_DIR, SCENE_ID)
OBJECT_DIR = os.path.join(SCENE_DIR, OBJECT_ID)

def load_instances_json():
    """加载 instances.json"""
    instances_path = os.path.join(SCENE_DIR, "instances.json")
    with open(instances_path, 'r') as f:
        return json.load(f)

def load_info_json():
    """加载 info.json"""
    info_path = os.path.join(OBJECT_DIR, "info.json")
    with open(info_path, 'r') as f:
        return json.load(f)

def load_frame_data(timestamp, info):
    """加载帧数据"""
    # 从 info.json 中找到对应帧
    frame_info = None
    for f in info:
        if f['timestamp'] == timestamp:
            frame_info = f
            break
    
    if frame_info is None:
        return None, None, None, None
    
    # 加载深度图
    depth_dir = os.path.join(OBJECT_DIR, "gt", str(timestamp))
    depth_path = os.path.join(depth_dir, "depth.png")
    
    if os.path.exists(depth_path):
        # depth.png 是 16 位整数，需要转换
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth_img.astype(np.float32) / 1000.0  # 转换为米
    else:
        depth = None
    
    # 加载 mask
    mask_path = os.path.join(OBJECT_DIR, "mask", f"{timestamp}.png")
    if os.path.exists(mask_path):
        mask = np.array(Image.open(mask_path).convert('L')) > 0
    else:
        mask = None
    
    # 相机参数
    K = np.array(frame_info['gt_depth_K'])
    RT = np.array(frame_info['gt_RT'])  # world-to-camera
    
    return depth, mask, K, RT

def unproject_depth_to_world(depth, mask, K, RT, rt_is_c2w=False):
    """将深度图根据 mask 反投影到世界坐标系
    
    Args:
        depth: 深度图 (H, W)
        mask: 物体 mask (H, W)
        K: 内参矩阵 (3, 3)
        RT: 变换矩阵 (4, 4)
        rt_is_c2w: True 表示 RT 是 camera-to-world，False 表示 world-to-camera
    """
    H, W = depth.shape
    
    # 内参
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # 获取有效像素
    if mask is not None:
        # 调整 mask 大小以匹配深度图
        if mask.shape != depth.shape:
            mask_resized = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        else:
            mask_resized = mask
        valid_mask = mask_resized & (depth > 0)
    else:
        valid_mask = depth > 0
    
    ys, xs = np.where(valid_mask)
    depths = depth[valid_mask]
    
    # 反投影到相机坐标系
    cam_x = (xs - cx) * depths / fx
    cam_y = (ys - cy) * depths / fy
    cam_z = depths
    
    # 相机坐标系点
    cam_points = np.stack([cam_x, cam_y, cam_z], axis=1)  # (N, 3)
    
    # 获取 camera-to-world 变换
    if rt_is_c2w:
        c2w = RT
    else:
        c2w = np.linalg.inv(RT)
    
    # 转换到世界坐标系
    cam_points_homo = np.concatenate([cam_points, np.ones((len(cam_points), 1))], axis=1)  # (N, 4)
    world_points = (c2w @ cam_points_homo.T).T[:, :3]  # (N, 3)
    
    return world_points

def get_bbox_corners(bbox_info):
    """获取 bbox 的 8 个角点（世界坐标系）
    
    优先使用 instances.json 中直接存储的 corners 字段
    """
    # 优先使用直接存储的 corners
    if 'corners' in bbox_info and bbox_info['corners'] is not None:
        return np.array(bbox_info['corners'], dtype=np.float32)
    
    # 如果没有 corners，则用 position/scale/R 计算
    center = np.array(bbox_info['position'])
    scale = np.array(bbox_info['scale'])  # 半轴长度
    R = np.array(bbox_info.get('R', np.eye(3)))
    
    # 局部坐标系下的 8 个角点
    local_corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=np.float32)
    
    # 缩放
    scaled_corners = local_corners * scale
    
    # 旋转 + 平移
    world_corners = (R @ scaled_corners.T).T + center
    
    return world_corners

def main():
    print("=" * 60)
    print("调试：检查 bbox 和深度图反投影点云的对齐情况")
    print("=" * 60)
    
    # 1. 加载 instances.json
    instances = load_instances_json()
    
    # 找到目标物体
    target_instance = None
    for inst in instances:
        if inst.get('id') == OBJECT_ID:
            target_instance = inst
            break
    
    if target_instance is None:
        print(f"未找到物体 {OBJECT_ID}")
        return
    
    # bbox 信息直接在实例对象上，不是 bbox_3d 子字段
    bbox_info = {
        'position': target_instance.get('position'),
        'scale': target_instance.get('scale'),
        'R': target_instance.get('R'),
        'corners': target_instance.get('corners')
    }
    print(f"\n1. GT Bbox 信息:")
    print(f"   position: {bbox_info.get('position')}")
    print(f"   scale (半轴): {bbox_info.get('scale')}")
    print(f"   R: {bbox_info.get('R')}")
    
    # 计算 bbox 角点
    bbox_corners = get_bbox_corners(bbox_info)
    bbox_min = bbox_corners.min(axis=0)
    bbox_max = bbox_corners.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_extent = bbox_max - bbox_min
    
    print(f"\n   Bbox corners min: {bbox_min}")
    print(f"   Bbox corners max: {bbox_max}")
    print(f"   Bbox center (from corners): {bbox_center}")
    print(f"   Bbox extent: {bbox_extent}")
    print(f"   Bbox diagonal: {np.linalg.norm(bbox_extent):.4f}")
    
    # 2. 加载 info.json
    info = load_info_json()
    print(f"\n2. info.json 帧数: {len(info)}")
    
    # 获取第一帧的 timestamp
    first_frame = info[0]
    timestamp = first_frame['timestamp']
    print(f"   使用第一帧 timestamp: {timestamp}")
    
    # 3. 加载帧数据
    depth, mask, K, RT = load_frame_data(timestamp, info)
    
    if depth is None:
        print("   深度图加载失败!")
        return
    
    print(f"   深度图形状: {depth.shape}")
    print(f"   深度范围: [{depth[depth > 0].min():.4f}, {depth[depth > 0].max():.4f}]")
    if mask is not None:
        print(f"   Mask 形状: {mask.shape}, 像素数: {mask.sum()}")
    
    print(f"\n   相机内参 K:\n{K}")
    print(f"\n   相机外参 RT (world-to-camera):\n{RT}")
    
    # 4. 反投影点云 - 测试两种 RT 定义
    print(f"\n3. 反投影点云:")
    
    # 方案 A: RT 是 world-to-camera
    world_points_w2c = unproject_depth_to_world(depth, mask, K, RT, rt_is_c2w=False)
    # 方案 B: RT 是 camera-to-world  
    world_points_c2w = unproject_depth_to_world(depth, mask, K, RT, rt_is_c2w=True)
    
    print(f"\n   方案 A (RT = world-to-camera):")
    print(f"   点数: {len(world_points_w2c)}")
    if len(world_points_w2c) > 0:
        pc_center_a = world_points_w2c.mean(axis=0)
        print(f"   点云 center: {pc_center_a}")
        print(f"   与 bbox center 距离: {np.linalg.norm(bbox_center - pc_center_a):.4f}")
    
    print(f"\n   方案 B (RT = camera-to-world):")
    print(f"   点数: {len(world_points_c2w)}")
    if len(world_points_c2w) > 0:
        pc_center_b = world_points_c2w.mean(axis=0)
        print(f"   点云 center: {pc_center_b}")
        print(f"   与 bbox center 距离: {np.linalg.norm(bbox_center - pc_center_b):.4f}")
    
    # 选择距离更近的方案
    dist_a = np.linalg.norm(bbox_center - world_points_w2c.mean(axis=0)) if len(world_points_w2c) > 0 else float('inf')
    dist_b = np.linalg.norm(bbox_center - world_points_c2w.mean(axis=0)) if len(world_points_c2w) > 0 else float('inf')
    
    if dist_b < dist_a:
        print(f"\n   >>> 方案 B 更接近 bbox，RT 应该是 camera-to-world!")
        world_points = world_points_c2w
        rt_type = "camera-to-world"
    else:
        print(f"\n   >>> 方案 A 更接近 bbox，RT 应该是 world-to-camera!")
        world_points = world_points_w2c
        rt_type = "world-to-camera"
    
    print(f"\n4. 使用 {rt_type} 的反投影结果:")
    print(f"   点数: {len(world_points)}")
    
    if len(world_points) > 0:
        pc_min = world_points.min(axis=0)
        pc_max = world_points.max(axis=0)
        pc_center = world_points.mean(axis=0)
        pc_extent = pc_max - pc_min
        
        print(f"   点云 min: {pc_min}")
        print(f"   点云 max: {pc_max}")
        print(f"   点云 center: {pc_center}")
        print(f"   点云 extent: {pc_extent}")
        print(f"   点云 diagonal: {np.linalg.norm(pc_extent):.4f}")
        
        # 5. 比较 bbox 和点云
        print(f"\n4. 对比分析:")
        print(f"   Center 差异: {np.linalg.norm(bbox_center - pc_center):.4f}")
        print(f"   Bbox center:  {bbox_center}")
        print(f"   PC center:    {pc_center}")
        print(f"   差异向量:     {bbox_center - pc_center}")
        
        print(f"\n   Extent 比例:")
        print(f"   Bbox extent: {bbox_extent}")
        print(f"   PC extent:   {pc_extent}")
        print(f"   比例 (bbox/pc): {bbox_extent / (pc_extent + 1e-6)}")
        
        # 检查点云有多少在 bbox 内
        in_bbox = np.all((world_points >= bbox_min) & (world_points <= bbox_max), axis=1)
        print(f"\n   点云在 bbox 内的比例: {in_bbox.sum() / len(world_points) * 100:.1f}%")
        
        # 6. 检查 bbox 和点云的对角线比例
        bbox_diagonal = np.linalg.norm(bbox_extent)
        pc_diagonal = np.linalg.norm(pc_extent)
        print(f"\n5. 对角线比较:")
        print(f"   Bbox diagonal: {bbox_diagonal:.4f}")
        print(f"   PC diagonal:   {pc_diagonal:.4f}")
        print(f"   比例 (bbox/pc): {bbox_diagonal / (pc_diagonal + 1e-6):.4f}")

def test_bbox_projection():
    """测试 bbox 投影到图像"""
    print("\n" + "=" * 60)
    print("测试 Bbox 投影")
    print("=" * 60)
    
    # 加载数据
    instances = load_instances_json()
    target_instance = None
    for inst in instances:
        if inst.get('id') == OBJECT_ID:
            target_instance = inst
            break
    
    bbox_info = {
        'position': target_instance.get('position'),
        'scale': target_instance.get('scale'),
        'R': target_instance.get('R'),
        'corners': target_instance.get('corners')
    }
    
    info = load_info_json()
    first_frame = info[0]
    timestamp = first_frame['timestamp']
    
    # 相机参数
    K_depth = np.array(first_frame['gt_depth_K'])  # 针对深度图 512x384
    RT = np.array(first_frame['gt_RT'])  # camera-to-world
    
    # RGB 图像尺寸
    rgb_width, rgb_height = 1024, 768
    depth_width, depth_height = 512, 384
    
    # 缩放内参到 RGB 图像尺寸
    scale_x = rgb_width / depth_width
    scale_y = rgb_height / depth_height
    K_rgb = np.array([
        [K_depth[0, 0] * scale_x, K_depth[0, 1], K_depth[0, 2] * scale_x],
        [K_depth[1, 0], K_depth[1, 1] * scale_y, K_depth[1, 2] * scale_y],
        [K_depth[2, 0], K_depth[2, 1], K_depth[2, 2]]
    ])
    
    print(f"\n1. 相机参数:")
    print(f"   K_depth (512x384):\n{K_depth}")
    print(f"   K_rgb (1024x768):\n{K_rgb}")
    print(f"   RT (camera-to-world):\n{RT}")
    
    # 计算 bbox 角点
    bbox_corners = get_bbox_corners(bbox_info)
    print(f"\n2. Bbox 角点 (世界坐标系):")
    for i, corner in enumerate(bbox_corners):
        print(f"   Corner {i}: {corner}")
    
    # 投影到图像 (使用 RGB 图像尺寸的内参)
    fx, fy = K_rgb[0, 0], K_rgb[1, 1]
    cx, cy = K_rgb[0, 2], K_rgb[1, 2]
    
    # RT 是 camera-to-world，需要 world-to-camera
    R_c2w = RT[:3, :3]
    t_c2w = RT[:3, 3]  # 相机在世界坐标系的位置
    
    print(f"\n3. 投影计算:")
    print(f"   fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"   相机位置 (世界坐标系): {t_c2w}")
    
    projected_corners = []
    for i, world_pt in enumerate(bbox_corners):
        # 世界坐标 -> 相机坐标
        # P_cam = R^T @ (P_world - t_cam)
        diff = world_pt - t_c2w
        cam_pt = R_c2w.T @ diff
        
        if cam_pt[2] > 0.01:
            u = fx * cam_pt[0] / cam_pt[2] + cx
            v = fy * cam_pt[1] / cam_pt[2] + cy
            projected_corners.append((u, v))
            print(f"   Corner {i}: world={world_pt} -> cam={cam_pt} -> pixel=({u:.1f}, {v:.1f})")
        else:
            projected_corners.append(None)
            print(f"   Corner {i}: behind camera")
    
    # 计算投影的边界框
    valid_corners = [c for c in projected_corners if c is not None]
    if valid_corners:
        us = [c[0] for c in valid_corners]
        vs = [c[1] for c in valid_corners]
        print(f"\n4. 投影边界框:")
        print(f"   u: [{min(us):.1f}, {max(us):.1f}]")
        print(f"   v: [{min(vs):.1f}, {max(vs):.1f}]")
        print(f"   图像尺寸: {rgb_width}x{rgb_height}")
        
        # 检查是否在图像范围内
        if min(us) >= 0 and max(us) <= rgb_width and min(vs) >= 0 and max(vs) <= rgb_height:
            print(f"   状态: 在图像范围内 ✓")
        else:
            print(f"   状态: 超出图像范围 ✗")
            
        # 计算投影尺寸占图像的比例
        bbox_u_size = max(us) - min(us)
        bbox_v_size = max(vs) - min(vs)
        print(f"\n   投影尺寸: {bbox_u_size:.1f} x {bbox_v_size:.1f}")
        print(f"   占图像比例: {bbox_u_size/rgb_width*100:.1f}% x {bbox_v_size/rgb_height*100:.1f}%")

if __name__ == "__main__":
    main()
    test_bbox_projection()
