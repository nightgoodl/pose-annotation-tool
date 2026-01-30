#!/usr/bin/env python3
"""
验证手动标注的pose是否正确
将decode出来的mesh应用pose后投影到相机平面
"""

import numpy as np
import json
import trimesh
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
import sys

# 添加项目路径
sys.path.insert(0, '/root/csz/yingbo/sam-3d-objects')

# 配置
SCENE_ID = "43649408"
OBJECT_ID = "ea55f94a-fabe-46d6-97c8-db1e5cc77e04"
FRAME_ID = "10099387528125"

DATA_ROOT = "/root/csz/data_partcrafter"
LASA1M_ROOT = f"{DATA_ROOT}/LASA1M"
STAGE2_ROOT = f"{DATA_ROOT}/LASA1M_SAM_STAGE2_V3_DISTRIBUTED"
MANUAL_ROOT = f"{DATA_ROOT}/LASA1M_ALIGNED_Manual"
CHECKPOINT_DIR = "/root/csz/yingbo/sam-3d-objects/checkpoints/partcrafter"

def load_camera_params(scene_id, object_id, frame_id, image_width, image_height):
    """加载相机参数，并根据图像尺寸缩放内参"""
    info_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/info.json"
    with open(info_path) as f:
        info = json.load(f)
    
    frame_timestamp = int(frame_id)
    for item in info:
        if item.get('timestamp') == frame_timestamp:
            K_original = np.array(item['gt_depth_K'])  # 3x3 内参（针对深度图尺寸）
            RT = np.array(item['gt_RT'])               # 4x4 camera-to-world
            
            # 深度图尺寸：512x384 (宽x高)
            depth_width = 512
            depth_height = 384
            
            # 根据实际图像尺寸缩放内参
            scale_x = image_width / depth_width
            scale_y = image_height / depth_height
            print(f"内参缩放比例: x={scale_x}, y={scale_y}")
            
            K_scaled = K_original.copy()
            K_scaled[0, 0] *= scale_x  # fx
            K_scaled[0, 2] *= scale_x  # cx
            K_scaled[1, 1] *= scale_y  # fy
            K_scaled[1, 2] *= scale_y  # cy
            
            print(f"原始内参 K:\n{K_original}")
            print(f"缩放后内参 K:\n{K_scaled}")
            
            return K_scaled, RT
    
    raise ValueError(f"Frame {frame_id} not found")

def decode_mesh_via_service(scene_id, object_id):
    """通过decoder service获取mesh（使用缓存）"""
    import requests
    
    # 调用decoder service
    url = f"http://localhost:8083/decode?scene_id={scene_id}&object_id={object_id}&texture=false"
    print(f"请求decoder service: {url}")
    
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Decoder service error: {response.text}")
    
    result = response.json()
    if not result.get('success'):
        raise RuntimeError(f"Decode failed: {result}")
    
    # 加载生成的mesh
    mesh_url = result['mesh_url']
    mesh_path = f"/tmp/mesh_decoder_cache/{os.path.basename(mesh_url)}"
    print(f"加载mesh: {mesh_path}")
    
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh

def load_pose(scene_id, object_id, frame_id):
    """加载手动标注的pose"""
    pose_path = f"{MANUAL_ROOT}/{scene_id}/{object_id}/world_pose_{frame_id}.npy"
    print(f"加载pose: {pose_path}")
    return np.load(pose_path)

def load_rgb_image(scene_id, object_id, frame_id):
    """加载RGB图像"""
    rgb_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/raw_jpg/{frame_id}.jpg"
    print(f"加载图像: {rgb_path}")
    return Image.open(rgb_path)

def project_mesh_to_image(mesh, world_pose, K, RT, image_width=512, image_height=384):
    """
    将mesh投影到图像平面
    
    mesh: Z-up坐标系的mesh
    world_pose: 4x4 model-to-world变换矩阵
    K: 3x3 相机内参
    RT: 4x4 camera-to-world变换矩阵
    """
    # 获取mesh顶点
    vertices = np.array(mesh.vertices)  # (N, 3)
    print(f"Mesh顶点数: {len(vertices)}")
    print(f"Mesh顶点范围(原始): x=[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}], "
          f"y=[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}], "
          f"z=[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")
    
    # decoder mesh范围是(-0.5, 0.5)，而ultrashape mesh范围是(-1, 1)
    # 需要将decoder mesh缩放2倍以匹配原始world_pose
    MESH_SCALE = 2.0
    vertices = vertices * MESH_SCALE
    print(f"Mesh顶点范围(缩放后): x=[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}], "
          f"y=[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}], "
          f"z=[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")
    
    # 1. 应用world_pose: P_world = world_pose @ P_local
    vertices_homo = np.hstack([vertices, np.ones((len(vertices), 1))])  # (N, 4)
    vertices_world = (world_pose @ vertices_homo.T).T[:, :3]  # (N, 3)
    print(f"世界坐标范围: x=[{vertices_world[:,0].min():.3f}, {vertices_world[:,0].max():.3f}], "
          f"y=[{vertices_world[:,1].min():.3f}, {vertices_world[:,1].max():.3f}], "
          f"z=[{vertices_world[:,2].min():.3f}, {vertices_world[:,2].max():.3f}]")
    
    # 2. 世界坐标到相机坐标: P_cam = RT^{-1} @ P_world
    # RT是camera-to-world，所以需要求逆
    RT_inv = np.linalg.inv(RT)
    vertices_world_homo = np.hstack([vertices_world, np.ones((len(vertices_world), 1))])
    vertices_cam = (RT_inv @ vertices_world_homo.T).T[:, :3]  # (N, 3)
    print(f"相机坐标范围: x=[{vertices_cam[:,0].min():.3f}, {vertices_cam[:,0].max():.3f}], "
          f"y=[{vertices_cam[:,1].min():.3f}, {vertices_cam[:,1].max():.3f}], "
          f"z=[{vertices_cam[:,2].min():.3f}, {vertices_cam[:,2].max():.3f}]")
    
    # 3. 投影到图像平面: p = K @ P_cam
    # 只保留z>0的点（在相机前方）
    valid_mask = vertices_cam[:, 2] > 0
    vertices_cam_valid = vertices_cam[valid_mask]
    
    # 投影
    projected = (K @ vertices_cam_valid.T).T  # (N, 3)
    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]
    
    # 过滤在图像范围内的点
    in_image = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    u_valid = u[in_image]
    v_valid = v[in_image]
    
    print(f"有效投影点数: {len(u_valid)} / {len(vertices)}")
    
    return u_valid, v_valid

def load_mask(scene_id, object_id, frame_id):
    """加载mask图像"""
    mask_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/mask/{frame_id}.png"
    if os.path.exists(mask_path):
        print(f"加载mask: {mask_path}")
        return Image.open(mask_path)
    return None

def visualize(rgb_image, u, v, mask_image=None, output_path="verify_pose_result.png"):
    """可视化投影结果"""
    fig, axes = plt.subplots(1, 2 if mask_image else 1, figsize=(20 if mask_image else 12, 9))
    
    if mask_image:
        ax1, ax2 = axes
    else:
        ax1 = axes
    
    # 左图：投影结果
    ax1.imshow(rgb_image)
    ax1.scatter(u, v, c='lime', s=1, alpha=0.5)
    ax1.set_title(f"Mesh Projection (points: {len(u)})")
    ax1.set_xlim(0, rgb_image.width)
    ax1.set_ylim(rgb_image.height, 0)
    
    # 右图：mask叠加
    if mask_image:
        ax2.imshow(rgb_image)
        mask_array = np.array(mask_image.convert('L'))
        ax2.imshow(mask_array, alpha=0.5, cmap='Reds')
        ax2.scatter(u, v, c='lime', s=1, alpha=0.5)
        ax2.set_title("Projection + Mask")
        ax2.set_xlim(0, rgb_image.width)
        ax2.set_ylim(rgb_image.height, 0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"结果保存到: {output_path}")
    plt.close()

def main():
    print("=" * 60)
    print("验证手动标注的Pose")
    print("=" * 60)
    
    # 先加载图像获取尺寸
    rgb_image = load_rgb_image(SCENE_ID, OBJECT_ID, FRAME_ID)
    print(f"\n图像尺寸: {rgb_image.size}")
    
    # 加载相机参数（根据图像尺寸缩放内参）
    K, RT = load_camera_params(SCENE_ID, OBJECT_ID, FRAME_ID, rgb_image.width, rgb_image.height)
    print(f"\n相机外参 RT (camera-to-world):\n{RT}")
    
    mesh = decode_mesh_via_service(SCENE_ID, OBJECT_ID)
    world_pose = load_pose(SCENE_ID, OBJECT_ID, FRAME_ID)
    print(f"\nWorld Pose:\n{world_pose}")
    
    mask_image = load_mask(SCENE_ID, OBJECT_ID, FRAME_ID)
    
    # 投影
    print("\n" + "=" * 60)
    print("投影计算")
    print("=" * 60)
    u, v = project_mesh_to_image(mesh, world_pose, K, RT, 
                                  image_width=rgb_image.width, 
                                  image_height=rgb_image.height)
    
    # 可视化
    output_path = f"/root/csz/yingbo/sam-3d-objects/pose-annotation-tool/verify_pose_{SCENE_ID}_{OBJECT_ID[:8]}_{FRAME_ID}.png"
    visualize(rgb_image, u, v, mask_image, output_path)

if __name__ == "__main__":
    main()
