#!/usr/bin/env python3
"""
Nvdiffrast 渲染服务 - 使用 GPU 光栅化渲染 mesh

提供高效的 mesh 渲染功能，支持：
- 带颜色/纹理的 mesh 渲染
- 实时渲染（保持 context 常驻）
- 返回 RGBA PNG 图像
"""

import io
import os
import numpy as np
import torch
import trimesh
from typing import Dict, Optional, Tuple
from PIL import Image

import nvdiffrast.torch as dr


class MeshRenderer:
    """Nvdiffrast mesh 渲染器"""
    
    def __init__(self, device: str = 'cuda:0'):
        """
        初始化渲染器
        
        Args:
            device: CUDA 设备
        """
        self.device = torch.device(device)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self._mesh_cache: Dict[str, dict] = {}  # mesh 缓存
        print(f"[MeshRenderer] 初始化完成，设备: {device}")
    
    def _load_mesh(self, mesh_path: str) -> dict:
        """
        加载并缓存 mesh
        
        Args:
            mesh_path: mesh 文件路径 (GLB/OBJ)
            
        Returns:
            包含 vertices, faces, vertex_colors 的字典
        """
        if mesh_path in self._mesh_cache:
            return self._mesh_cache[mesh_path]
        
        # 加载 mesh
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # 获取顶点和面
        vertices = torch.tensor(mesh.vertices, device=self.device, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, device=self.device, dtype=torch.int32)
        
        # 获取顶点颜色
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vertex_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        else:
            # 默认灰色
            vertex_colors = np.ones((len(mesh.vertices), 3), dtype=np.float32) * 0.7
        
        vertex_colors = torch.tensor(vertex_colors, device=self.device, dtype=torch.float32)
        
        mesh_data = {
            'vertices': vertices,
            'faces': faces,
            'vertex_colors': vertex_colors,
        }
        
        # 缓存
        self._mesh_cache[mesh_path] = mesh_data
        print(f"[MeshRenderer] 加载 mesh: {mesh_path}, 顶点数: {len(vertices)}, 面数: {len(faces)}")
        
        return mesh_data
    
    def clear_cache(self, mesh_path: Optional[str] = None):
        """清除 mesh 缓存"""
        if mesh_path:
            self._mesh_cache.pop(mesh_path, None)
        else:
            self._mesh_cache.clear()
    
    def render(
        self,
        mesh_path: str,
        pose: np.ndarray,
        intrinsics: dict,
        extrinsics: np.ndarray,
        image_size: Tuple[int, int],
        near: float = 0.01,
        far: float = 100.0,
    ) -> np.ndarray:
        """
        渲染 mesh 到图像
        
        Args:
            mesh_path: mesh 文件路径
            pose: 4x4 object-to-world 变换矩阵
            intrinsics: 相机内参 {'fx', 'fy', 'cx', 'cy'}
            extrinsics: 4x4 camera-to-world 变换矩阵
            image_size: (H, W) 输出图像尺寸
            near: 近裁剪面
            far: 远裁剪面
            
        Returns:
            RGBA 图像 numpy 数组 (H, W, 4), uint8
        """
        H, W = image_size
        
        # 加载 mesh
        mesh_data = self._load_mesh(mesh_path)
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        vertex_colors = mesh_data['vertex_colors']
        
        # Mesh 缩放系数（与前端凸包渲染保持一致）
        MESH_SCALE = 2.0
        vertices = vertices * MESH_SCALE
        
        # 转换为 tensor
        pose_t = torch.tensor(pose, device=self.device, dtype=torch.float32)
        extrinsics_t = torch.tensor(extrinsics, device=self.device, dtype=torch.float32)
        
        # 1. 应用 object pose 变换 (object -> world)
        vertices_homo = torch.cat([vertices, torch.ones(len(vertices), 1, device=self.device)], dim=1)
        vertices_world = (pose_t @ vertices_homo.T).T[:, :3]
        
        # 2. World -> Camera 变换
        # extrinsics 是 camera-to-world，需要求逆得到 world-to-camera
        w2c = torch.inverse(extrinsics_t)
        vertices_homo_world = torch.cat([vertices_world, torch.ones(len(vertices_world), 1, device=self.device)], dim=1)
        vertices_cam = (w2c @ vertices_homo_world.T).T[:, :3]
        
        # 3. Camera -> Clip 变换 (使用 OpenGL 投影矩阵)
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        
        # 构建 OpenGL 投影矩阵
        # nvdiffrast 使用 OpenGL 坐标系: X右, Y上, Z向外(相机看向-Z)
        # 需要翻转 Y 和 Z
        proj = torch.zeros((4, 4), device=self.device, dtype=torch.float32)
        proj[0, 0] = 2 * fx / W
        proj[1, 1] = -2 * fy / H  # 翻转 Y
        proj[0, 2] = 2 * cx / W - 1
        proj[1, 2] = -(2 * cy / H - 1)  # 翻转 Y
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = -1
        
        # OpenCV 相机坐标系 -> OpenGL 相机坐标系
        # OpenCV: X右, Y下, Z前
        # OpenGL: X右, Y上, Z后
        vertices_gl = vertices_cam.clone()
        vertices_gl[:, 1] = -vertices_cam[:, 1]  # Y 翻转
        vertices_gl[:, 2] = -vertices_cam[:, 2]  # Z 翻转
        
        vertices_homo_cam = torch.cat([vertices_gl, torch.ones(len(vertices_gl), 1, device=self.device)], dim=1)
        vertices_clip = (proj @ vertices_homo_cam.T).T
        
        # 4. 光栅化
        # nvdiffrast 需要 (B, V, 4) 形状，且必须是 contiguous
        vertices_clip = vertices_clip[None, :, :].contiguous()  # (1, V, 4)
        faces = faces.contiguous()
        
        rast, _ = dr.rasterize(self.glctx, vertices_clip, faces, resolution=[H, W])
        
        # 5. 插值顶点颜色
        vertex_colors_batch = vertex_colors[None, :, :]  # (1, V, 3)
        color, _ = dr.interpolate(vertex_colors_batch, rast, faces)
        
        # 6. 抗锯齿
        color = dr.antialias(color, rast, vertices_clip, faces)
        
        # 7. 生成 alpha 通道
        alpha = (rast[..., 3:4] > 0).float()
        
        # 8. 合并 RGBA
        rgba = torch.cat([color, alpha], dim=-1)  # (1, H, W, 4)
        
        # 转换为 numpy uint8
        rgba_np = (rgba[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        
        return rgba_np
    
    def render_to_png(
        self,
        mesh_path: str,
        pose: np.ndarray,
        intrinsics: dict,
        extrinsics: np.ndarray,
        image_size: Tuple[int, int],
    ) -> bytes:
        """
        渲染 mesh 并返回 PNG 字节
        
        Returns:
            PNG 图像字节数据
        """
        rgba = self.render(mesh_path, pose, intrinsics, extrinsics, image_size)
        
        # 转换为 PNG
        img = Image.fromarray(rgba, 'RGBA')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()


# 全局渲染器实例
_global_renderer: Optional[MeshRenderer] = None


def get_renderer(device: str = 'cuda:0') -> MeshRenderer:
    """获取全局渲染器实例（单例模式）"""
    global _global_renderer
    if _global_renderer is None:
        _global_renderer = MeshRenderer(device=device)
    return _global_renderer


def render_mesh(
    mesh_path: str,
    pose: np.ndarray,
    intrinsics: dict,
    extrinsics: np.ndarray,
    image_size: Tuple[int, int],
) -> bytes:
    """
    渲染 mesh 的便捷函数
    
    Args:
        mesh_path: mesh 文件路径
        pose: 4x4 object-to-world 变换矩阵
        intrinsics: 相机内参 {'fx', 'fy', 'cx', 'cy'}
        extrinsics: 4x4 camera-to-world 变换矩阵
        image_size: (H, W) 输出图像尺寸
        
    Returns:
        PNG 图像字节数据
    """
    renderer = get_renderer()
    return renderer.render_to_png(mesh_path, pose, intrinsics, extrinsics, image_size)


if __name__ == '__main__':
    # 测试渲染
    import sys
    
    # 测试参数
    test_mesh = "/root/csz/yingbo/MV-SAM3D/reconstruction_lasa1m/42444908/0a91e3c9-e5fb-4989-bd03-6e397d9607b7/mesh.glb"
    
    if not os.path.exists(test_mesh):
        print(f"测试 mesh 不存在: {test_mesh}")
        sys.exit(1)
    
    # 单位矩阵作为 pose
    pose = np.eye(4, dtype=np.float32)
    
    # 测试相机参数
    intrinsics = {'fx': 500, 'fy': 500, 'cx': 320, 'cy': 240}
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[2, 3] = 3  # 相机在 z=3 处
    
    # 渲染
    renderer = get_renderer()
    png_data = renderer.render_to_png(test_mesh, pose, intrinsics, extrinsics, (480, 640))
    
    # 保存测试结果
    with open('/tmp/render_test.png', 'wb') as f:
        f.write(png_data)
    print(f"测试渲染完成，保存到 /tmp/render_test.png")
