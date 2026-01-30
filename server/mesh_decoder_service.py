#!/usr/bin/env python3
"""
Mesh解码服务 - 从latent解码带颜色的mesh

功能：
1. 从stage2_slat_latent.pt解码mesh
2. 使用Gaussian进行texture baking或vertex color
3. 生成带颜色的GLB文件供前端加载
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time

# 项目路径
PROJECT_ROOT = "/root/csz/yingbo/sam-3d-objects"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['LIDRA_SKIP_INIT'] = '1'

import torch
import numpy as np
import trimesh

# 数据路径
DATA_ROOT = "/root/csz/data_partcrafter"
STAGE2_ROOT = f"{DATA_ROOT}/LASA1M_SAM_STAGE2_V3_DISTRIBUTED"
CACHE_DIR = "/tmp/mesh_decoder_cache"

# 全局解码器实例
_decoder = None
_decoder_lock = threading.Lock()


def get_decoder():
    """获取或初始化解码器（懒加载）"""
    global _decoder
    if _decoder is None:
        with _decoder_lock:
            if _decoder is None:
                print("初始化解码器...")
                _decoder = MeshDecoderWithColor()
                print("解码器初始化完成")
    return _decoder


class MeshDecoderWithColor:
    """带颜色的Mesh解码器"""
    
    def __init__(self, checkpoint_dir: str = f"{PROJECT_ROOT}/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.mesh_decoder = None
        self.gs_decoder = None
        self._mesh_initialized = False
        self._gs_initialized = False
        
        # 确保缓存目录存在
        os.makedirs(CACHE_DIR, exist_ok=True)
    
    def _init_mesh_decoder(self):
        """初始化mesh解码器"""
        if self._mesh_initialized:
            return
        
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        
        decoder_config_path = os.path.join(self.checkpoint_dir, "slat_decoder_mesh.yaml")
        decoder_ckpt_path = os.path.join(self.checkpoint_dir, "slat_decoder_mesh.ckpt")
        
        config = OmegaConf.load(decoder_config_path)
        self.mesh_decoder = instantiate(config)
        
        import warnings
        warnings.filterwarnings('ignore')
        checkpoint = torch.load(decoder_ckpt_path, map_location='cpu', weights_only=False)
        
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items()}
        
        self.mesh_decoder.load_state_dict(new_state_dict, strict=False)
        self.mesh_decoder = self.mesh_decoder.cuda().eval()
        self._mesh_initialized = True
        print("Mesh解码器已初始化")
    
    def _init_gs_decoder(self):
        """初始化Gaussian解码器"""
        if self._gs_initialized:
            return
        
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        
        decoder_config_path = os.path.join(self.checkpoint_dir, "slat_decoder_gs.yaml")
        decoder_ckpt_path = os.path.join(self.checkpoint_dir, "slat_decoder_gs.ckpt")
        
        config = OmegaConf.load(decoder_config_path)
        self.gs_decoder = instantiate(config)
        
        import warnings
        warnings.filterwarnings('ignore')
        checkpoint = torch.load(decoder_ckpt_path, map_location='cpu', weights_only=False)
        
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items()}
        
        self.gs_decoder.load_state_dict(new_state_dict, strict=False)
        self.gs_decoder = self.gs_decoder.cuda().eval()
        self._gs_initialized = True
        print("Gaussian解码器已初始化")
    
    def decode_with_color(self, latent_path: str, output_path: str, 
                          use_texture_baking: bool = True,
                          texture_size: int = 1024) -> str:
        """
        解码latent为带颜色的mesh
        
        Args:
            latent_path: latent文件路径
            output_path: 输出GLB路径
            use_texture_baking: 是否使用纹理烘焙（否则使用顶点颜色）
            texture_size: 纹理大小
        
        Returns:
            output_path: 输出文件路径
        """
        self._init_mesh_decoder()
        self._init_gs_decoder()
        
        from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor
        from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils
        
        import warnings
        warnings.filterwarnings('ignore')
        
        # 加载latent
        latent_data = torch.load(latent_path, map_location='cuda', weights_only=False)
        
        slat = SparseTensor(
            feats=latent_data['feats'].cuda(),
            coords=latent_data['coords'].int().cuda(),
        )
        
        # 解码mesh
        with torch.no_grad():
            mesh_outputs = self.mesh_decoder.forward(slat)
            gs_outputs = self.gs_decoder.forward(slat)
        
        # 提取mesh和gaussian
        mesh_data = mesh_outputs[0] if isinstance(mesh_outputs, list) else mesh_outputs
        gaussian = gs_outputs[0] if isinstance(gs_outputs, list) else gs_outputs
        
        # 使用postprocessing_utils生成带颜色的GLB
        try:
            glb_mesh = postprocessing_utils.to_glb(
                gaussian,
                mesh_data,
                simplify=0.95,
                texture_size=texture_size,
                verbose=False,
                with_mesh_postprocess=True,
                with_texture_baking=use_texture_baking,
                use_vertex_color=not use_texture_baking,
                rendering_engine="nvdiffrast",
            )
            
            # 保存
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            glb_mesh.export(output_path)
            print(f"Mesh已保存: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"texture baking失败，尝试使用顶点颜色: {e}")
            # 回退到顶点颜色
            return self._decode_with_vertex_color(mesh_data, output_path)
    
    def _decode_with_vertex_color(self, mesh_data, output_path: str) -> str:
        """使用顶点颜色生成mesh"""
        vertices = mesh_data.vertices.float().cpu().numpy()
        faces = mesh_data.faces.cpu().numpy()
        
        # 提取顶点颜色
        if hasattr(mesh_data, 'vertex_attrs') and mesh_data.vertex_attrs is not None:
            vert_colors = mesh_data.vertex_attrs[:, :3].cpu().numpy()
            # 确保颜色在0-255范围
            if vert_colors.max() <= 1.0:
                vert_colors = (vert_colors * 255).astype(np.uint8)
            # 添加alpha通道
            vert_colors = np.concatenate([vert_colors, np.full((len(vert_colors), 1), 255, dtype=np.uint8)], axis=1)
        else:
            # 默认灰色
            vert_colors = np.full((len(vertices), 4), [180, 180, 180, 255], dtype=np.uint8)
        
        # 旋转mesh (from z-up to y-up for GLB standard)
        vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = vert_colors
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
        print(f"Mesh (vertex color)已保存: {output_path}")
        
        return output_path
    
    def get_cache_path(self, latent_path: str, use_texture_baking: bool) -> str:
        """获取缓存路径"""
        # 使用latent路径的hash作为缓存key
        key = f"{latent_path}_{use_texture_baking}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:16]
        return os.path.join(CACHE_DIR, f"{hash_key}.glb")
    
    def decode_cached(self, latent_path: str, use_texture_baking: bool = True) -> str:
        """解码并缓存结果"""
        cache_path = self.get_cache_path(latent_path, use_texture_baking)
        
        if os.path.exists(cache_path):
            print(f"使用缓存: {cache_path}")
            return cache_path
        
        return self.decode_with_color(latent_path, cache_path, use_texture_baking)


class DecoderServiceHandler(BaseHTTPRequestHandler):
    """解码服务HTTP处理器"""
    
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        if path == '/decode':
            self.handle_decode(query)
        elif path == '/health':
            self.send_json({'status': 'ok'})
        elif path.startswith('/cache/'):
            self.serve_cache_file(path[7:])
        else:
            self.send_error(404, 'Not Found')
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
    
    def send_json(self, data, status=200):
        response = json.dumps(data, ensure_ascii=False)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
    
    def handle_decode(self, query):
        """处理解码请求"""
        scene_id = query.get('scene_id', [None])[0]
        object_id = query.get('object_id', [None])[0]
        frame_id = query.get('frame_id', [None])[0]
        use_texture = query.get('texture', ['true'])[0].lower() == 'true'
        
        if not all([scene_id, object_id]):
            self.send_json({'error': 'scene_id and object_id required'}, 400)
            return
        
        # 查找latent文件
        if frame_id:
            latent_pattern = f"{STAGE2_ROOT}/{scene_id}/{frame_id}/object_*_{object_id}/stage2_slat_latent.pt"
        else:
            latent_pattern = f"{STAGE2_ROOT}/{scene_id}/*/object_*_{object_id}/stage2_slat_latent.pt"
        
        import glob
        latent_files = glob.glob(latent_pattern)
        
        if not latent_files:
            self.send_json({'error': 'Latent file not found'}, 404)
            return
        
        latent_path = latent_files[0]
        
        try:
            decoder = get_decoder()
            output_path = decoder.decode_cached(latent_path, use_texture)
            
            # 返回缓存文件的URL
            cache_filename = os.path.basename(output_path)
            self.send_json({
                'success': True,
                'mesh_url': f'/cache/{cache_filename}',
                'latent_path': latent_path
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({'error': str(e)}, 500)
    
    def serve_cache_file(self, filename):
        """提供缓存文件"""
        filepath = os.path.join(CACHE_DIR, filename)
        
        if not os.path.exists(filepath):
            self.send_error(404, 'File not found')
            return
        
        with open(filepath, 'rb') as f:
            data = f.read()
        
        self.send_response(200)
        self.send_header('Content-Type', 'model/gltf-binary')
        self.send_header('Content-Length', len(data))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)


def run_server(port=8083):
    """运行解码服务"""
    server = HTTPServer(('0.0.0.0', port), DecoderServiceHandler)
    print(f"Mesh解码服务运行在 http://localhost:{port}")
    print(f"缓存目录: {CACHE_DIR}")
    server.serve_forever()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Mesh解码服务')
    parser.add_argument('--port', type=int, default=8083, help='服务端口')
    args = parser.parse_args()
    
    run_server(args.port)
