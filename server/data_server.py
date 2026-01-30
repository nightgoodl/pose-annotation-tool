#!/usr/bin/env python3
"""
数据服务器 - 为位姿标注工具提供数据API

基于 COORDINATE_SYSTEM_SUMMARY.md 的坐标系约定:
- 相机坐标系: OpenCV标准, +X右, +Y下, +Z前
- RT矩阵: camera-to-world (P_world = R @ P_cam + t)
- ultrashape输出: Y-up, 需要转换为Z-up后再应用world_pose
"""

import os
import json
import glob
import numpy as np
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import struct

# 数据路径配置
DATA_ROOT = "/root/csz/data_partcrafter"
LASA1M_ROOT = f"{DATA_ROOT}/LASA1M"
STAGE2_ROOT = f"{DATA_ROOT}/LASA1M_SAM_STAGE2_V3_DISTRIBUTED"
ALIGNED_ROOT = f"{DATA_ROOT}/LASA1M_ALIGNED_WORLD_V2.5"

class DataAPIHandler(SimpleHTTPRequestHandler):
    """数据API处理器"""
    
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        # API路由
        if path == '/api/scenes':
            self.handle_list_scenes()
        elif path == '/api/objects':
            scene_id = query.get('scene_id', [None])[0]
            self.handle_list_objects(scene_id)
        elif path == '/api/aligned_objects':
            self.handle_list_aligned_objects()
        elif path == '/api/object_data':
            scene_id = query.get('scene_id', [None])[0]
            object_id = query.get('object_id', [None])[0]
            frame_id = query.get('frame_id', [None])[0]
            self.handle_get_object_data(scene_id, object_id, frame_id)
        elif path == '/api/decode':
            # 代理解码请求
            scene_id = query.get('scene_id', [None])[0]
            object_id = query.get('object_id', [None])[0]
            self.handle_decode_proxy(scene_id, object_id)
        elif path.startswith('/data/'):
            # 静态文件服务
            self.serve_data_file(path[6:])
        elif path.startswith('/cache/'):
            # 代理缓存文件
            self.serve_cache_file(path[7:])
        else:
            # 默认静态文件服务
            super().do_GET()
    
    def send_json(self, data, status=200):
        """发送JSON响应"""
        response = json.dumps(data, ensure_ascii=False)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
    
    def send_binary(self, data, content_type='application/octet-stream'):
        """发送二进制响应"""
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)
    
    def do_OPTIONS(self):
        """处理CORS预检请求"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
    
    def do_POST(self):
        """处理POST请求"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        # 读取请求体
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_json({'error': 'Invalid JSON'}, 400)
            return
        
        if path == '/api/save_pose':
            self.handle_save_pose(data)
        else:
            self.send_json({'error': 'Unknown endpoint'}, 404)
    
    def handle_save_pose(self, data):
        """
        保存对齐后的pose
        
        输入格式:
        {
            "scene_id": "42447226",
            "object_id": "1d4fcd26-1247-4482-897f-abf12714d392",
            "frame_id": "212390324019833",
            "pose": [[...], [...], [...], [...]],  # 4x4矩阵
            "scale": 1.0,
            "error": 0.01,
            "point_pairs": [...]  # 可选，用于调试
        }
        
        """
        scene_id = data.get('scene_id')
        object_id = data.get('object_id')
        frame_id = data.get('frame_id')
        pose = data.get('pose')
        scale = data.get('scale', 1.0)
        error = data.get('error', 0.0)
        
        if not all([scene_id, object_id, frame_id, pose]):
            self.send_json({'error': 'Missing required fields: scene_id, object_id, frame_id, pose'}, 400)
            return
        
        # 保存目录 
        save_dir = f"/root/csz/data_partcrafter/LASA1M_ALIGNED_Manual/{scene_id}/{object_id}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存pose为npy文件
        pose_array = np.array(pose, dtype=np.float64)
        pose_path = f"{save_dir}/world_pose_{frame_id}.npy"
        np.save(pose_path, pose_array)
        print(f"[save_pose] 保存pose到: {pose_path}")
        print(f"[save_pose] Pose矩阵:\n{pose_array}")
        
        # 保存结果元数据
        result = {
            "scene_id": scene_id,
            "object_id": object_id,
            "frame_id": frame_id,
            "success": True,
            "scale": scale,
            "alignment_error": error,
            "source": "manual_annotation"
        }
        result_path = f"{save_dir}/result_{frame_id}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[save_pose] 保存结果到: {result_path}")
        
        self.send_json({
            'success': True,
            'pose_path': pose_path,
            'result_path': result_path
        })
    
    def handle_list_scenes(self):
        """列出所有场景"""
        scenes = []
        for scene_dir in sorted(glob.glob(f"{LASA1M_ROOT}/*")):
            if os.path.isdir(scene_dir):
                scene_id = os.path.basename(scene_dir)
                scenes.append({
                    'scene_id': scene_id,
                    'object_count': len(glob.glob(f"{scene_dir}/*"))
                })
        self.send_json({'scenes': scenes})
    
    def handle_list_objects(self, scene_id):
        """列出场景中的物体"""
        if not scene_id:
            self.send_json({'error': 'scene_id required'}, 400)
            return
        
        objects = []
        scene_path = f"{LASA1M_ROOT}/{scene_id}"
        if os.path.exists(scene_path):
            for obj_dir in sorted(glob.glob(f"{scene_path}/*")):
                if os.path.isdir(obj_dir):
                    object_id = os.path.basename(obj_dir)
                    # 检查是否有对齐结果
                    aligned_path = f"{ALIGNED_ROOT}/{scene_id}/{object_id}"
                    has_alignment = os.path.exists(aligned_path)
                    objects.append({
                        'object_id': object_id,
                        'has_alignment': has_alignment
                    })
        self.send_json({'objects': objects, 'scene_id': scene_id})
    
    def handle_list_aligned_objects(self):
        """列出所有已对齐的物体"""
        aligned_objects = []
        
        for scene_dir in sorted(glob.glob(f"{ALIGNED_ROOT}/*")):
            if not os.path.isdir(scene_dir):
                continue
            scene_id = os.path.basename(scene_dir)
            
            for obj_dir in sorted(glob.glob(f"{scene_dir}/*")):
                if not os.path.isdir(obj_dir):
                    continue
                object_id = os.path.basename(obj_dir)
                
                # 查找world_pose文件
                world_pose_files = glob.glob(f"{obj_dir}/world_pose_*.npy")
                for wp_file in world_pose_files:
                    frame_id = os.path.basename(wp_file).replace('world_pose_', '').replace('.npy', '')
                    
                    # 读取result.json获取IOU
                    result_file = f"{obj_dir}/result_{frame_id}.json"
                    iou = None
                    if os.path.exists(result_file):
                        with open(result_file) as f:
                            result = json.load(f)
                            iou = result.get('iou')
                    
                    aligned_objects.append({
                        'scene_id': scene_id,
                        'object_id': object_id,
                        'frame_id': frame_id,
                        'iou': iou
                    })
        
        # 按IOU降序排列
        aligned_objects.sort(key=lambda x: x.get('iou') or 0, reverse=True)
        self.send_json({'aligned_objects': aligned_objects[:100]})  # 只返回前100个
    
    def handle_get_object_data(self, scene_id, object_id, frame_id):
        """获取物体的完整数据"""
        if not all([scene_id, object_id, frame_id]):
            self.send_json({'error': 'scene_id, object_id, frame_id required'}, 400)
            return
        
        try:
            data = self.load_object_data(scene_id, object_id, frame_id)
            self.send_json(data)
        except Exception as e:
            self.send_json({'error': str(e)}, 500)
    
    def load_object_data(self, scene_id, object_id, frame_id):
        """加载物体数据"""
        # 1. 加载info.json获取相机参数
        info_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/info.json"
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"info.json not found: {info_path}")
        
        with open(info_path) as f:
            info = json.load(f)
        
        # 查找对应frame的相机参数
        frame_info = None
        frame_timestamp = int(frame_id)
        for item in info:
            if item.get('timestamp') == frame_timestamp:
                frame_info = item
                break
        
        if not frame_info:
            # 尝试找最近的frame
            timestamps = [item.get('timestamp', 0) for item in info]
            if timestamps:
                closest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - frame_timestamp))
                frame_info = info[closest_idx]
                print(f"Using closest frame: {frame_info.get('timestamp')} for requested {frame_id}")
        
        if not frame_info:
            raise ValueError(f"Frame {frame_id} not found in info.json")
        
        # 2. 提取相机内参和外参
        K = frame_info.get('gt_depth_K')  # 3x3 内参
        RT = frame_info.get('gt_RT')       # 4x4 camera-to-world
        
        # 3. 加载world_pose
        world_pose_path = f"{ALIGNED_ROOT}/{scene_id}/{object_id}/world_pose_{frame_id}.npy"
        if os.path.exists(world_pose_path):
            world_pose = np.load(world_pose_path).tolist()
        else:
            world_pose = None
        
        # 4. 查找mesh文件和latent文件
        mesh_path = None
        mesh_url = None
        latent_path = None
        
        # 在STAGE2中查找GLB和latent
        stage2_pattern = f"{STAGE2_ROOT}/{scene_id}/*/object_*_{object_id}"
        stage2_dirs = glob.glob(stage2_pattern)
        if stage2_dirs:
            stage2_dir = stage2_dirs[0]
            # 查找GLB文件
            glb_files = glob.glob(f"{stage2_dir}/*.glb")
            if glb_files:
                mesh_path = glb_files[0]
                mesh_url = f"/data/mesh/{scene_id}/{object_id}/{os.path.basename(mesh_path)}"
            # 查找latent文件
            latent_file = f"{stage2_dir}/stage2_slat_latent.pt"
            if os.path.exists(latent_file):
                latent_path = latent_file
        
        # 5. 图像和mask路径 - 使用sam2的mask（与raw_jpg尺寸一致）
        rgb_path = None
        mask_path = None
        actual_frame = frame_id
        image_width = 0
        image_height = 0
        
        rgb_files = sorted(glob.glob(f"{LASA1M_ROOT}/{scene_id}/{object_id}/raw_jpg/*.jpg"))
        if rgb_files:
            rgb_timestamps = [int(os.path.basename(f).replace('.jpg', '')) for f in rgb_files]
            closest_idx = min(range(len(rgb_timestamps)), key=lambda i: abs(rgb_timestamps[i] - frame_timestamp))
            rgb_path = rgb_files[closest_idx]
            actual_frame = os.path.basename(rgb_path).replace('.jpg', '')
            
            # 动态读取实际图像尺寸
            from PIL import Image
            with Image.open(rgb_path) as img:
                image_width, image_height = img.size
            print(f"Image size: {image_width}x{image_height}")
            
            # 使用sam_mask_sam2目录的mask（与raw_jpg尺寸一致）
            mask_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/sam_mask_sam2/{actual_frame}.png"
            # 如果sam2 mask不存在，回退到原始mask
            if not os.path.exists(mask_path):
                mask_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/mask/{actual_frame}.png"
            print(f"Using closest frame: {actual_frame} (requested: {frame_id})")
        
        # 确定mask的子目录
        mask_subdir = 'sam_mask_sam2' if mask_path and 'sam_mask_sam2' in mask_path else 'mask'
        rgb_url = f"/data/image/{scene_id}/{object_id}/raw_jpg/{os.path.basename(rgb_path)}" if rgb_path and os.path.exists(rgb_path) else None
        mask_url = f"/data/image/{scene_id}/{object_id}/{mask_subdir}/{os.path.basename(mask_path)}" if mask_path and os.path.exists(mask_path) else None
        
        # 6. 深度图路径
        depth_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/gt/{frame_id}/depth.png"
        if not os.path.exists(depth_path):
            # 尝试找对应的gt目录
            gt_dirs = sorted(glob.glob(f"{LASA1M_ROOT}/{scene_id}/{object_id}/gt/*"))
            if gt_dirs:
                gt_timestamps = [int(os.path.basename(d)) for d in gt_dirs]
                closest_idx = min(range(len(gt_timestamps)), key=lambda i: abs(gt_timestamps[i] - frame_timestamp))
                actual_gt_frame = os.path.basename(gt_dirs[closest_idx])
                depth_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/gt/{actual_gt_frame}/depth.png"
        
        depth_url = f"/data/depth/{scene_id}/{object_id}/{os.path.basename(os.path.dirname(depth_path))}" if os.path.exists(depth_path) else None
        
        # 7. 加载IOU结果
        result_path = f"{ALIGNED_ROOT}/{scene_id}/{object_id}/result_{frame_id}.json"
        alignment_result = None
        if os.path.exists(result_path):
            with open(result_path) as f:
                alignment_result = json.load(f)
        
        # 内参需要根据图像尺寸调整
        # gt_depth_K 是针对深度图尺寸的，需要根据实际图像尺寸缩放
        # 深度图尺寸：512x384 (宽x高)
        depth_width = 512
        depth_height = 384
        
        K_scaled = None
        if K and image_width > 0 and image_height > 0:
            scale_x = image_width / depth_width
            scale_y = image_height / depth_height
            print(f"Intrinsics scale: x={scale_x}, y={scale_y}")
            K_scaled = [
                [K[0][0] * scale_x, K[0][1], K[0][2] * scale_x],
                [K[1][0], K[1][1] * scale_y, K[1][2] * scale_y],
                [K[2][0], K[2][1], K[2][2]]
            ]
        
        return {
            'scene_id': scene_id,
            'object_id': object_id,
            'frame_id': frame_id,
            'camera_intrinsics': {
                'K': K_scaled,  # 使用缩放后的内参
                'K_original': K,  # 原始内参
                'width': image_width,   # 实际图像宽度
                'height': image_height  # 实际图像高度
            },
            'camera_extrinsics': RT,  # camera-to-world
            'world_pose': world_pose,  # Model-to-World (Y-up转Z-up后)
            'mesh_url': mesh_url,
            'mesh_path': mesh_path,
            'latent_path': latent_path,  # 用于解码服务
            'decoder_url': f"http://localhost:8083/decode?scene_id={scene_id}&object_id={object_id}&texture=true" if latent_path else None,
            'rgb_url': rgb_url,
            'mask_url': mask_url,
            'depth_url': depth_url,
            'alignment_result': alignment_result
        }
    
    def serve_data_file(self, path):
        """服务数据文件"""
        parts = path.split('/')
        print(f"[serve_data_file] path={path}, parts={parts}")
        
        if parts[0] == 'image':
            # /data/image/{scene_id}/{object_id}/{subdir}/{filename}
            try:
                scene_id, object_id, subdir, filename = parts[1], parts[2], parts[3], parts[4]
                file_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/{subdir}/{filename}"
                print(f"[serve_data_file] image file_path={file_path}")
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        data = f.read()
                    content_type = 'image/jpeg' if filename.endswith('.jpg') else 'image/png'
                    self.send_binary(data, content_type)
                else:
                    self.send_error(404, f"File not found: {file_path}")
            except Exception as e:
                print(f"[serve_data_file] error: {e}")
                self.send_error(500, str(e))
        
        elif parts[0] == 'mesh':
            # /data/mesh/{scene_id}/{object_id}/{filename}
            scene_id, object_id, filename = parts[1], parts[2], parts[3]
            # 查找mesh文件
            pattern = f"{STAGE2_ROOT}/{scene_id}/*/object_*_{object_id}/{filename}"
            files = glob.glob(pattern)
            if files:
                with open(files[0], 'rb') as f:
                    data = f.read()
                self.send_binary(data, 'model/gltf-binary')
            else:
                self.send_error(404, f"Mesh not found: {pattern}")
        
        elif parts[0] == 'depth':
            # /data/depth/{scene_id}/{object_id}/{frame_id}
            # 返回深度图的二进制数据 (Float32)
            scene_id, object_id, frame_id = parts[1], parts[2], parts[3]
            depth_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/gt/{frame_id}/depth.png"
            
            if os.path.exists(depth_path):
                # 读取深度图并转换为Float32
                import cv2
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_img is not None:
                    # 假设深度图是16位或32位
                    if depth_img.dtype == np.uint16:
                        depth_float = depth_img.astype(np.float32) / 1000.0  # mm to m
                    else:
                        depth_float = depth_img.astype(np.float32)
                    
                    # 返回元数据和二进制数据
                    response = {
                        'width': depth_float.shape[1],
                        'height': depth_float.shape[0],
                        'data': depth_float.flatten().tolist()
                    }
                    self.send_json(response)
                else:
                    self.send_error(500, "Failed to read depth image")
            else:
                self.send_error(404, f"Depth not found: {depth_path}")
        
        else:
            self.send_error(404, f"Unknown data type: {parts[0]}")
    
    def handle_decode_proxy(self, scene_id, object_id):
        """代理解码请求到mesh_decoder_service"""
        import urllib.request
        import urllib.error
        
        if not scene_id or not object_id:
            self.send_json({'error': 'scene_id and object_id required'}, 400)
            return
        
        try:
            decode_url = f"http://localhost:8083/decode?scene_id={scene_id}&object_id={object_id}&texture=true"
            print(f"[decode_proxy] 请求: {decode_url}")
            
            with urllib.request.urlopen(decode_url, timeout=300) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            if data.get('success') and data.get('mesh_url'):
                # 将mesh_url改为通过本服务器代理
                data['mesh_url'] = data['mesh_url']  # /cache/xxx.glb
                print(f"[decode_proxy] 成功: {data['mesh_url']}")
            
            self.send_json(data)
        except urllib.error.URLError as e:
            print(f"[decode_proxy] 连接失败: {e}")
            self.send_json({'error': f'Decoder service unavailable: {e}'}, 503)
        except Exception as e:
            print(f"[decode_proxy] 错误: {e}")
            self.send_json({'error': str(e)}, 500)
    
    def serve_cache_file(self, filename):
        """代理缓存文件从mesh_decoder_service"""
        import urllib.request
        import urllib.error
        
        try:
            cache_url = f"http://localhost:8083/cache/{filename}"
            print(f"[serve_cache] 请求: {cache_url}")
            
            with urllib.request.urlopen(cache_url, timeout=60) as response:
                data = response.read()
            
            self.send_response(200)
            self.send_header('Content-Type', 'model/gltf-binary')
            self.send_header('Content-Length', len(data))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data)
        except urllib.error.URLError as e:
            print(f"[serve_cache] 连接失败: {e}")
            self.send_error(503, f'Cache service unavailable: {e}')
        except Exception as e:
            print(f"[serve_cache] 错误: {e}")
            self.send_error(500, str(e))


def run_server(port=8080):
    """运行服务器"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, DataAPIHandler)
    print(f"Data server running at http://localhost:{port}")
    print(f"API endpoints:")
    print(f"  GET /api/scenes - List all scenes")
    print(f"  GET /api/objects?scene_id=xxx - List objects in scene")
    print(f"  GET /api/aligned_objects - List aligned objects")
    print(f"  GET /api/object_data?scene_id=xxx&object_id=xxx&frame_id=xxx - Get object data")
    httpd.serve_forever()


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
