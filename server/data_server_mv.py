#!/usr/bin/env python3
"""
多视角数据服务器 - 为多视角位姿标注工具提供数据API

数据来源:
- 多视角重建mesh: /root/csz/yingbo/MV-SAM3D/reconstruction_lasa1m/{scene_id}/{object_id}/mesh.glb
- 元数据(LASA1M): /root/csz/data_partcrafter/LASA1M/{scene_id}/{object_id}/info.json
- 图像: /root/csz/data_partcrafter/LASA1M/{scene_id}/{object_id}/raw_jpg/
"""

import os
import json
import glob
import numpy as np
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# 数据路径配置
DATA_ROOT = "/root/csz/data_partcrafter"
LASA1M_ROOT = f"{DATA_ROOT}/LASA1M"
MV_RECON_ROOT = "/root/csz/yingbo/MV-SAM3D/reconstruction_lasa1m"
MV_ALIGNED_ROOT = f"{DATA_ROOT}/LASA1M_ALIGNED_MV"  # 多视角对齐结果保存目录

# 全局缓存
_mv_objects_cache = None
_mv_objects_cache_time = 0
CACHE_TTL = 300  # 缓存有效期5分钟


class MVDataAPIHandler(SimpleHTTPRequestHandler):
    """多视角数据API处理器"""
    
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        # API路由
        if path == '/api/refresh_cache':
            # 增量刷新缓存 - 只更新有变化的对象
            global _mv_objects_cache, _mv_objects_cache_time
            scene_id = query.get('scene_id', [None])[0]
            object_id = query.get('object_id', [None])[0]
            
            if scene_id and object_id and _mv_objects_cache:
                # 增量更新单个对象
                self._update_single_object_cache(scene_id, object_id)
                self.send_json({'success': True, 'message': 'Object cache updated'})
            else:
                # 完全刷新缓存
                _mv_objects_cache = None
                _mv_objects_cache_time = 0
                self.send_json({'success': True, 'message': 'Cache refreshed'})
        
        elif path == '/api/mv_objects':
            page = int(query.get('page', [1])[0])
            page_size = int(query.get('page_size', [50])[0])
            scene_filter = query.get('scene', [None])[0]
            self.handle_list_mv_objects(page, page_size, scene_filter)
        elif path == '/api/mv_object_data':
            scene_id = query.get('scene_id', [None])[0]
            object_id = query.get('object_id', [None])[0]
            num_frames = int(query.get('num_frames', [8])[0])
            self.handle_get_mv_object_data(scene_id, object_id, num_frames)
        elif path.startswith('/data/'):
            # 静态文件服务
            self.serve_data_file(path[6:])
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
        
        if path == '/api/save_mv_pose':
            self.handle_save_mv_pose(data)
        else:
            self.send_json({'error': 'Unknown endpoint'}, 404)
    
    def handle_save_mv_pose(self, data):
        """
        保存多视角对齐后的pose
        
        输入格式:
        {
            "scene_id": "42447226",
            "object_id": "1d4fcd26-1247-4482-897f-abf12714d392",
            "pose": [[...], [...], [...], [...]],  # 4x4矩阵
            "scale": 1.0,
            "error": 0.01,
            "point_pairs": [...],  # 所有帧的点对
            "category": "valid" | "fixed" | "invalid"  # 分类状态
        }
        """
        scene_id = data.get('scene_id')
        object_id = data.get('object_id')
        pose = data.get('pose')
        scale = data.get('scale', 1.0)
        error = data.get('error', 0.0)
        point_pairs = data.get('point_pairs', [])
        category = data.get('category', 'valid')
        
        if not all([scene_id, object_id, pose]):
            self.send_json({'error': 'Missing required fields: scene_id, object_id, pose'}, 400)
            return
        
        # 保存目录
        save_dir = f"{MV_ALIGNED_ROOT}/{scene_id}/{object_id}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存pose为npy文件
        pose_array = np.array(pose, dtype=np.float64)
        pose_path = f"{save_dir}/world_pose.npy"
        np.save(pose_path, pose_array)
        print(f"[save_mv_pose] 保存pose到: {pose_path}")
        print(f"[save_mv_pose] Pose矩阵:\n{pose_array}")
        
        # 保存结果元数据
        result = {
            "scene_id": scene_id,
            "object_id": object_id,
            "success": True,
            "scale": scale,
            "alignment_error": error,
            "source": "manual_mv_annotation",
            "num_point_pairs": len(point_pairs),
            "category": category
        }
        result_path = f"{save_dir}/result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[save_mv_pose] 保存结果到: {result_path}")
        
        # 保存点对详情
        if point_pairs:
            pairs_path = f"{save_dir}/point_pairs.json"
            with open(pairs_path, 'w') as f:
                json.dump(point_pairs, f, indent=2)
        
        self.send_json({
            'success': True,
            'pose_path': pose_path,
            'result_path': result_path
        })
    
    def handle_list_mv_objects(self, page=1, page_size=50, scene_filter=None):
        """列出多视角重建物体（带分页和缓存）"""
        global _mv_objects_cache, _mv_objects_cache_time
        
        # 检查缓存是否有效
        current_time = time.time()
        if _mv_objects_cache is None or (current_time - _mv_objects_cache_time) > CACHE_TTL:
            print("[handle_list_mv_objects] 重建缓存...")
            start_time = time.time()
            _mv_objects_cache = self._build_mv_objects_cache()
            _mv_objects_cache_time = current_time
            print(f"[handle_list_mv_objects] 缓存构建完成，耗时 {time.time() - start_time:.2f}s，共 {len(_mv_objects_cache)} 个物体")
        
        # 应用场景过滤
        if scene_filter:
            filtered = [o for o in _mv_objects_cache if o['scene_id'] == scene_filter]
        else:
            filtered = _mv_objects_cache
        
        total = len(filtered)
        
        # 分页
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_data = filtered[start_idx:end_idx]
        
        # 获取所有场景ID列表（用于前端筛选）
        all_scenes = sorted(set(o['scene_id'] for o in _mv_objects_cache))
        
        self.send_json({
            'mv_objects': page_data,
            'total': total,
            'page': page,
            'page_size': page_size,
            'total_pages': (total + page_size - 1) // page_size,
            'scenes': all_scenes[:100]  # 只返回前100个场景用于筛选
        })
    
    def _update_single_object_cache(self, scene_id, object_id):
        """增量更新单个对象的缓存状态"""
        global _mv_objects_cache
        if not _mv_objects_cache:
            return
        
        # 读取该对象的最新状态
        aligned_path = f"{MV_ALIGNED_ROOT}/{scene_id}/{object_id}"
        has_alignment = os.path.exists(f"{aligned_path}/world_pose.npy")
        
        category = None
        result_path = f"{aligned_path}/result.json"
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r') as f:
                    result_data = json.load(f)
                    category = result_data.get('category')
            except:
                pass
        
        # 更新缓存中的对象
        for obj in _mv_objects_cache:
            if obj['scene_id'] == scene_id and obj['object_id'] == object_id:
                obj['has_alignment'] = has_alignment
                obj['category'] = category
                print(f"[Cache] 更新对象 {scene_id}/{object_id}: category={category}, has_alignment={has_alignment}")
                break
    
    def _build_mv_objects_cache(self):
        """构建物体列表缓存（优化：不计算帧数）"""
        mv_objects = []
        
        # 遍历多视角重建目录
        for scene_dir in sorted(glob.glob(f"{MV_RECON_ROOT}/*")):
            if not os.path.isdir(scene_dir):
                continue
            scene_id = os.path.basename(scene_dir)
            
            for obj_dir in sorted(glob.glob(f"{scene_dir}/*")):
                if not os.path.isdir(obj_dir):
                    continue
                object_id = os.path.basename(obj_dir)
                
                # 检查是否有mesh文件
                mesh_path = f"{obj_dir}/mesh.glb"
                if not os.path.exists(mesh_path):
                    continue
                
                # 检查LASA1M中是否有对应数据（只检查目录存在）
                lasa_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}"
                if not os.path.exists(lasa_path):
                    continue
                
                # 检查是否已有对齐结果和分类状态
                aligned_path = f"{MV_ALIGNED_ROOT}/{scene_id}/{object_id}"
                has_alignment = os.path.exists(f"{aligned_path}/world_pose.npy")
                
                # 读取分类状态
                category = None
                result_path = f"{aligned_path}/result.json"
                if os.path.exists(result_path):
                    try:
                        with open(result_path, 'r') as f:
                            result_data = json.load(f)
                            category = result_data.get('category')
                    except:
                        pass
                
                # 不再计算帧数（太慢），加载时再获取
                mv_objects.append({
                    'scene_id': scene_id,
                    'object_id': object_id,
                    'num_frames': -1,  # 延迟加载
                    'has_alignment': has_alignment,
                    'category': category
                })
        
        # 按scene_id排序
        mv_objects.sort(key=lambda x: (x['scene_id'], x['object_id']))
        return mv_objects
    
    def handle_get_mv_object_data(self, scene_id, object_id, num_frames=8):
        """获取多视角物体的完整数据"""
        if not all([scene_id, object_id]):
            self.send_json({'error': 'scene_id, object_id required'}, 400)
            return
        
        try:
            data = self.load_mv_object_data(scene_id, object_id, num_frames)
            self.send_json(data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({'error': str(e)}, 500)
    
    def load_mv_object_data(self, scene_id, object_id, num_frames=8):
        """加载多视角物体数据"""
        # 1. 检查mesh文件
        mesh_path = f"{MV_RECON_ROOT}/{scene_id}/{object_id}/mesh.glb"
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        mesh_url = f"/data/mv_mesh/{scene_id}/{object_id}/mesh.glb"
        
        # 2. 加载info.json获取所有帧的相机参数
        info_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/info.json"
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"info.json not found: {info_path}")
        
        with open(info_path) as f:
            info = json.load(f)
        
        # 3. 获取所有RGB文件并按时间戳排序
        rgb_dir = f"{LASA1M_ROOT}/{scene_id}/{object_id}/raw_jpg"
        rgb_files = sorted(glob.glob(f"{rgb_dir}/*.jpg"))
        if not rgb_files:
            raise FileNotFoundError(f"No RGB files found in {rgb_dir}")
        
        # 4. 两阶段帧选择：先均匀采样12帧，再按mask比例选择num_frames帧
        total_frames = len(rgb_files)
        SAMPLE_POOL_SIZE = 12  # 均匀采样的候选池大小
        
        if total_frames <= num_frames:
            # 帧数不足，全部使用
            selected_indices = list(range(total_frames))
        elif total_frames <= SAMPLE_POOL_SIZE:
            # 帧数少于候选池大小，全部作为候选
            selected_indices = list(range(total_frames))
        else:
            # 均匀采样12帧作为候选池
            step = total_frames / SAMPLE_POOL_SIZE
            selected_indices = [int(i * step) for i in range(SAMPLE_POOL_SIZE)]
        
        print(f"[load_mv_object_data] Sampled {len(selected_indices)} candidate frames from {total_frames} total")
        
        # 5. 构建帧数据
        frames = []
        
        # 建立timestamp到info的映射
        timestamp_to_info = {}
        for item in info:
            ts = item.get('timestamp')
            if ts:
                timestamp_to_info[ts] = item
        
        # 深度图尺寸
        depth_width = 512
        depth_height = 384
        
        for idx in selected_indices:
            rgb_path = rgb_files[idx]
            frame_id = os.path.basename(rgb_path).replace('.jpg', '')
            frame_timestamp = int(frame_id)
            
            # 获取相机参数
            frame_info = timestamp_to_info.get(frame_timestamp)
            if not frame_info:
                # 找最近的
                timestamps = list(timestamp_to_info.keys())
                if timestamps:
                    closest_ts = min(timestamps, key=lambda t: abs(t - frame_timestamp))
                    frame_info = timestamp_to_info[closest_ts]
            
            if not frame_info:
                continue
            
            # 获取图像尺寸
            from PIL import Image
            with Image.open(rgb_path) as img:
                image_width, image_height = img.size
            
            # 提取相机参数
            K = frame_info.get('gt_depth_K')  # 3x3 内参
            RT = frame_info.get('gt_RT')       # 4x4 camera-to-world
            
            # 内参需要根据图像尺寸调整
            K_scaled = None
            if K and image_width > 0 and image_height > 0:
                scale_x = image_width / depth_width
                scale_y = image_height / depth_height
                K_scaled = [
                    [K[0][0] * scale_x, K[0][1], K[0][2] * scale_x],
                    [K[1][0], K[1][1] * scale_y, K[1][2] * scale_y],
                    [K[2][0], K[2][1], K[2][2]]
                ]
            
            # 检查mask - 直接使用点云直投的mask文件夹
            mask_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/mask/{frame_id}.png"
            mask_subdir = 'mask'
            
            # 计算mask比例 (mask像素数 / 总像素数)
            mask_ratio = 0.0
            if os.path.exists(mask_path):
                try:
                    with Image.open(mask_path) as mask_img:
                        mask_array = np.array(mask_img)
                        if len(mask_array.shape) == 3:
                            mask_array = mask_array[:, :, 0]  # 取第一通道
                        mask_pixels = np.sum(mask_array > 128)  # 非零像素
                        total_pixels = mask_array.shape[0] * mask_array.shape[1]
                        mask_ratio = mask_pixels / total_pixels if total_pixels > 0 else 0.0
                except Exception as e:
                    print(f"[load_mv_object_data] Error calculating mask ratio: {e}")
            
            # 检查深度图
            depth_url = None
            gt_dirs = sorted(glob.glob(f"{LASA1M_ROOT}/{scene_id}/{object_id}/gt/*"))
            if gt_dirs:
                gt_timestamps = [int(os.path.basename(d)) for d in gt_dirs]
                closest_idx = min(range(len(gt_timestamps)), key=lambda i: abs(gt_timestamps[i] - frame_timestamp))
                actual_gt_frame = os.path.basename(gt_dirs[closest_idx])
                depth_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/gt/{actual_gt_frame}/depth.png"
                if os.path.exists(depth_path):
                    depth_url = f"/data/depth/{scene_id}/{object_id}/{actual_gt_frame}"
            
            frames.append({
                'frame_id': frame_id,
                'frame_index': idx,
                'rgb_url': f"/data/image/{scene_id}/{object_id}/raw_jpg/{frame_id}.jpg",
                'mask_url': f"/data/image/{scene_id}/{object_id}/{mask_subdir}/{frame_id}.png" if os.path.exists(mask_path) else None,
                'depth_url': depth_url,
                'mask_ratio': mask_ratio,  # mask占比，用于排序
                'camera_intrinsics': {
                    'K': K_scaled,
                    'K_original': K,
                    'width': image_width,
                    'height': image_height
                },
                'camera_extrinsics': RT  # camera-to-world
            })
        
        # 6. 按mask比例降序排序，选择mask最大的num_frames帧
        frames.sort(key=lambda f: f.get('mask_ratio', 0), reverse=True)
        
        # 从候选池中选择mask最大的num_frames帧
        if len(frames) > num_frames:
            frames = frames[:num_frames]
            print(f"[load_mv_object_data] Selected top {num_frames} frames by mask_ratio from {SAMPLE_POOL_SIZE} candidates")
        
        # 更新frame_index为排序后的索引
        for i, frame in enumerate(frames):
            frame['frame_index'] = i
        
        mask_ratios = [round(f.get('mask_ratio', 0), 3) for f in frames]
        print(f"[load_mv_object_data] Final {len(frames)} frames, mask_ratios: {mask_ratios}")
        
        # 7. 加载已有的world_pose
        world_pose = None
        aligned_path = f"{MV_ALIGNED_ROOT}/{scene_id}/{object_id}/world_pose.npy"
        if os.path.exists(aligned_path):
            world_pose = np.load(aligned_path).tolist()
        
        # 8. 读取GT bbox（从instances.json）
        gt_bbox = None
        instances_path = f"{LASA1M_ROOT}/{scene_id}/instances.json"
        if os.path.exists(instances_path):
            try:
                with open(instances_path, 'r') as f:
                    instances = json.load(f)
                for inst in instances:
                    if inst.get('id') == object_id:
                        gt_bbox = {
                            'position': inst.get('position'),   # 中心点（世界坐标）
                            'scale': inst.get('scale'),         # 半轴长度（局部坐标系）
                            'R': inst.get('R')                  # 旋转矩阵 (可选)
                        }
                        break
            except Exception as e:
                print(f"[load_mv_object_data] 警告: 加载instances.json失败: {e}")
        
        return {
            'scene_id': scene_id,
            'object_id': object_id,
            'mesh_url': mesh_url,
            'mesh_path': mesh_path,
            'world_pose': world_pose,
            'frames': frames,
            'total_frames': total_frames,
            'gt_bbox': gt_bbox
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
        
        elif parts[0] == 'mv_mesh':
            # /data/mv_mesh/{scene_id}/{object_id}/mesh.glb
            scene_id, object_id, filename = parts[1], parts[2], parts[3]
            file_path = f"{MV_RECON_ROOT}/{scene_id}/{object_id}/{filename}"
            print(f"[serve_data_file] mv_mesh file_path={file_path}")
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = f.read()
                self.send_binary(data, 'model/gltf-binary')
            else:
                self.send_error(404, f"Mesh not found: {file_path}")
        
        elif parts[0] == 'depth':
            # /data/depth/{scene_id}/{object_id}/{frame_id}
            scene_id, object_id, frame_id = parts[1], parts[2], parts[3]
            depth_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/gt/{frame_id}/depth.png"
            
            if os.path.exists(depth_path):
                import cv2
                depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_img is not None:
                    if depth_img.dtype == np.uint16:
                        depth_float = depth_img.astype(np.float32) / 1000.0  # mm to m
                    else:
                        depth_float = depth_img.astype(np.float32)
                    
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


def warmup_cache():
    """启动时预热缓存"""
    global _mv_objects_cache, _mv_objects_cache_time
    print("[warmup_cache] 正在预热缓存...")
    start_time = time.time()
    
    mv_objects = []
    for scene_dir in sorted(glob.glob(f"{MV_RECON_ROOT}/*")):
        if not os.path.isdir(scene_dir):
            continue
        scene_id = os.path.basename(scene_dir)
        
        for obj_dir in sorted(glob.glob(f"{scene_dir}/*")):
            if not os.path.isdir(obj_dir):
                continue
            object_id = os.path.basename(obj_dir)
            
            mesh_path = f"{obj_dir}/mesh.glb"
            if not os.path.exists(mesh_path):
                continue
            
            lasa_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}"
            if not os.path.exists(lasa_path):
                continue
            
            aligned_path = f"{MV_ALIGNED_ROOT}/{scene_id}/{object_id}"
            has_alignment = os.path.exists(f"{aligned_path}/world_pose.npy")
            
            # 读取分类状态
            category = None
            result_path = f"{aligned_path}/result.json"
            if os.path.exists(result_path):
                try:
                    with open(result_path, 'r') as f:
                        result_data = json.load(f)
                        category = result_data.get('category')
                except:
                    pass
            
            mv_objects.append({
                'scene_id': scene_id,
                'object_id': object_id,
                'num_frames': -1,
                'has_alignment': has_alignment,
                'category': category
            })
    
    mv_objects.sort(key=lambda x: (x['scene_id'], x['object_id']))
    _mv_objects_cache = mv_objects
    _mv_objects_cache_time = time.time()
    
    print(f"[warmup_cache] 缓存预热完成，耗时 {time.time() - start_time:.2f}s，共 {len(mv_objects)} 个物体")


def run_server(port=8084):
    """运行服务器"""
    # 启动时预热缓存
    warmup_cache()
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, MVDataAPIHandler)
    print(f"Multi-View Data server running at http://localhost:{port}")
    print(f"API endpoints:")
    print(f"  GET /api/mv_objects - List all multi-view objects")
    print(f"  GET /api/mv_object_data?scene_id=xxx&object_id=xxx&num_frames=8 - Get MV object data")
    print(f"  POST /api/save_mv_pose - Save aligned pose")
    httpd.serve_forever()


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8084
    run_server(port)
