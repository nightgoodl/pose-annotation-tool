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
import trimesh
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# 导入数据库模块
import db

# 渲染服务（懒加载）
_render_service = None
def get_render_service():
    global _render_service
    if _render_service is None:
        from render_service import get_renderer
        _render_service = get_renderer()
        print("[render_service] 初始化nvdiffrast渲染服务")
    return _render_service

# 数据路径配置
DATA_ROOT = "/root/csz/data_partcrafter"
LASA1M_ROOT = f"{DATA_ROOT}/LASA1M"
MV_RECON_ROOT = "/root/csz/yingbo/MV-SAM3D/reconstruction_lasa1m_v4"
MV_SAM_V3_ROOT = f"{DATA_ROOT}/LASA1M_MV_SAM_v3"
ANNOTATE_ROOT = f"{DATA_ROOT}/LASA1M_ANNOTATE"
MV_ALIGNED_ROOT = f"{DATA_ROOT}/LASA1M_ALIGNED_MV"  # 多视角对齐结果保存目录

# TOS 模式配置
TOS_MODE = os.environ.get("USE_TOS", "0") == "1"
MANIFEST_PATH = os.environ.get("MANIFEST_PATH", os.path.join(os.path.dirname(__file__), "manifest.json"))
if TOS_MODE:
    import tos_client
    print(f"[TOS_MODE] 已启用 TOS 数据源, manifest: {MANIFEST_PATH}, cache: {tos_client.TOS_CACHE_DIR}")

# 全局缓存
_mv_objects_cache = None
_mv_objects_cache_time = 0
CACHE_TTL = 300  # 缓存有效期5分钟

# 物体数据缓存 (load_mv_object_data 结果)
from collections import OrderedDict
_object_data_cache = OrderedDict()  # key: "scene_id/object_id/num_frames" -> data dict
_OBJECT_DATA_CACHE_MAX = 50  # 最多缓存50个物体

# 场景级缓存
_instances_cache = {}  # key: scene_id -> instances list
_mesh_info_cache = {}  # key: "scene_id/object_id" -> mesh_info dict
_image_size_cache = {}  # key: image_path -> (width, height)


class MVDataAPIHandler(SimpleHTTPRequestHandler):
    """多视角数据API处理器"""
    
    def get_current_user(self):
        """从请求头获取当前用户，返回 (user_dict, error_msg)"""
        auth_header = self.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return None, 'Missing or invalid Authorization header'
        token = auth_header[7:]
        user = db.get_user_by_token(token)
        if not user:
            return None, 'Invalid or expired token'
        if not user.get('is_active'):
            return None, 'User account is disabled'
        return user, None
    
    def require_auth(self):
        """要求认证，返回用户或发送401错误"""
        user, error = self.get_current_user()
        if not user:
            self.send_json({'error': error, 'code': 'UNAUTHORIZED'}, 401)
            return None
        return user
    
    def require_admin(self):
        """要求管理员权限，返回用户或发送错误"""
        user = self.require_auth()
        if not user:
            return None
        if user.get('role') != 'admin':
            self.send_json({'error': 'Admin access required', 'code': 'FORBIDDEN'}, 403)
            return None
        return user
    
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        # API路由
        # ========== 认证相关 ==========
        if path == '/api/auth/me':
            user = self.require_auth()
            if user:
                stats = db.get_user_stats(user['id'])
                active_scenes = db.get_user_active_scenes(user['id'])
                self.send_json({
                    'user': user,
                    'stats': stats,
                    'active_scenes': active_scenes
                })
        
        # ========== 管理员API ==========
        elif path == '/api/admin/users':
            user = self.require_admin()
            if user:
                users = db.get_all_user_stats()
                self.send_json({'users': users})
        
        elif path == '/api/admin/assignments':
            user = self.require_admin()
            if user:
                user_id = query.get('user_id', [None])[0]
                status = query.get('status', [None])[0]
                user_id = int(user_id) if user_id else None
                assignments = db.get_all_assignments(user_id, status)
                self.send_json({'assignments': assignments})
        
        elif path == '/api/admin/stats':
            user = self.require_admin()
            if user:
                users_stats = db.get_all_user_stats()
                # 获取所有场景统计
                all_scenes = self._get_all_scenes()
                active_assignments = db.get_all_assignments(status='active')
                completed_assignments = db.get_all_assignments(status='completed')
                assigned_scenes = {a['scene_id'] for a in active_assignments}
                completed_scenes = {a['scene_id'] for a in completed_assignments}
                
                self.send_json({
                    'users': users_stats,
                    'scenes': {
                        'total': len(all_scenes),
                        'assigned': len(assigned_scenes),
                        'completed': len(completed_scenes),
                        'unassigned': len(set(all_scenes) - assigned_scenes - completed_scenes)
                    }
                })
        
        elif path == '/api/admin/logs':
            user = self.require_admin()
            if user:
                user_id = query.get('user_id', [None])[0]
                date_from = query.get('date_from', [None])[0]
                date_to = query.get('date_to', [None])[0]
                limit = int(query.get('limit', [100])[0])
                offset = int(query.get('offset', [0])[0])
                user_id = int(user_id) if user_id else None
                
                logs = db.get_annotation_logs(user_id, date_from, date_to, limit, offset)
                total = db.get_logs_count(user_id, date_from, date_to)
                self.send_json({'logs': logs, 'total': total})
        
        elif path == '/api/admin/scenes':
            user = self.require_admin()
            if user:
                # 返回所有场景及其状态
                all_scenes = self._get_all_scenes()
                active_assignments = db.get_all_assignments(status='active')
                completed_assignments = db.get_all_assignments(status='completed')
                
                scene_status = {}
                for s in all_scenes:
                    scene_status[s] = {'status': 'unassigned', 'user': None}
                for a in active_assignments:
                    scene_status[a['scene_id']] = {'status': 'active', 'user': a['username'], 'user_id': a['user_id']}
                for a in completed_assignments:
                    if a['scene_id'] in scene_status and scene_status[a['scene_id']]['status'] == 'unassigned':
                        scene_status[a['scene_id']] = {'status': 'completed', 'user': a['username']}
                
                self.send_json({'scenes': scene_status})
        
        # ========== 原有API ==========
        elif path == '/api/aligned_objects':
            self.handle_list_aligned_objects()
        elif path == '/api/refresh_cache':
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
        
        elif path == '/api/next_mv_task':
            current_scene_id = query.get('current_scene_id', [None])[0]
            exclude_object_id = query.get('exclude_object_id', [None])[0]
            self.handle_next_mv_task(current_scene_id, exclude_object_id)
        elif path == '/api/mv_objects':
            page = int(query.get('page', [1])[0])
            page_size = int(query.get('page_size', [50])[0])
            scene_filter = query.get('scene', [None])[0]
            sort_by = query.get('sort_by', [None])[0]
            sort_order = query.get('sort_order', ['desc'])[0]
            status_filter = query.get('status', [None])[0]
            self.handle_list_mv_objects(page, page_size, scene_filter, sort_by, sort_order, status_filter)
        elif path == '/api/mv_object_data':
            scene_id = query.get('scene_id', [None])[0]
            object_id = query.get('object_id', [None])[0]
            num_frames = int(query.get('num_frames', [8])[0])
            self.handle_get_mv_object_data(scene_id, object_id, num_frames)
        elif path == '/api/preload':
            # 预加载下一个物体数据到缓存（前端异步调用）
            scene_id = query.get('scene_id', [None])[0]
            object_id = query.get('object_id', [None])[0]
            num_frames = int(query.get('num_frames', [8])[0])
            self.handle_preload(scene_id, object_id, num_frames)
        elif path == '/api/object_data':
            # Compatibility endpoint: convert mv_object_data to single-view format
            scene_id = query.get('scene_id', [None])[0]
            object_id = query.get('object_id', [None])[0]
            frame_id = query.get('frame_id', [None])[0]
            
            if frame_id == 'multi_view':
                # This is a multi-view object, convert to single-view format using first frame
                self.handle_get_object_data_compat(scene_id, object_id)
            else:
                # Single-view not supported in this server
                self.send_json({
                    'error': 'Single-view objects not supported. Use /api/mv_object_data for multi-view objects.',
                    'hint': 'For multi-view objects, use frame_id=multi_view or call /api/mv_object_data directly'
                }, 400)
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
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
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
            data = json.loads(body.decode('utf-8')) if body else {}
        except json.JSONDecodeError:
            self.send_json({'error': 'Invalid JSON'}, 400)
            return
        
        # ========== 认证API ==========
        if path == '/api/auth/login':
            username = data.get('username', '')
            password = data.get('password', '')
            user = db.verify_user(username, password)
            if user:
                token = db.create_token(user['id'])
                self.send_json({
                    'success': True,
                    'token': token,
                    'user': user
                })
            else:
                self.send_json({'success': False, 'error': 'Invalid username or password'}, 401)
        
        elif path == '/api/auth/logout':
            auth_header = self.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
                db.revoke_token(token)
            self.send_json({'success': True})
        
        # ========== 管理员API ==========
        elif path == '/api/admin/users':
            user = self.require_admin()
            if user:
                username = data.get('username')
                password = data.get('password')
                batch_size = data.get('batch_size', 5)
                role = data.get('role', 'annotator')
                
                if not username or not password:
                    self.send_json({'error': 'Username and password required'}, 400)
                    return
                
                user_id = db.create_user(username, password, role, batch_size)
                if user_id:
                    self.send_json({'success': True, 'user_id': user_id})
                else:
                    self.send_json({'error': 'Username already exists'}, 400)
        
        elif path.startswith('/api/admin/users/'):
            user = self.require_admin()
            if user:
                # 解析用户ID: /api/admin/users/123
                try:
                    target_user_id = int(path.split('/')[-1])
                except ValueError:
                    self.send_json({'error': 'Invalid user ID'}, 400)
                    return
                
                updates = {}
                if 'password' in data:
                    updates['password'] = data['password']
                if 'batch_size' in data:
                    updates['batch_size'] = data['batch_size']
                if 'is_active' in data:
                    updates['is_active'] = 1 if data['is_active'] else 0
                if 'role' in data:
                    updates['role'] = data['role']
                
                if updates:
                    success = db.update_user(target_user_id, **updates)
                    self.send_json({'success': success})
                else:
                    self.send_json({'error': 'No valid fields to update'}, 400)
        
        # ========== 原有API ==========
        elif path == '/api/claim_scenes':
            # 标注员自助领取场景（强制增量领取）
            user = self.require_auth()
            if not user:
                return
            
            assigned = self._assign_scenes_to_user(user)
            active_scenes = db.get_user_active_scenes(user['id'])
            
            self.send_json({
                'success': True,
                'assigned': assigned or [],
                'active_scenes': active_scenes,
                'message': f'已领取 {len(assigned) if assigned else 0} 个新场景' if assigned else '没有可领取的场景'
            })
        elif path == '/api/save_mv_pose':
            self.handle_save_mv_pose(data)
        elif path == '/api/render_mesh':
            self.handle_render_mesh(data)
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
        average_iou = data.get('average_iou', 0.0)
        result = {
            "scene_id": scene_id,
            "object_id": object_id,
            "success": True,
            "scale": scale,
            "alignment_error": error,
            "source": "manual_mv_annotation",
            "num_point_pairs": len(point_pairs),
            "category": category,
            "average_iou": average_iou,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")
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
        
        # 立即更新缓存，避免 next_mv_task 重复返回已标注物体
        self._update_single_object_cache(scene_id, object_id)
        
        # TOS 模式：标注完成后删除本地 tar 缓存以释放磁盘空间
        if TOS_MODE:
            # 清除内存缓存
            cache_key = f"{scene_id}/{object_id}"
            _mesh_info_cache.pop(cache_key, None)
            for k in list(_object_data_cache.keys()):
                if k.startswith(f"{scene_id}/{object_id}/"):
                    _object_data_cache.pop(k, None)
            # 删除解压的 tar 目录
            tos_client.delete_object_cache(scene_id, object_id)
        
        # 记录标注日志（只记录成功对齐的，invalid 和 align_difficult 不计入统计）
        user, _ = self.get_current_user()
        if user:
            if category == 'fixed':
                # 只有成功对齐的才记录到统计
                db.log_annotation(user['id'], scene_id, object_id, category, 'align')
            
            # 检查场景是否完成
            self._check_and_complete_scene(user['id'], scene_id)
        
        self.send_json({
            'success': True,
            'pose_path': pose_path,
            'result_path': result_path
        })
    
    def handle_render_mesh(self, data):
        """
        渲染mesh到指定相机视角
        
        输入格式:
        {
            "mesh_path": "/path/to/mesh.glb",
            "pose": [[...], [...], [...], [...]],  # 4x4矩阵
            "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240},
            "extrinsics": [[...], [...], [...], [...]],  # 4x4 camera-to-world
            "image_size": [480, 640]  # [H, W]
        }
        
        输出: PNG图像（RGBA）
        """
        mesh_path = data.get('mesh_path')
        pose = data.get('pose')
        intrinsics = data.get('intrinsics')
        extrinsics = data.get('extrinsics')
        image_size = data.get('image_size')
        
        if not all([mesh_path, pose, intrinsics, extrinsics, image_size]):
            self.send_json({'error': 'Missing required fields'}, 400)
            return
        
        if not os.path.exists(mesh_path):
            self.send_json({'error': f'Mesh not found: {mesh_path}'}, 404)
            return
        
        try:
            render_service = get_render_service()
            
            pose_np = np.array(pose, dtype=np.float64)
            extrinsics_np = np.array(extrinsics, dtype=np.float64)
            image_size_tuple = tuple(image_size)
            
            png_data = render_service.render_to_png(
                mesh_path, pose_np, intrinsics, extrinsics_np, image_size_tuple
            )
            
            self.send_binary(png_data, 'image/png')
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({'error': str(e)}, 500)
    
    def handle_list_mv_objects(self, page=1, page_size=50, scene_filter=None, sort_by=None, sort_order='desc', status_filter=None):
        """列出多视角重建物体（带分页、缓存、排序和状态过滤）"""
        global _mv_objects_cache, _mv_objects_cache_time
        
        # 检查缓存是否有效
        current_time = time.time()
        if _mv_objects_cache is None or (current_time - _mv_objects_cache_time) > CACHE_TTL:
            print("[handle_list_mv_objects] 重建缓存...")
            start_time = time.time()
            _mv_objects_cache = self._build_mv_objects_cache()
            _mv_objects_cache_time = current_time
            print(f"[handle_list_mv_objects] 缓存构建完成，耗时 {time.time() - start_time:.2f}s，共 {len(_mv_objects_cache)} 个物体")
        
        # 获取当前用户
        user, _ = self.get_current_user()
        is_admin = user and user.get('role') == 'admin'
        
        # 获取用户的活跃场景（非管理员只能看到分配给自己的场景）
        user_scenes = None
        if user and not is_admin:
            user_scenes = set(db.get_user_active_scenes(user['id']))
            # 不再自动领取，由用户手动点击"领取场景"按钮
        
        # 应用场景过滤
        filtered = list(_mv_objects_cache)  # copy
        
        # 非管理员只能看到分配给自己的场景
        if user_scenes is not None:
            filtered = [o for o in filtered if o['scene_id'] in user_scenes]
        
        if scene_filter:
            filtered = [o for o in filtered if o['scene_id'] == scene_filter]
        
        # 应用状态过滤
        if status_filter:
            if status_filter == 'pending':
                filtered = [o for o in filtered if not o.get('has_alignment') and o.get('category') not in ('fixed', 'invalid', 'align_difficult')]
            elif status_filter == 'aligned':
                filtered = [o for o in filtered if o.get('category') == 'fixed' or (o.get('has_alignment') and o.get('category') not in ('invalid', 'align_difficult'))]
            elif status_filter == 'invalid':
                filtered = [o for o in filtered if o.get('category') == 'invalid']
            elif status_filter == 'align_difficult':
                filtered = [o for o in filtered if o.get('category') == 'align_difficult']
        
        # 应用排序
        reverse = (sort_order == 'desc')
        if sort_by == 'average_iou':
            filtered.sort(key=lambda x: x.get('average_iou', 0) or 0, reverse=reverse)
        elif sort_by == 'num_point_pairs':
            filtered.sort(key=lambda x: x.get('num_point_pairs', 0) or 0, reverse=reverse)
        elif sort_by == 'saved_at':
            filtered.sort(key=lambda x: x.get('saved_at') or '', reverse=reverse)
        # 默认不额外排序，保持 scene_id + object_id 顺序
        
        # 用户可见的物体列表（用于统计，不受状态过滤影响）
        user_objects = [o for o in _mv_objects_cache if user_scenes is None or o['scene_id'] in user_scenes]
        
        # 统计各状态数量（基于用户可见的物体）
        all_count = len(user_objects)
        pending_count = sum(1 for o in user_objects if not o.get('has_alignment') and o.get('category') not in ('fixed', 'invalid', 'align_difficult'))
        aligned_count = sum(1 for o in user_objects if o.get('category') == 'fixed' or (o.get('has_alignment') and o.get('category') not in ('invalid', 'align_difficult')))
        invalid_count = sum(1 for o in user_objects if o.get('category') == 'invalid')
        align_difficult_count = sum(1 for o in user_objects if o.get('category') == 'align_difficult')
        
        total = len(filtered)
        
        # 分页
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_data = filtered[start_idx:end_idx]
        
        # 获取场景ID列表（用于前端筛选，基于用户可见的物体）
        all_scenes = sorted(set(o['scene_id'] for o in user_objects))
        
        self.send_json({
            'mv_objects': page_data,
            'total': total,
            'page': page,
            'page_size': page_size,
            'total_pages': (total + page_size - 1) // page_size,
            'scenes': all_scenes[:100],
            'stats': {
                'all': all_count,
                'pending': pending_count,
                'aligned': aligned_count,
                'invalid': invalid_count,
                'align_difficult': align_difficult_count
            }
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
        num_point_pairs = 0
        average_iou = 0.0
        saved_at = None
        result_path = f"{aligned_path}/result.json"
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r') as f:
                    result_data = json.load(f)
                    category = result_data.get('category')
                    num_point_pairs = result_data.get('num_point_pairs', 0)
                    average_iou = result_data.get('average_iou', 0.0)
                    saved_at = result_data.get('saved_at')
            except:
                pass
        
        # 更新缓存中的对象
        for obj in _mv_objects_cache:
            if obj['scene_id'] == scene_id and obj['object_id'] == object_id:
                obj['has_alignment'] = has_alignment
                obj['category'] = category
                obj['num_point_pairs'] = num_point_pairs
                obj['average_iou'] = average_iou
                obj['saved_at'] = saved_at
                print(f"[Cache] 更新对象 {scene_id}/{object_id}: category={category}, has_alignment={has_alignment}, iou={average_iou}")
                break
    
    def handle_list_aligned_objects(self):
        """列出所有已对齐的物体（单视图兼容性）"""
        aligned_objects = []
        
        # 对于多视角数据，从MV_ALIGNED_ROOT读取
        for scene_dir in sorted(glob.glob(f"{MV_ALIGNED_ROOT}/*")):
            if not os.path.isdir(scene_dir):
                continue
            scene_id = os.path.basename(scene_dir)
            
            for obj_dir in sorted(glob.glob(f"{scene_dir}/*")):
                if not os.path.isdir(obj_dir):
                    continue
                object_id = os.path.basename(obj_dir)
                
                # 查找world_pose文件（多视角只有一个world_pose.npy）
                world_pose_file = f"{obj_dir}/world_pose.npy"
                if os.path.exists(world_pose_file):
                    # 读取result.json获取信息
                    result_file = f"{obj_dir}/result.json"
                    error = None
                    category = None
                    if os.path.exists(result_file):
                        try:
                            with open(result_file) as f:
                                result = json.load(f)
                                error = result.get('alignment_error')
                                category = result.get('category')
                        except:
                            pass
                    
                    aligned_objects.append({
                        'scene_id': scene_id,
                        'object_id': object_id,
                        'frame_id': 'multi_view',  # 多视角标记
                        'iou': None,  # 多视角没有单帧IOU
                        'error': error,
                        'category': category
                    })
        
        # 按scene_id排序
        aligned_objects.sort(key=lambda x: (x['scene_id'], x['object_id']))
        self.send_json({'aligned_objects': aligned_objects[:200]})  # 返回前200个
    
    def handle_next_mv_task(self, current_scene_id, exclude_object_id):
        """
        获取下一个待标注的多视角物体。
        
        对于普通用户：只从分配给自己的场景中获取任务，如果当前批次完成则自动领取新场景。
        对于管理员：可以看到所有待标注物体。
        """
        global _mv_objects_cache, _mv_objects_cache_time
        
        # 确保缓存已建立
        current_time = time.time()
        if _mv_objects_cache is None or (current_time - _mv_objects_cache_time) > CACHE_TTL:
            _mv_objects_cache = self._build_mv_objects_cache()
            _mv_objects_cache_time = current_time
        
        # 获取当前用户
        user, _ = self.get_current_user()
        is_admin = user and user.get('role') == 'admin'
        
        # 获取用户的活跃场景
        user_scenes = None
        if user and not is_admin:
            user_scenes = set(db.get_user_active_scenes(user['id']))
            # 不再自动领取，由用户手动点击"领取场景"按钮
        
        # 筛选待标注物体
        pending_objects = []
        for obj in _mv_objects_cache:
            # 排除当前正在标注的物体
            if (obj['scene_id'] == current_scene_id and 
                obj['object_id'] == exclude_object_id):
                continue
            
            # 跳过已标注的
            if obj.get('has_alignment') or obj.get('category') in ('fixed', 'invalid', 'align_difficult'):
                continue
            
            # 非管理员只能看到分配给自己的场景
            if user_scenes is not None and obj['scene_id'] not in user_scenes:
                continue
            
            pending_objects.append({
                'scene_id': obj['scene_id'],
                'object_id': obj['object_id'],
                'is_same_scene': obj['scene_id'] == current_scene_id
            })
        
        # 如果当前批次没有待标注物体，标记已完成的场景（但不自动领取新场景）
        if not pending_objects and user and not is_admin:
            # 标记已完成的场景
            if user_scenes:
                for scene_id in user_scenes:
                    if self._is_scene_completed(scene_id):
                        db.mark_scene_completed(user['id'], scene_id)
            # 不再自动领取，由用户手动点击"领取场景"按钮
        
        if not pending_objects:
            self.send_json({
                'success': True,
                'has_next': False,
                'data': None,
                'remaining_count': 0,
                'message': 'No more tasks available'
            })
            return
        
        # 排序：优先同场景
        pending_objects.sort(key=lambda x: (
            0 if x['is_same_scene'] else 1,
            x['scene_id'],
            x['object_id']
        ))
        
        next_task = pending_objects[0]
        remaining_count = len(pending_objects)
        
        print(f"[next_mv_task] 下一个任务: scene={next_task['scene_id']}, "
              f"object={next_task['object_id']}, remaining={remaining_count}, user={user['username'] if user else 'anonymous'}")
        
        self.send_json({
            'success': True,
            'has_next': True,
            'data': {
                'scene_id': next_task['scene_id'],
                'object_id': next_task['object_id'],
            },
            'remaining_count': remaining_count
        })
    
    def _get_all_scenes(self):
        """获取所有场景ID列表"""
        global _mv_objects_cache
        if _mv_objects_cache:
            return sorted(set(obj['scene_id'] for obj in _mv_objects_cache))
        
        # 从目录获取
        scenes = []
        for scene_dir in glob.glob(f"{MV_SAM_V3_ROOT}/*"):
            if os.path.isdir(scene_dir):
                scenes.append(os.path.basename(scene_dir))
        return sorted(scenes)
    
    def _is_scene_completed(self, scene_id):
        """检查场景是否已完成（所有物体都已标注）"""
        global _mv_objects_cache
        if not _mv_objects_cache:
            return False
        
        for obj in _mv_objects_cache:
            if obj['scene_id'] != scene_id:
                continue
            # 如果有任何物体未标注，场景未完成
            if not obj.get('has_alignment') and obj.get('category') not in ('fixed', 'invalid', 'align_difficult'):
                return False
        return True
    
    def _assign_scenes_to_user(self, user, count=None, force_add=False):
        """为用户分配场景
        
        Args:
            user: 用户信息
            count: 要领取的数量（默认使用 batch_size）
            force_add: 如果为 True，则强制增量领取 count 个；否则补齐到 batch_size
        
        Returns:
            list: 新分配的场景ID列表
        """
        if not user:
            return []
        
        user_id = user['id']
        batch_size = count if count else user.get('batch_size', 5)
        
        # 获取当前活跃场景数
        active_scenes = db.get_user_active_scenes(user_id)
        
        if force_add:
            # 强制增量领取指定数量
            need_count = batch_size
        else:
            # 补齐到 batch_size
            need_count = batch_size - len(active_scenes)
        
        if need_count <= 0:
            return []
        
        # 获取所有场景
        all_scenes = self._get_all_scenes()
        
        # 获取未分配的场景（只选择有待标注物体的场景）
        unassigned = db.get_unassigned_scenes(all_scenes, need_count * 2)  # 多取一些以便筛选
        
        # 筛选出有待标注物体的场景
        scenes_to_assign = []
        for scene_id in unassigned:
            if not self._is_scene_completed(scene_id):
                scenes_to_assign.append(scene_id)
                if len(scenes_to_assign) >= need_count:
                    break
        
        if scenes_to_assign:
            assigned_count = db.assign_scenes_to_user(user_id, scenes_to_assign)
            print(f"[assign_scenes] 为用户 {user['username']} 分配了 {assigned_count} 个场景: {scenes_to_assign}")
            return scenes_to_assign
        return []
    
    def _check_and_complete_scene(self, user_id, scene_id):
        """检查场景是否完成，如果完成则标记"""
        if self._is_scene_completed(scene_id):
            db.mark_scene_completed(user_id, scene_id)
            print(f"[complete_scene] 场景 {scene_id} 已完成")
    
    def _build_mv_objects_cache(self):
        """构建物体列表缓存"""
        if TOS_MODE:
            return self._build_mv_objects_cache_tos()
        return self._build_mv_objects_cache_local()
    
    def _build_mv_objects_cache_tos(self):
        """TOS 模式：从 manifest.json 构建物体列表"""
        mv_objects = []
        
        if not os.path.exists(MANIFEST_PATH):
            print(f"[_build_mv_objects_cache_tos] manifest 文件不存在: {MANIFEST_PATH}")
            return mv_objects
        
        with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        for entry in manifest.get('objects', []):
            scene_id = entry['scene_id']
            object_id = entry['object_id']
            
            # TOS 模式下缩略图不可用（tar 包在浏览列表时尚未下载）
            thumbnail_url = None
            
            # 检查是否已有对齐结果和分类状态
            aligned_path = f"{MV_ALIGNED_ROOT}/{scene_id}/{object_id}"
            has_alignment = os.path.exists(f"{aligned_path}/world_pose.npy")
            
            category = None
            num_point_pairs = 0
            average_iou = 0.0
            saved_at = None
            result_path = f"{aligned_path}/result.json"
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    result_data = json.load(f)
                    category = result_data.get('category')
                    num_point_pairs = result_data.get('num_point_pairs', 0)
                    average_iou = result_data.get('average_iou', 0.0)
                    saved_at = result_data.get('saved_at')
            
            mv_objects.append({
                'scene_id': scene_id,
                'object_id': object_id,
                'num_frames': entry.get('num_frames', -1),
                'has_mesh': True,  # tar 包内一定有 mesh
                'has_alignment': has_alignment,
                'category': category,
                'obj_category': entry.get('category', ''),
                'thumbnail_url': thumbnail_url,
                'num_point_pairs': num_point_pairs,
                'average_iou': average_iou,
                'saved_at': saved_at
            })
        
        mv_objects.sort(key=lambda x: (x['scene_id'], x['object_id']))
        return mv_objects
    
    def _build_mv_objects_cache_local(self):
        """本地模式：基于 v3 mask/image 数据，检查 v4 mesh 存在"""
        mv_objects = []
        
        # 遍历 MV_SAM_V3 目录
        for scene_dir in sorted(glob.glob(f"{MV_SAM_V3_ROOT}/*")):
            if not os.path.isdir(scene_dir):
                continue
            scene_id = os.path.basename(scene_dir)
            
            for obj_dir in sorted(glob.glob(f"{scene_dir}/*")):
                if not os.path.isdir(obj_dir):
                    continue
                object_id = os.path.basename(obj_dir)
                if object_id in ("vis", "images"):
                    continue
                
                # 必须有 v3 metadata
                meta_path = f"{obj_dir}/frame_selection_metadata.json"
                if not os.path.exists(meta_path):
                    continue
                
                # 检查 v4 mesh（可选，重建中的物体也列出但标记无mesh）
                mesh_path = f"{MV_RECON_ROOT}/{scene_id}/{object_id}/mesh.glb"
                has_mesh = os.path.exists(mesh_path)
                
                # 获取缩略图URL（v3 第一张 image）
                thumbnail_url = None
                v3_images_dir = f"{obj_dir}/images"
                if os.path.isdir(v3_images_dir):
                    img_files = sorted(glob.glob(f"{v3_images_dir}/*.png"))
                    if img_files:
                        first_idx = os.path.basename(img_files[0]).replace('.png', '')
                        thumbnail_url = f"/data/v3_image/{scene_id}/{object_id}/{first_idx}.png"
                
                # 检查是否已有对齐结果和分类状态
                aligned_path = f"{MV_ALIGNED_ROOT}/{scene_id}/{object_id}"
                has_alignment = os.path.exists(f"{aligned_path}/world_pose.npy")
                
                # 读取result.json获取全部元数据
                category = None
                num_point_pairs = 0
                average_iou = 0.0
                saved_at = None
                result_path = f"{aligned_path}/result.json"
                if os.path.exists(result_path):
                    try:
                        with open(result_path, 'r') as f:
                            result_data = json.load(f)
                            category = result_data.get('category')
                            num_point_pairs = result_data.get('num_point_pairs', 0)
                            average_iou = result_data.get('average_iou', 0.0)
                            saved_at = result_data.get('saved_at')
                    except:
                        pass
                
                mv_objects.append({
                    'scene_id': scene_id,
                    'object_id': object_id,
                    'num_frames': -1,  # 延迟加载
                    'has_mesh': has_mesh,
                    'has_alignment': has_alignment,
                    'category': category,
                    'thumbnail_url': thumbnail_url,
                    'num_point_pairs': num_point_pairs,
                    'average_iou': average_iou,
                    'saved_at': saved_at
                })
        
        # 按scene_id排序
        mv_objects.sort(key=lambda x: (x['scene_id'], x['object_id']))
        return mv_objects
    
    def handle_get_mv_object_data(self, scene_id, object_id, num_frames=8):
        """获取多视角物体的完整数据（带缓存）"""
        if not all([scene_id, object_id]):
            self.send_json({'error': 'scene_id, object_id required'}, 400)
            return
        
        try:
            cache_key = f"{scene_id}/{object_id}/{num_frames}"
            
            # 检查缓存（但 world_pose/point_pairs 需要实时读取，且需检测 mesh 更新）
            if cache_key in _object_data_cache:
                data = _object_data_cache[cache_key]
                mesh_path = data.get('mesh_path', f"{MV_RECON_ROOT}/{scene_id}/{object_id}/mesh.glb")
                mesh_mtime = os.path.getmtime(mesh_path) if os.path.exists(mesh_path) else 0
                aligned_path = f"{MV_ALIGNED_ROOT}/{scene_id}/{object_id}/world_pose.npy"
                alignment_stale = False
                if os.path.exists(aligned_path):
                    if mesh_mtime > os.path.getmtime(aligned_path):
                        alignment_stale = True
                        print(f"[mv_object_data] ⚠️ 缓存命中但 mesh 已更新，丢弃旧对齐: {cache_key}")
                        data['world_pose'] = None
                        data['point_pairs'] = []
                        # 清除过期的 mesh_info 缓存
                        mk = f"{scene_id}/{object_id}"
                        _mesh_info_cache.pop(mk, None)
                        data['mesh_info'] = self._get_mesh_info(scene_id, object_id)
                    else:
                        data['world_pose'] = np.load(aligned_path).tolist()
                        # 也实时读取 point_pairs
                        pairs_path = f"{MV_ALIGNED_ROOT}/{scene_id}/{object_id}/point_pairs.json"
                        if os.path.exists(pairs_path):
                            with open(pairs_path, 'r') as f:
                                data['point_pairs'] = json.load(f)
                        else:
                            data['point_pairs'] = []
                else:
                    data['world_pose'] = None
                    data['point_pairs'] = []
                # 移到最近使用
                _object_data_cache.move_to_end(cache_key)
                print(f"[mv_object_data] 缓存命中: {cache_key} (stale={alignment_stale})")
                self.send_json(data)
                return
            
            start_time = time.time()
            data = self.load_mv_object_data(scene_id, object_id, num_frames)
            elapsed = time.time() - start_time
            print(f"[mv_object_data] 加载耗时: {elapsed:.2f}s for {cache_key}")
            
            # 存入缓存
            _object_data_cache[cache_key] = data
            if len(_object_data_cache) > _OBJECT_DATA_CACHE_MAX:
                _object_data_cache.popitem(last=False)  # 删除最旧的
            
            self.send_json(data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({'error': str(e)}, 500)
    
    def handle_preload(self, scene_id, object_id, num_frames=8):
        """预加载物体数据到缓存（前端异步触发，无需完整响应）"""
        if not all([scene_id, object_id]):
            self.send_json({'success': False, 'error': 'scene_id, object_id required'})
            return
        try:
            cache_key = f"{scene_id}/{object_id}/{num_frames}"
            if cache_key in _object_data_cache:
                self.send_json({'success': True, 'cached': True})
                return
            
            start_time = time.time()
            data = self.load_mv_object_data(scene_id, object_id, num_frames)
            _object_data_cache[cache_key] = data
            if len(_object_data_cache) > _OBJECT_DATA_CACHE_MAX:
                _object_data_cache.popitem(last=False)
            elapsed = time.time() - start_time
            print(f"[preload] 预加载完成: {cache_key} ({elapsed:.2f}s)")
            self.send_json({'success': True, 'cached': False, 'elapsed': round(elapsed, 2)})
        except Exception as e:
            print(f"[preload] 预加载失败: {e}")
            self.send_json({'success': False, 'error': str(e)})
    
    def handle_get_object_data_compat(self, scene_id, object_id):
        """
        兼容性端点: 将多视角数据转换为单视图格式
        使用第一帧的数据来兼容单视图App
        """
        if not all([scene_id, object_id]):
            self.send_json({'error': 'scene_id, object_id required'}, 400)
            return
        
        try:
            # 加载多视角数据（只需要1帧）
            mv_data = self.load_mv_object_data(scene_id, object_id, num_frames=1)
            
            if not mv_data['frames']:
                self.send_json({'error': 'No frames available'}, 404)
                return
            
            # 使用第一帧的数据
            first_frame = mv_data['frames'][0]
            
            # 转换为单视图格式
            single_view_data = {
                'scene_id': scene_id,
                'object_id': object_id,
                'frame_id': first_frame['frame_id'],
                'rgb_url': first_frame['rgb_url'],
                'mask_url': first_frame['mask_url'],
                'depth_url': first_frame['depth_url'],
                'camera_intrinsics': first_frame['camera_intrinsics'],
                'camera_extrinsics': first_frame['camera_extrinsics'],
                'world_pose': mv_data['world_pose'],
                'mesh_url': mv_data['mesh_url'],
                'cad_model_url': mv_data['mesh_url'],  # 别名
                'gt_bbox': mv_data.get('gt_bbox'),
                'mesh_info': mv_data.get('mesh_info')
            }
            
            self.send_json(single_view_data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({'error': str(e)}, 500)
    
    def _get_image_size(self, path):
        """获取图像尺寸（带缓存）"""
        if path in _image_size_cache:
            return _image_size_cache[path]
        from PIL import Image
        with Image.open(path) as img:
            size = img.size  # (width, height)
        _image_size_cache[path] = size
        return size
    
    def _get_instances(self, scene_id):
        """获取场景的 instances.json（带缓存，优先 ANNOTATE，fallback LASA1M）"""
        if scene_id in _instances_cache:
            return _instances_cache[scene_id]
        for base in (ANNOTATE_ROOT, LASA1M_ROOT):
            instances_path = f"{base}/{scene_id}/instances.json"
            if os.path.exists(instances_path):
                try:
                    with open(instances_path, 'r') as f:
                        data = json.load(f)
                    _instances_cache[scene_id] = data
                    return data
                except:
                    pass
        _instances_cache[scene_id] = []
        return []
    
    def _get_mesh_info(self, scene_id, object_id, mesh_path_override=None):
        """获取mesh_info（带缓存）"""
        cache_key = f"{scene_id}/{object_id}"
        if cache_key in _mesh_info_cache:
            return _mesh_info_cache[cache_key]
        
        if mesh_path_override:
            mesh_path = mesh_path_override
        elif TOS_MODE:
            obj_dir = tos_client.get_object_dir(scene_id, object_id)
            mesh_path = os.path.join(obj_dir, "mesh.glb")
        else:
            mesh_path = f"{MV_RECON_ROOT}/{scene_id}/{object_id}/mesh.glb"
        mesh_info = None
        try:
            mesh = trimesh.load(mesh_path)
            vertices = None
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) > 0:
                    all_vertices = []
                    for geom in mesh.geometry.values():
                        if hasattr(geom, 'vertices'):
                            all_vertices.append(np.array(geom.vertices))
                    if all_vertices:
                        vertices = np.vstack(all_vertices)
            elif hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                vertices = np.array(mesh.vertices)
            
            if vertices is not None and len(vertices) > 0:
                mesh_info = {
                    'center': vertices.mean(axis=0).tolist(),
                    'extent': (vertices.max(axis=0) - vertices.min(axis=0)).tolist()
                }
        except Exception as e:
            print(f"[_get_mesh_info] 警告: {e}")
        
        _mesh_info_cache[cache_key] = mesh_info
        return mesh_info
    
    def load_mv_object_data(self, scene_id, object_id, num_frames=8):
        """加载多视角物体数据"""
        if TOS_MODE:
            return self._load_mv_object_data_tos(scene_id, object_id, num_frames)
        return self._load_mv_object_data_local(scene_id, object_id, num_frames)
    
    def _load_mv_object_data_tos(self, scene_id, object_id, num_frames=8):
        """TOS 模式：从下载解压的 tar 缓存加载物体数据"""
        # 1. 确保 tar 已下载并解压到本地缓存
        obj_dir = tos_client.ensure_object_cached(scene_id, object_id)
        
        # 2. 读取 mesh
        mesh_path = os.path.join(obj_dir, "mesh.glb")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found in tar cache: {mesh_path}")
        mesh_url = f"/data/tar_mesh/{scene_id}/{object_id}/mesh.glb"
        
        # 3. 读取 meta.json 获取 bbox 和基本信息
        meta_path = os.path.join(obj_dir, "meta.json")
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        total_frames = meta.get('num_frames', 0)
        
        # 4. 构建帧数据（从 {idx}.cam.json 读取相机参数）
        frames = []
        for idx in range(min(total_frames, num_frames)):
            idx_str = f"{idx:05d}"
            
            # 检查 RGB 文件存在
            rgb_path = os.path.join(obj_dir, f"{idx_str}.rgb.webp")
            if not os.path.exists(rgb_path):
                continue
            
            # 读取相机参数
            cam_path = os.path.join(obj_dir, f"{idx_str}.cam.json")
            if not os.path.exists(cam_path):
                continue
            with open(cam_path, 'r') as f:
                cam = json.load(f)
            
            # 获取图像尺寸
            image_width, image_height = self._get_image_size(rgb_path)
            
            K = cam.get('gt_depth_K')
            RT = cam.get('gt_RT')
            image_K = cam.get('image_K')
            
            # image_K 是全分辨率内参，可直接使用
            K_scaled = image_K
            
            # mask
            mask_path = os.path.join(obj_dir, f"{idx_str}.mask.png")
            mask_exists = os.path.exists(mask_path)
            mask_ratio = cam.get('mask_ratio', 0.0)
            
            # depth
            depth_path = os.path.join(obj_dir, f"{idx_str}.depth.png")
            depth_url = f"/data/tar_depth/{scene_id}/{object_id}/{idx_str}" if os.path.exists(depth_path) else None
            
            frames.append({
                'frame_id': str(cam.get('timestamp', idx)),
                'frame_index': idx,
                'rgb_url': f"/data/tar_rgb/{scene_id}/{object_id}/{idx_str}.webp",
                'mask_url': f"/data/tar_mask/{scene_id}/{object_id}/{idx_str}.png" if mask_exists else None,
                'depth_url': depth_url,
                'mask_ratio': mask_ratio,
                'camera_intrinsics': {
                    'K': K_scaled,
                    'K_original': K,
                    'width': image_width,
                    'height': image_height
                },
                'camera_extrinsics': RT
            })
        
        for i, frame in enumerate(frames):
            frame['frame_index'] = i
        
        # 5. 加载已有的 world_pose
        world_pose = None
        aligned_npy = f"{MV_ALIGNED_ROOT}/{scene_id}/{object_id}/world_pose.npy"
        if os.path.exists(aligned_npy):
            world_pose = np.load(aligned_npy).tolist()
        
        # 6. GT bbox 从 meta.json
        gt_bbox = {
            'position': meta.get('position'),
            'scale': meta.get('scale'),
            'R': meta.get('R'),
            'corners': meta.get('corners')
        }
        obj_category = meta.get('category')
        obj_caption = meta.get('caption')
        
        # 7. mesh_info
        mesh_info = self._get_mesh_info(scene_id, object_id, mesh_path_override=mesh_path)
        
        # 8. 已保存的关键点对
        saved_point_pairs = []
        pairs_path = f"{MV_ALIGNED_ROOT}/{scene_id}/{object_id}/point_pairs.json"
        if os.path.exists(pairs_path):
            with open(pairs_path, 'r') as f:
                saved_point_pairs = json.load(f)
        
        return {
            'scene_id': scene_id,
            'object_id': object_id,
            'mesh_url': mesh_url,
            'mesh_path': mesh_path,
            'world_pose': world_pose,
            'frames': frames,
            'total_frames': total_frames,
            'gt_bbox': gt_bbox,
            'obj_category': obj_category,
            'obj_caption': obj_caption,
            'mesh_info': mesh_info,
            'point_pairs': saved_point_pairs
        }
    
    def _load_mv_object_data_local(self, scene_id, object_id, num_frames=8):
        """本地模式：v3 固定帧 + ANNOTATE depth/info"""
        # 1. 检查mesh文件
        mesh_path = f"{MV_RECON_ROOT}/{scene_id}/{object_id}/mesh.glb"
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        mesh_url = f"/data/mv_mesh/{scene_id}/{object_id}/mesh.glb"
        
        # 2. 加载 v3 frame_selection_metadata（固定帧列表）
        v3_meta_path = f"{MV_SAM_V3_ROOT}/{scene_id}/{object_id}/frame_selection_metadata.json"
        if not os.path.exists(v3_meta_path):
            raise FileNotFoundError(f"v3 metadata not found: {v3_meta_path}")
        
        with open(v3_meta_path) as f:
            v3_metadata = json.load(f)
        
        if not isinstance(v3_metadata, list) or not v3_metadata:
            raise ValueError(f"Empty v3 metadata: {v3_meta_path}")
        
        # 3. 加载 info.json 获取相机参数（优先 ANNOTATE，fallback LASA1M, v2）
        info = None
        for base in (ANNOTATE_ROOT, LASA1M_ROOT, f"{DATA_ROOT}/LASA1M_v2"):
            info_path = f"{base}/{scene_id}/{object_id}/info.json"
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                break
        if info is None:
            raise FileNotFoundError(f"info.json not found for {scene_id}/{object_id}")
        
        timestamp_to_info = {item.get('timestamp'): item for item in info if item.get('timestamp')}
        depth_width, depth_height = 512, 384
        
        # 4. 构建帧数据（使用 v3 固定帧，最多 num_frames 帧）
        total_frames = len(v3_metadata)
        selected_entries = v3_metadata[:num_frames] if len(v3_metadata) > num_frames else v3_metadata
        
        frames = []
        for entry in selected_entries:
            idx = entry.get("index", 0)
            ts = entry.get("timestamp")
            if ts is None:
                continue
            
            # v3 RGB image
            rgb_path = f"{MV_SAM_V3_ROOT}/{scene_id}/{object_id}/images/{idx}.png"
            if not os.path.exists(rgb_path):
                continue
            
            # 获取图像尺寸（缓存）
            image_width, image_height = self._get_image_size(rgb_path)
            
            # 相机参数
            frame_info = timestamp_to_info.get(ts)
            if not frame_info:
                timestamps = list(timestamp_to_info.keys())
                if timestamps:
                    closest_ts = min(timestamps, key=lambda t: abs(t - ts))
                    frame_info = timestamp_to_info[closest_ts]
            if not frame_info:
                continue
            
            K = frame_info.get('gt_depth_K')
            RT = frame_info.get('gt_RT')
            
            K_scaled = None
            if K and image_width > 0 and image_height > 0:
                scale_x = image_width / depth_width
                scale_y = image_height / depth_height
                K_scaled = [
                    [K[0][0] * scale_x, K[0][1], K[0][2] * scale_x],
                    [K[1][0], K[1][1] * scale_y, K[1][2] * scale_y],
                    [K[2][0], K[2][1], K[2][2]]
                ]
            
            # v3 mask
            mask_path = f"{MV_SAM_V3_ROOT}/{scene_id}/{object_id}/{object_id}/{idx}.png"
            mask_exists = os.path.exists(mask_path)
            
            # mask比例
            mask_ratio = entry.get("mask_ratio", 0.0)
            if mask_ratio == 0.0 and mask_exists:
                from PIL import Image
                with Image.open(mask_path) as mask_img:
                    mask_array = np.array(mask_img)
                    if len(mask_array.shape) == 3:
                        mask_array = mask_array[:, :, 0]
                    mask_pixels = np.sum(mask_array > 128)
                    total_pixels = mask_array.shape[0] * mask_array.shape[1]
                    mask_ratio = mask_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # depth（优先 ANNOTATE，fallback 旧路径）
            depth_url = None
            annotate_depth = f"{ANNOTATE_ROOT}/{scene_id}/{object_id}/depth/{idx}.png"
            if os.path.exists(annotate_depth):
                depth_url = f"/data/annotate_depth/{scene_id}/{object_id}/{idx}"
            else:
                # fallback: 旧 LASA1M gt 路径
                gt_depth = f"{LASA1M_ROOT}/{scene_id}/{object_id}/gt/{ts}/depth.png"
                if os.path.exists(gt_depth):
                    depth_url = f"/data/depth/{scene_id}/{object_id}/{ts}"
            
            frames.append({
                'frame_id': str(ts),
                'frame_index': idx,
                'rgb_url': f"/data/v3_image/{scene_id}/{object_id}/{idx}.png",
                'mask_url': f"/data/v3_mask/{scene_id}/{object_id}/{idx}.png" if mask_exists else None,
                'depth_url': depth_url,
                'mask_ratio': mask_ratio,
                'camera_intrinsics': {
                    'K': K_scaled,
                    'K_original': K,
                    'width': image_width,
                    'height': image_height
                },
                'camera_extrinsics': RT
            })
        
        for i, frame in enumerate(frames):
            frame['frame_index'] = i
        
        # 5. 加载已有的world_pose（检测 mesh 是否比对齐数据更新）
        world_pose = None
        aligned_path = f"{MV_ALIGNED_ROOT}/{scene_id}/{object_id}/world_pose.npy"
        mesh_mtime = os.path.getmtime(mesh_path)
        alignment_stale = False
        if os.path.exists(aligned_path):
            aligned_mtime = os.path.getmtime(aligned_path)
            if mesh_mtime > aligned_mtime:
                alignment_stale = True
                print(f"[load_mv_object_data] ⚠️ mesh 已更新 (mesh={mesh_mtime:.0f} > aligned={aligned_mtime:.0f})，丢弃旧对齐数据: {scene_id}/{object_id}")
            else:
                world_pose = np.load(aligned_path).tolist()
        
        # 6. 读取GT bbox + category（缓存 instances.json）
        gt_bbox = None
        obj_category = None
        obj_caption = None
        instances = self._get_instances(scene_id)
        for inst in instances:
            if inst.get('id') == object_id:
                gt_bbox = {
                    'position': inst.get('position'),
                    'scale': inst.get('scale'),
                    'R': inst.get('R'),
                    'corners': inst.get('corners')
                }
                obj_category = inst.get('category')
                obj_caption = inst.get('caption')
                break
        
        # 7. mesh_info（缓存 trimesh.load，但 mesh 更新时需清缓存）
        if alignment_stale:
            cache_key = f"{scene_id}/{object_id}"
            _mesh_info_cache.pop(cache_key, None)
        mesh_info = self._get_mesh_info(scene_id, object_id)
        
        # 8. 加载已保存的关键点对（mesh 更新时丢弃旧点对）
        saved_point_pairs = []
        pairs_path = f"{MV_ALIGNED_ROOT}/{scene_id}/{object_id}/point_pairs.json"
        if os.path.exists(pairs_path) and not alignment_stale:
            with open(pairs_path, 'r') as f:
                saved_point_pairs = json.load(f)
        
        return {
            'scene_id': scene_id,
            'object_id': object_id,
            'mesh_url': mesh_url,
            'mesh_path': mesh_path,
            'world_pose': world_pose,
            'frames': frames,
            'total_frames': total_frames,
            'gt_bbox': gt_bbox,
            'obj_category': obj_category,
            'obj_caption': obj_caption,
            'mesh_info': mesh_info,
            'point_pairs': saved_point_pairs
        }
    
    def _serve_depth_png(self, depth_path):
        """读取 depth PNG 并返回 JSON（共用逻辑）"""
        if not os.path.exists(depth_path):
            self.send_error(404, f"Depth not found: {depth_path}")
            return
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
    
    def serve_data_file(self, path):
        """服务数据文件"""
        parts = path.split('/')
        print(f"[serve_data_file] path={path}, parts={parts}")
        
        if parts[0] == 'v3_image':
            # /data/v3_image/{scene_id}/{object_id}/{idx}.png
            scene_id, object_id, filename = parts[1], parts[2], parts[3]
            file_path = f"{MV_SAM_V3_ROOT}/{scene_id}/{object_id}/images/{filename}"
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = f.read()
                self.send_binary(data, 'image/png')
            else:
                self.send_error(404, f"v3 image not found: {file_path}")
        
        elif parts[0] == 'v3_mask':
            # /data/v3_mask/{scene_id}/{object_id}/{idx}.png
            scene_id, object_id, filename = parts[1], parts[2], parts[3]
            file_path = f"{MV_SAM_V3_ROOT}/{scene_id}/{object_id}/{object_id}/{filename}"
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = f.read()
                self.send_binary(data, 'image/png')
            else:
                self.send_error(404, f"v3 mask not found: {file_path}")
        
        elif parts[0] == 'annotate_depth':
            # /data/annotate_depth/{scene_id}/{object_id}/{idx}
            scene_id, object_id, idx = parts[1], parts[2], parts[3]
            depth_path = f"{ANNOTATE_ROOT}/{scene_id}/{object_id}/depth/{idx}.png"
            self._serve_depth_png(depth_path)
        
        elif parts[0] == 'image':
            # /data/image/{scene_id}/{object_id}/{subdir}/{filename} (legacy)
            try:
                scene_id, object_id, subdir, filename = parts[1], parts[2], parts[3], parts[4]
                file_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/{subdir}/{filename}"
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
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = f.read()
                self.send_binary(data, 'model/gltf-binary')
            else:
                self.send_error(404, f"Mesh not found: {file_path}")
        
        elif parts[0] == 'depth':
            # /data/depth/{scene_id}/{object_id}/{frame_id} (legacy)
            scene_id, object_id, frame_id = parts[1], parts[2], parts[3]
            depth_path = f"{LASA1M_ROOT}/{scene_id}/{object_id}/gt/{frame_id}/depth.png"
            self._serve_depth_png(depth_path)
        
        elif parts[0] == 'tar_rgb':
            # /data/tar_rgb/{scene_id}/{object_id}/{idx}.webp
            scene_id, object_id, filename = parts[1], parts[2], parts[3]
            obj_dir = tos_client.get_object_dir(scene_id, object_id)
            file_path = os.path.join(obj_dir, filename.replace('.webp', '.rgb.webp'))
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = f.read()
                self.send_binary(data, 'image/webp')
            else:
                self.send_error(404, f"tar rgb not found: {file_path}")
        
        elif parts[0] == 'tar_mask':
            # /data/tar_mask/{scene_id}/{object_id}/{idx}.png
            scene_id, object_id, filename = parts[1], parts[2], parts[3]
            obj_dir = tos_client.get_object_dir(scene_id, object_id)
            file_path = os.path.join(obj_dir, filename.replace('.png', '.mask.png'))
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = f.read()
                self.send_binary(data, 'image/png')
            else:
                self.send_error(404, f"tar mask not found: {file_path}")
        
        elif parts[0] == 'tar_depth':
            # /data/tar_depth/{scene_id}/{object_id}/{idx}
            scene_id, object_id, idx_str = parts[1], parts[2], parts[3]
            obj_dir = tos_client.get_object_dir(scene_id, object_id)
            depth_path = os.path.join(obj_dir, f"{idx_str}.depth.png")
            self._serve_depth_png(depth_path)
        
        elif parts[0] == 'tar_mesh':
            # /data/tar_mesh/{scene_id}/{object_id}/mesh.glb
            scene_id, object_id, filename = parts[1], parts[2], parts[3]
            obj_dir = tos_client.get_object_dir(scene_id, object_id)
            file_path = os.path.join(obj_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = f.read()
                self.send_binary(data, 'model/gltf-binary')
            else:
                self.send_error(404, f"tar mesh not found: {file_path}")
        
        elif parts[0] == 'tar_render':
            # /data/tar_render/{scene_id}/{object_id}/{filename}.webp
            scene_id, object_id, filename = parts[1], parts[2], parts[3]
            obj_dir = tos_client.get_object_dir(scene_id, object_id)
            file_path = os.path.join(obj_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = f.read()
                self.send_binary(data, 'image/webp')
            else:
                self.send_error(404, f"tar render not found: {file_path}")
        
        else:
            self.send_error(404, f"Unknown data type: {parts[0]}")


def warmup_cache():
    """启动时预热缓存（委托给 _build_mv_objects_cache）"""
    global _mv_objects_cache, _mv_objects_cache_time
    print("[warmup_cache] 正在预热缓存...")
    start_time = time.time()
    
    handler = MVDataAPIHandler.__new__(MVDataAPIHandler)
    _mv_objects_cache = handler._build_mv_objects_cache()
    _mv_objects_cache_time = time.time()
    
    print(f"[warmup_cache] 缓存预热完成，耗时 {time.time() - start_time:.2f}s，共 {len(_mv_objects_cache)} 个物体")


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
