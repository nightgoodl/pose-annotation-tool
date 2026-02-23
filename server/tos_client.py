#!/usr/bin/env python3
"""
TOS (Tencent Object Storage) 客户端工具
从 tos://ycj-data-backup/LASA1M_WDS_TAR_MESH/ 下载并管理 tar 数据包
"""

import os
import shutil
import subprocess
import tarfile
import threading

# 配置
TOSUTIL_PATH = "/root/tosutil"
TOS_BUCKET = "tos://ycj-data-backup"
TOS_PREFIX = "LASA1M_WDS_TAR_MESH/LASA1M_WDS_TAR_MESH"
TOS_CACHE_DIR = os.environ.get("TOS_CACHE_DIR", "/tmp/tos_tar_cache")

# 下载锁，防止同一物体并发下载
_download_locks = {}
_locks_lock = threading.Lock()


def _get_lock(key):
    """获取指定key的锁（线程安全）"""
    with _locks_lock:
        if key not in _download_locks:
            _download_locks[key] = threading.Lock()
        return _download_locks[key]


def get_object_dir(scene_id, object_id):
    """获取物体解压后的本地目录路径"""
    return os.path.join(TOS_CACHE_DIR, f"{scene_id}_{object_id}")


def is_object_cached(scene_id, object_id):
    """检查物体是否已缓存在本地"""
    obj_dir = get_object_dir(scene_id, object_id)
    # 检查目录存在且包含 meta.json（解压完成的标志）
    return os.path.isdir(obj_dir) and os.path.exists(os.path.join(obj_dir, "meta.json"))


def ensure_object_cached(scene_id, object_id):
    """
    确保物体数据已缓存到本地。如果不存在则从 TOS 下载并解压。
    
    Returns:
        str: 解压后的本地目录路径
    
    Raises:
        RuntimeError: 下载或解压失败
    """
    obj_dir = get_object_dir(scene_id, object_id)
    
    if is_object_cached(scene_id, object_id):
        return obj_dir
    
    lock_key = f"{scene_id}_{object_id}"
    lock = _get_lock(lock_key)
    
    with lock:
        # double-check
        if is_object_cached(scene_id, object_id):
            return obj_dir
        
        tar_name = f"{scene_id}_{object_id}.tar"
        tos_path = f"{TOS_BUCKET}/{TOS_PREFIX}/{tar_name}"
        
        os.makedirs(TOS_CACHE_DIR, exist_ok=True)
        local_tar_path = os.path.join(TOS_CACHE_DIR, tar_name)
        
        print(f"[tos_client] 开始下载: {tos_path}")
        
        # 下载 tar 文件
        result = subprocess.run(
            [TOSUTIL_PATH, "cp", tos_path, local_tar_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            # 清理可能的残留文件
            if os.path.exists(local_tar_path):
                os.remove(local_tar_path)
            raise RuntimeError(
                f"TOS 下载失败: {tos_path}\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        
        if not os.path.exists(local_tar_path):
            raise RuntimeError(f"TOS 下载后文件不存在: {local_tar_path}")
        
        print(f"[tos_client] 下载完成，开始解压: {local_tar_path}")
        
        # 解压到目标目录
        os.makedirs(obj_dir, exist_ok=True)
        with tarfile.open(local_tar_path, 'r') as tf:
            tf.extractall(path=obj_dir)
        
        # 删除 tar 文件（只保留解压后的内容）
        os.remove(local_tar_path)
        
        print(f"[tos_client] 解压完成: {obj_dir}")
        return obj_dir


def delete_object_cache(scene_id, object_id):
    """
    删除物体的本地缓存目录（标注完成后调用以释放磁盘空间）
    
    Returns:
        bool: 是否成功删除
    """
    obj_dir = get_object_dir(scene_id, object_id)
    if os.path.isdir(obj_dir):
        shutil.rmtree(obj_dir, ignore_errors=True)
        print(f"[tos_client] 已删除缓存: {obj_dir}")
        return True
    return False
