#!/usr/bin/env python3
"""
从本地 tar 文件目录生成 manifest.json
扫描每个 tar 包中的 meta.json，提取物体元数据

用法:
    python generate_manifest.py [tar_dir] [output_path]
    
默认:
    tar_dir: /root/csz/data_partcrafter/LASA1M_WDS_TAR_MESH
    output_path: ./manifest.json
"""

import os
import sys
import json
import tarfile
import io
from collections import defaultdict


def extract_meta_from_tar(tar_path):
    """从 tar 包中读取 meta.json（不解压整个包）"""
    with tarfile.open(tar_path, 'r') as tf:
        for member in tf.getmembers():
            if member.name == 'meta.json':
                f = tf.extractfile(member)
                if f:
                    return json.load(io.TextIOWrapper(f, encoding='utf-8'))
    return None


def generate_manifest(tar_dir, output_path):
    """扫描 tar 目录并生成 manifest"""
    tar_files = sorted([f for f in os.listdir(tar_dir) if f.endswith('.tar')])
    print(f"[generate_manifest] 发现 {len(tar_files)} 个 tar 文件")
    
    objects = []
    scenes = defaultdict(int)
    errors = []
    
    for i, tar_name in enumerate(tar_files):
        if (i + 1) % 1000 == 0:
            print(f"[generate_manifest] 进度: {i + 1}/{len(tar_files)}")
        
        # 从文件名解析 scene_id 和 object_id
        base = tar_name.replace('.tar', '')
        parts = base.split('_', 1)
        if len(parts) != 2:
            errors.append(f"无法解析文件名: {tar_name}")
            continue
        
        scene_id, object_id = parts[0], parts[1]
        
        # 尝试从 tar 中读取 meta.json
        tar_path = os.path.join(tar_dir, tar_name)
        meta = None
        try:
            meta = extract_meta_from_tar(tar_path)
        except Exception as e:
            errors.append(f"读取 {tar_name} 失败: {e}")
        
        entry = {
            'scene_id': scene_id,
            'object_id': object_id,
        }
        
        if meta:
            entry['num_frames'] = meta.get('num_frames', 0)
            entry['category'] = meta.get('category', '')
            entry['caption'] = meta.get('caption', '')
        else:
            # 无法读取 meta，仅记录基本信息
            entry['num_frames'] = 0
            entry['category'] = ''
            entry['caption'] = ''
        
        objects.append(entry)
        scenes[scene_id] += 1
    
    # 按 scene_id + object_id 排序
    objects.sort(key=lambda x: (x['scene_id'], x['object_id']))
    
    manifest = {
        'version': 1,
        'total_objects': len(objects),
        'total_scenes': len(scenes),
        'objects': objects,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"[generate_manifest] 完成:")
    print(f"  物体数: {len(objects)}")
    print(f"  场景数: {len(scenes)}")
    print(f"  错误数: {len(errors)}")
    print(f"  输出: {output_path}")
    
    if errors:
        print(f"[generate_manifest] 错误列表 (前10个):")
        for e in errors[:10]:
            print(f"  - {e}")


if __name__ == '__main__':
    tar_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/csz/data_partcrafter/LASA1M_WDS_TAR_MESH"
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "manifest.json")
    generate_manifest(tar_dir, output_path)
