#!/usr/bin/env python3
"""
数据库管理模块 - 用户认证、场景分配、标注日志

使用 SQLite 存储：
- users: 用户信息
- scene_assignments: 场景分配记录
- annotation_logs: 标注日志
"""

import os
import sqlite3
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from contextlib import contextmanager

# 数据库文件路径
DB_PATH = os.path.join(os.path.dirname(__file__), 'annotation.db')

# 内存中的 token 映射 (token -> user_id)
_token_store: Dict[str, int] = {}


@contextmanager
def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """初始化数据库表结构"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'annotator',
                batch_size INTEGER NOT NULL DEFAULT 5,
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
        ''')
        
        # 场景分配表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scene_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                scene_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                assigned_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(scene_id, status) 
            )
        ''')
        # 注意：UNIQUE(scene_id, status) 允许同一场景被不同用户完成后重新分配
        # 但同一时刻只能有一个 active 分配
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_assignments_user ON scene_assignments(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_assignments_scene ON scene_assignments(scene_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_assignments_status ON scene_assignments(status)')
        
        # 标注日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS annotation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                scene_id TEXT NOT NULL,
                object_id TEXT NOT NULL,
                category TEXT,
                action TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_user ON annotation_logs(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_date ON annotation_logs(created_at)')
        
        conn.commit()
        
        # 创建默认管理员账号（如果不存在）
        cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',))
        if not cursor.fetchone():
            create_user('admin', 'admin123', 'admin', batch_size=999)
            print('[db] 已创建默认管理员账号: admin / admin123')


def hash_password(password: str) -> str:
    """密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username: str, password: str, role: str = 'annotator', batch_size: int = 5) -> Optional[int]:
    """创建用户"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO users (username, password_hash, role, batch_size, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, hash_password(password), role, batch_size, datetime.now().isoformat()))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None  # 用户名已存在


def verify_user(username: str, password: str) -> Optional[Dict]:
    """验证用户登录"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, role, batch_size, is_active
            FROM users WHERE username = ? AND password_hash = ?
        ''', (username, hash_password(password)))
        row = cursor.fetchone()
        if row and row['is_active']:
            return dict(row)
        return None


def get_user_by_id(user_id: int) -> Optional[Dict]:
    """根据ID获取用户"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, role, batch_size, is_active, created_at
            FROM users WHERE id = ?
        ''', (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_users() -> List[Dict]:
    """列出所有用户"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, role, batch_size, is_active, created_at
            FROM users ORDER BY id
        ''')
        return [dict(row) for row in cursor.fetchall()]


def update_user(user_id: int, **kwargs) -> bool:
    """更新用户信息"""
    allowed_fields = {'password', 'role', 'batch_size', 'is_active'}
    updates = []
    values = []
    
    for key, value in kwargs.items():
        if key in allowed_fields:
            if key == 'password':
                updates.append('password_hash = ?')
                values.append(hash_password(value))
            else:
                updates.append(f'{key} = ?')
                values.append(value)
    
    if not updates:
        return False
    
    values.append(user_id)
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(f'''
            UPDATE users SET {', '.join(updates)} WHERE id = ?
        ''', values)
        conn.commit()
        return cursor.rowcount > 0


# ========== Token 管理 ==========

def create_token(user_id: int) -> str:
    """创建登录 token"""
    token = secrets.token_hex(32)
    _token_store[token] = user_id
    return token


def verify_token(token: str) -> Optional[int]:
    """验证 token，返回 user_id"""
    return _token_store.get(token)


def revoke_token(token: str):
    """撤销 token"""
    _token_store.pop(token, None)


def get_user_by_token(token: str) -> Optional[Dict]:
    """根据 token 获取用户信息"""
    user_id = verify_token(token)
    if user_id:
        return get_user_by_id(user_id)
    return None


# ========== 场景分配 ==========

def get_user_active_scenes(user_id: int) -> List[str]:
    """获取用户当前活跃的场景列表"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT scene_id FROM scene_assignments
            WHERE user_id = ? AND status = 'active'
            ORDER BY assigned_at
        ''', (user_id,))
        return [row['scene_id'] for row in cursor.fetchall()]


def get_unassigned_scenes(all_scenes: List[str], limit: int) -> List[str]:
    """获取未分配的场景（不在任何用户的 active 分配中）"""
    with get_db() as conn:
        cursor = conn.cursor()
        # 获取所有 active 状态的场景
        cursor.execute('''
            SELECT scene_id FROM scene_assignments WHERE status = 'active'
        ''')
        assigned = {row['scene_id'] for row in cursor.fetchall()}
        
        # 过滤出未分配的场景
        unassigned = [s for s in all_scenes if s not in assigned]
        return unassigned[:limit]


def assign_scenes_to_user(user_id: int, scene_ids: List[str]) -> int:
    """将场景分配给用户"""
    if not scene_ids:
        return 0
    
    with get_db() as conn:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        count = 0
        for scene_id in scene_ids:
            try:
                cursor.execute('''
                    INSERT INTO scene_assignments (user_id, scene_id, status, assigned_at)
                    VALUES (?, ?, 'active', ?)
                ''', (user_id, scene_id, now))
                count += 1
            except sqlite3.IntegrityError:
                # 场景已被分配（active状态），跳过
                pass
        conn.commit()
        return count


def mark_scene_completed(user_id: int, scene_id: str):
    """标记场景为已完成"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE scene_assignments
            SET status = 'completed', completed_at = ?
            WHERE user_id = ? AND scene_id = ? AND status = 'active'
        ''', (datetime.now().isoformat(), user_id, scene_id))
        conn.commit()


def get_all_assignments(user_id: Optional[int] = None, status: Optional[str] = None) -> List[Dict]:
    """获取场景分配列表"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = '''
            SELECT sa.*, u.username
            FROM scene_assignments sa
            JOIN users u ON sa.user_id = u.id
            WHERE 1=1
        '''
        params = []
        if user_id:
            query += ' AND sa.user_id = ?'
            params.append(user_id)
        if status:
            query += ' AND sa.status = ?'
            params.append(status)
        query += ' ORDER BY sa.assigned_at DESC'
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


# ========== 标注日志 ==========

def log_annotation(user_id: int, scene_id: str, object_id: str, category: str, action: str):
    """记录标注日志"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO annotation_logs (user_id, scene_id, object_id, category, action, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, scene_id, object_id, category, action, datetime.now().isoformat()))
        conn.commit()


def get_user_stats(user_id: int) -> Dict:
    """获取用户标注统计（按唯一物体计数，同一物体多次保存只算1个）"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 总完成数（按唯一物体去重）
        cursor.execute('''
            SELECT COUNT(DISTINCT scene_id || '_' || object_id) as total 
            FROM annotation_logs WHERE user_id = ?
        ''', (user_id,))
        total = cursor.fetchone()['total']
        
        # 今日完成数（按唯一物体去重，以首次标注时间为准）
        today = datetime.now().date().isoformat()
        cursor.execute('''
            SELECT COUNT(*) as today_count FROM (
                SELECT scene_id, object_id, MIN(created_at) as first_annotated
                FROM annotation_logs WHERE user_id = ?
                GROUP BY scene_id, object_id
                HAVING first_annotated >= ?
            )
        ''', (user_id, today))
        today_count = cursor.fetchone()['today_count']
        
        # 本周完成数（按唯一物体去重，以首次标注时间为准）
        week_start = (datetime.now() - timedelta(days=datetime.now().weekday())).date().isoformat()
        cursor.execute('''
            SELECT COUNT(*) as week_count FROM (
                SELECT scene_id, object_id, MIN(created_at) as first_annotated
                FROM annotation_logs WHERE user_id = ?
                GROUP BY scene_id, object_id
                HAVING first_annotated >= ?
            )
        ''', (user_id, week_start))
        week_count = cursor.fetchone()['week_count']
        
        # 活跃场景数
        cursor.execute('''
            SELECT COUNT(*) as active_scenes FROM scene_assignments
            WHERE user_id = ? AND status = 'active'
        ''', (user_id,))
        active_scenes = cursor.fetchone()['active_scenes']
        
        # 已完成场景数
        cursor.execute('''
            SELECT COUNT(*) as completed_scenes FROM scene_assignments
            WHERE user_id = ? AND status = 'completed'
        ''', (user_id,))
        completed_scenes = cursor.fetchone()['completed_scenes']
        
        return {
            'total': total,
            'today': today_count,
            'week': week_count,
            'active_scenes': active_scenes,
            'completed_scenes': completed_scenes
        }


def get_all_user_stats() -> List[Dict]:
    """获取所有用户的统计信息"""
    users = list_users()
    result = []
    for user in users:
        stats = get_user_stats(user['id'])
        result.append({
            **user,
            **stats
        })
    return result


def get_annotation_logs(user_id: Optional[int] = None, 
                        date_from: Optional[str] = None,
                        date_to: Optional[str] = None,
                        limit: int = 100,
                        offset: int = 0) -> List[Dict]:
    """获取标注日志"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = '''
            SELECT al.*, u.username
            FROM annotation_logs al
            JOIN users u ON al.user_id = u.id
            WHERE 1=1
        '''
        params = []
        
        if user_id:
            query += ' AND al.user_id = ?'
            params.append(user_id)
        if date_from:
            query += ' AND al.created_at >= ?'
            params.append(date_from)
        if date_to:
            query += ' AND al.created_at <= ?'
            params.append(date_to)
        
        query += ' ORDER BY al.created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_logs_count(user_id: Optional[int] = None,
                   date_from: Optional[str] = None,
                   date_to: Optional[str] = None) -> int:
    """获取日志总数"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = 'SELECT COUNT(*) as count FROM annotation_logs WHERE 1=1'
        params = []
        
        if user_id:
            query += ' AND user_id = ?'
            params.append(user_id)
        if date_from:
            query += ' AND created_at >= ?'
            params.append(date_from)
        if date_to:
            query += ' AND created_at <= ?'
            params.append(date_to)
        
        cursor.execute(query, params)
        return cursor.fetchone()['count']


# 初始化数据库
init_db()
