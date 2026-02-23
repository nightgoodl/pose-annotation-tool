/**
 * 管理员后台页面
 * 
 * 功能：
 * - 用户管理：创建/编辑/禁用用户
 * - 分配概览：查看场景分配状态
 * - 统计面板：各用户完成量统计
 */

import { useEffect, useState, useCallback } from 'react';
import { apiFetch, isLoggedIn, isAdmin, logout, getStoredUser, type User } from '../utils/api';
import { LoginPage } from './LoginPage';

interface UserWithStats {
  id: number;
  username: string;
  role: 'admin' | 'annotator';
  batch_size: number;
  is_active: number;
  created_at: string;
  total: number;
  today: number;
  week: number;
  active_scenes: number;
  completed_scenes: number;
}

interface AnnotationLog {
  id: number;
  user_id: number;
  username: string;
  scene_id: string;
  object_id: string;
  category: string;
  action: string;
  created_at: string;
}

interface SceneStatus {
  [scene_id: string]: {
    status: 'unassigned' | 'active' | 'completed';
    user: string | null;
    user_id?: number;
  };
}

type TabType = 'users' | 'scenes' | 'stats' | 'logs';

export function AdminPage() {
  const [isAuthenticated, setIsAuthenticated] = useState(isLoggedIn());
  const [currentUser, setCurrentUser] = useState<User | null>(getStoredUser());
  const [activeTab, setActiveTab] = useState<TabType>('users');
  
  // 用户管理
  const [users, setUsers] = useState<UserWithStats[]>([]);
  const [loadingUsers, setLoadingUsers] = useState(false);
  
  // 场景状态
  const [scenes, setScenes] = useState<SceneStatus>({});
  const [loadingScenes, setLoadingScenes] = useState(false);
  
  // 日志
  const [logs, setLogs] = useState<AnnotationLog[]>([]);
  const [logsTotal, setLogsTotal] = useState(0);
  const [logsPage, setLogsPage] = useState(1);
  const [loadingLogs, setLoadingLogs] = useState(false);
  
  // 创建用户表单
  const [showCreateUser, setShowCreateUser] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newBatchSize, setNewBatchSize] = useState(5);
  const [createError, setCreateError] = useState<string | null>(null);
  
  // 编辑用户
  const [editingUser, setEditingUser] = useState<UserWithStats | null>(null);
  const [editPassword, setEditPassword] = useState('');
  const [editBatchSize, setEditBatchSize] = useState(5);
  
  // 加载用户列表
  const loadUsers = useCallback(async () => {
    setLoadingUsers(true);
    try {
      const data = await apiFetch<{ users: UserWithStats[] }>('/api/admin/users');
      setUsers(data.users || []);
    } catch (e) {
      console.error('Failed to load users:', e);
    }
    setLoadingUsers(false);
  }, []);
  
  // 加载场景状态
  const loadScenes = useCallback(async () => {
    setLoadingScenes(true);
    try {
      const data = await apiFetch<{ scenes: SceneStatus }>('/api/admin/scenes');
      setScenes(data.scenes || {});
    } catch (e) {
      console.error('Failed to load scenes:', e);
    }
    setLoadingScenes(false);
  }, []);
  
  // 加载日志
  const loadLogs = useCallback(async (page: number = 1) => {
    setLoadingLogs(true);
    try {
      const data = await apiFetch<{ logs: AnnotationLog[]; total: number }>(
        `/api/admin/logs?limit=50&offset=${(page - 1) * 50}`
      );
      setLogs(data.logs || []);
      setLogsTotal(data.total || 0);
      setLogsPage(page);
    } catch (e) {
      console.error('Failed to load logs:', e);
    }
    setLoadingLogs(false);
  }, []);
  
  // 初始加载
  useEffect(() => {
    if (isAuthenticated && isAdmin()) {
      loadUsers();
    }
  }, [isAuthenticated, loadUsers]);
  
  // Tab 切换时加载数据
  useEffect(() => {
    if (!isAuthenticated || !isAdmin()) return;
    
    if (activeTab === 'users') loadUsers();
    else if (activeTab === 'scenes') loadScenes();
    else if (activeTab === 'logs') loadLogs(1);
    else if (activeTab === 'stats') loadUsers();
  }, [activeTab, isAuthenticated, loadUsers, loadScenes, loadLogs]);
  
  // 创建用户
  const handleCreateUser = async () => {
    setCreateError(null);
    try {
      await apiFetch('/api/admin/users', {
        method: 'POST',
        body: JSON.stringify({
          username: newUsername,
          password: newPassword,
          batch_size: newBatchSize,
          role: 'annotator'
        })
      });
      setShowCreateUser(false);
      setNewUsername('');
      setNewPassword('');
      setNewBatchSize(5);
      loadUsers();
    } catch (e) {
      setCreateError(String(e));
    }
  };
  
  // 更新用户
  const handleUpdateUser = async () => {
    if (!editingUser) return;
    
    try {
      const updates: any = { batch_size: editBatchSize };
      if (editPassword) updates.password = editPassword;
      
      await apiFetch(`/api/admin/users/${editingUser.id}`, {
        method: 'POST',
        body: JSON.stringify(updates)
      });
      setEditingUser(null);
      setEditPassword('');
      loadUsers();
    } catch (e) {
      console.error('Failed to update user:', e);
    }
  };
  
  // 禁用/启用用户
  const handleToggleUser = async (user: UserWithStats) => {
    try {
      await apiFetch(`/api/admin/users/${user.id}`, {
        method: 'POST',
        body: JSON.stringify({ is_active: user.is_active ? 0 : 1 })
      });
      loadUsers();
    } catch (e) {
      console.error('Failed to toggle user:', e);
    }
  };
  
  // 登录成功
  const handleLoginSuccess = () => {
    setIsAuthenticated(true);
    setCurrentUser(getStoredUser());
  };
  
  // 登出
  const handleLogout = async () => {
    await logout();
    setIsAuthenticated(false);
    setCurrentUser(null);
  };
  
  // 未登录
  if (!isAuthenticated) {
    return <LoginPage onLoginSuccess={handleLoginSuccess} />;
  }
  
  // 非管理员
  if (!isAdmin()) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="bg-gray-800 p-8 rounded-xl text-center">
          <h1 className="text-xl text-white mb-4">权限不足</h1>
          <p className="text-gray-400 mb-4">您没有管理员权限</p>
          <button onClick={handleLogout} className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-white">
            返回登录
          </button>
        </div>
      </div>
    );
  }
  
  // 场景统计
  const sceneStats = {
    total: Object.keys(scenes).length,
    unassigned: Object.values(scenes).filter(s => s.status === 'unassigned').length,
    active: Object.values(scenes).filter(s => s.status === 'active').length,
    completed: Object.values(scenes).filter(s => s.status === 'completed').length,
  };
  
  return (
    <div className="min-h-screen bg-gray-900">
      {/* 顶部导航 */}
      <header className="h-14 bg-gray-800 border-b border-gray-700 flex items-center px-6">
        <h1 className="text-white font-semibold text-lg">管理员后台</h1>
        <div className="ml-auto flex items-center gap-4">
          <span className="text-gray-400 text-sm">
            {currentUser?.username}
          </span>
          <button onClick={handleLogout} className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-gray-300 text-sm">
            登出
          </button>
        </div>
      </header>
      
      {/* Tab 导航 */}
      <div className="border-b border-gray-700 px-6">
        <div className="flex gap-1">
          {[
            { key: 'users', label: '用户管理' },
            { key: 'scenes', label: '场景分配' },
            { key: 'stats', label: '统计概览' },
            { key: 'logs', label: '标注日志' },
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key as TabType)}
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.key
                  ? 'text-blue-400 border-blue-400'
                  : 'text-gray-400 border-transparent hover:text-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>
      
      {/* 内容区 */}
      <div className="p-6">
        {/* 用户管理 */}
        {activeTab === 'users' && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-white text-lg font-medium">用户列表</h2>
              <button
                onClick={() => setShowCreateUser(true)}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-white text-sm"
              >
                + 创建用户
              </button>
            </div>
            
            {loadingUsers ? (
              <div className="text-gray-400 text-center py-8">加载中...</div>
            ) : (
              <table className="w-full">
                <thead>
                  <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                    <th className="pb-3 font-medium">用户名</th>
                    <th className="pb-3 font-medium">角色</th>
                    <th className="pb-3 font-medium">批次大小</th>
                    <th className="pb-3 font-medium">今日完成</th>
                    <th className="pb-3 font-medium">总完成</th>
                    <th className="pb-3 font-medium">活跃场景</th>
                    <th className="pb-3 font-medium">状态</th>
                    <th className="pb-3 font-medium">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map(user => (
                    <tr key={user.id} className="border-b border-gray-800 text-sm">
                      <td className="py-3 text-white">{user.username}</td>
                      <td className="py-3">
                        <span className={`px-2 py-0.5 rounded text-xs ${
                          user.role === 'admin' ? 'bg-purple-600 text-white' : 'bg-gray-600 text-gray-200'
                        }`}>
                          {user.role === 'admin' ? '管理员' : '标注员'}
                        </span>
                      </td>
                      <td className="py-3 text-gray-300">{user.batch_size}</td>
                      <td className="py-3 text-green-400">{user.today}</td>
                      <td className="py-3 text-blue-400">{user.total}</td>
                      <td className="py-3 text-yellow-400">{user.active_scenes}</td>
                      <td className="py-3">
                        <span className={`px-2 py-0.5 rounded text-xs ${
                          user.is_active ? 'bg-green-600 text-white' : 'bg-red-600 text-white'
                        }`}>
                          {user.is_active ? '启用' : '禁用'}
                        </span>
                      </td>
                      <td className="py-3">
                        <div className="flex gap-2">
                          <button
                            onClick={() => {
                              setEditingUser(user);
                              setEditBatchSize(user.batch_size);
                              setEditPassword('');
                            }}
                            className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-gray-300 text-xs"
                          >
                            编辑
                          </button>
                          {user.role !== 'admin' && (
                            <button
                              onClick={() => handleToggleUser(user)}
                              className={`px-2 py-1 rounded text-xs ${
                                user.is_active
                                  ? 'bg-red-700 hover:bg-red-600 text-white'
                                  : 'bg-green-700 hover:bg-green-600 text-white'
                              }`}
                            >
                              {user.is_active ? '禁用' : '启用'}
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
        
        {/* 场景分配 */}
        {activeTab === 'scenes' && (
          <div>
            <div className="flex items-center gap-6 mb-6">
              <h2 className="text-white text-lg font-medium">场景分配状态</h2>
              <div className="flex gap-4 text-sm">
                <span className="text-gray-400">总计: <span className="text-white">{sceneStats.total}</span></span>
                <span className="text-gray-400">未分配: <span className="text-gray-300">{sceneStats.unassigned}</span></span>
                <span className="text-gray-400">进行中: <span className="text-yellow-400">{sceneStats.active}</span></span>
                <span className="text-gray-400">已完成: <span className="text-green-400">{sceneStats.completed}</span></span>
              </div>
              <button onClick={loadScenes} className="ml-auto px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-gray-300 text-sm">
                刷新
              </button>
            </div>
            
            {loadingScenes ? (
              <div className="text-gray-400 text-center py-8">加载中...</div>
            ) : (
              <div className="grid grid-cols-4 gap-2 max-h-[600px] overflow-auto">
                {Object.entries(scenes).sort().map(([sceneId, info]) => (
                  <div
                    key={sceneId}
                    className={`p-3 rounded border ${
                      info.status === 'completed' ? 'bg-green-900/30 border-green-700' :
                      info.status === 'active' ? 'bg-yellow-900/30 border-yellow-700' :
                      'bg-gray-800 border-gray-700'
                    }`}
                  >
                    <div className="text-white text-sm font-mono truncate">{sceneId}</div>
                    <div className="text-xs mt-1">
                      {info.status === 'completed' && <span className="text-green-400">已完成 ({info.user})</span>}
                      {info.status === 'active' && <span className="text-yellow-400">进行中 ({info.user})</span>}
                      {info.status === 'unassigned' && <span className="text-gray-500">未分配</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        
        {/* 统计概览 */}
        {activeTab === 'stats' && (
          <div>
            <h2 className="text-white text-lg font-medium mb-4">用户统计</h2>
            
            {loadingUsers ? (
              <div className="text-gray-400 text-center py-8">加载中...</div>
            ) : (
              <div className="grid grid-cols-3 gap-4">
                {users.filter(u => u.role !== 'admin').map(user => (
                  <div key={user.id} className="bg-gray-800 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-white font-medium">{user.username}</span>
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        user.is_active ? 'bg-green-600' : 'bg-red-600'
                      } text-white`}>
                        {user.is_active ? '在线' : '离线'}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <div className="text-gray-500">今日</div>
                        <div className="text-2xl text-green-400 font-bold">{user.today}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">本周</div>
                        <div className="text-2xl text-blue-400 font-bold">{user.week}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">总计</div>
                        <div className="text-xl text-white font-medium">{user.total}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">活跃场景</div>
                        <div className="text-xl text-yellow-400 font-medium">{user.active_scenes}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        
        {/* 标注日志 */}
        {activeTab === 'logs' && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-white text-lg font-medium">标注日志</h2>
              <span className="text-gray-400 text-sm">共 {logsTotal} 条</span>
            </div>
            
            {loadingLogs ? (
              <div className="text-gray-400 text-center py-8">加载中...</div>
            ) : (
              <>
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                      <th className="pb-3 font-medium">时间</th>
                      <th className="pb-3 font-medium">用户</th>
                      <th className="pb-3 font-medium">场景</th>
                      <th className="pb-3 font-medium">物体</th>
                      <th className="pb-3 font-medium">操作</th>
                      <th className="pb-3 font-medium">分类</th>
                    </tr>
                  </thead>
                  <tbody>
                    {logs.map(log => (
                      <tr key={log.id} className="border-b border-gray-800 text-sm">
                        <td className="py-2 text-gray-400">{log.created_at.replace('T', ' ').slice(0, 19)}</td>
                        <td className="py-2 text-white">{log.username}</td>
                        <td className="py-2 text-gray-300 font-mono text-xs">{log.scene_id}</td>
                        <td className="py-2 text-gray-300 font-mono text-xs truncate max-w-[200px]">{log.object_id}</td>
                        <td className="py-2">
                          <span className={`px-2 py-0.5 rounded text-xs ${
                            log.action === 'save' ? 'bg-green-600' :
                            log.action === 'invalid' ? 'bg-red-600' :
                            'bg-orange-600'
                          } text-white`}>
                            {log.action}
                          </span>
                        </td>
                        <td className="py-2 text-gray-400">{log.category}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                
                {/* 分页 */}
                {logsTotal > 50 && (
                  <div className="flex justify-center gap-2 mt-4">
                    <button
                      onClick={() => loadLogs(logsPage - 1)}
                      disabled={logsPage === 1}
                      className="px-3 py-1 bg-gray-700 text-white rounded text-sm disabled:opacity-50"
                    >
                      上一页
                    </button>
                    <span className="text-gray-400 text-sm px-4 py-1">
                      {logsPage} / {Math.ceil(logsTotal / 50)}
                    </span>
                    <button
                      onClick={() => loadLogs(logsPage + 1)}
                      disabled={logsPage >= Math.ceil(logsTotal / 50)}
                      className="px-3 py-1 bg-gray-700 text-white rounded text-sm disabled:opacity-50"
                    >
                      下一页
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
      
      {/* 创建用户弹窗 */}
      {showCreateUser && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-xl p-6 w-full max-w-md">
            <h3 className="text-white text-lg font-medium mb-4">创建用户</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-gray-400 text-sm mb-1">用户名</label>
                <input
                  type="text"
                  value={newUsername}
                  onChange={(e) => setNewUsername(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                  placeholder="请输入用户名"
                />
              </div>
              <div>
                <label className="block text-gray-400 text-sm mb-1">密码</label>
                <input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                  placeholder="请输入密码"
                />
              </div>
              <div>
                <label className="block text-gray-400 text-sm mb-1">批次大小（每次领取场景数）</label>
                <input
                  type="number"
                  value={newBatchSize}
                  onChange={(e) => setNewBatchSize(parseInt(e.target.value) || 5)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                  min={1}
                  max={50}
                />
              </div>
              
              {createError && (
                <div className="text-red-400 text-sm">{createError}</div>
              )}
            </div>
            
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowCreateUser(false)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
              >
                取消
              </button>
              <button
                onClick={handleCreateUser}
                disabled={!newUsername || !newPassword}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-white disabled:opacity-50"
              >
                创建
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* 编辑用户弹窗 */}
      {editingUser && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-xl p-6 w-full max-w-md">
            <h3 className="text-white text-lg font-medium mb-4">编辑用户: {editingUser.username}</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-gray-400 text-sm mb-1">新密码（留空不修改）</label>
                <input
                  type="password"
                  value={editPassword}
                  onChange={(e) => setEditPassword(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                  placeholder="留空不修改密码"
                />
              </div>
              <div>
                <label className="block text-gray-400 text-sm mb-1">批次大小</label>
                <input
                  type="number"
                  value={editBatchSize}
                  onChange={(e) => setEditBatchSize(parseInt(e.target.value) || 5)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                  min={1}
                  max={50}
                />
              </div>
            </div>
            
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setEditingUser(null)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
              >
                取消
              </button>
              <button
                onClick={handleUpdateUser}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-white"
              >
                保存
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AdminPage;
