/**
 * API 请求封装 - 自动带 token 认证
 */

// 获取 API 基础路径
const getApiBasePath = () => {
  const basePath = (import.meta as any).env?.BASE_URL || '/';
  return basePath.replace(/\/$/, '');
};

// Token 存储
const TOKEN_KEY = 'annotation_token';
const USER_KEY = 'annotation_user';

export interface User {
  id: number;
  username: string;
  role: 'admin' | 'annotator';
  batch_size: number;
  is_active: number;
}

export interface UserStats {
  total: number;
  today: number;
  week: number;
  active_scenes: number;
  completed_scenes: number;
}

export interface AuthResponse {
  user: User;
  stats: UserStats;
  active_scenes: string[];
}

// 获取存储的 token
export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

// 设置 token
export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

// 清除 token
export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

// 获取存储的用户信息
export function getStoredUser(): User | null {
  const userStr = localStorage.getItem(USER_KEY);
  if (userStr) {
    try {
      return JSON.parse(userStr);
    } catch {
      return null;
    }
  }
  return null;
}

// 设置用户信息
export function setStoredUser(user: User): void {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

// 获取认证头（用于直接 fetch 调用）
export function getAuthHeaders(): Record<string, string> {
  const token = getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

// 带认证的 fetch 封装
export async function apiFetch<T = any>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const basePath = getApiBasePath();
  const url = path.startsWith('http') ? path : `${basePath}${path}`;
  
  const token = getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {}),
  };
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  const response = await fetch(url, {
    ...options,
    headers,
  });
  
  // 处理 401 未授权
  if (response.status === 401) {
    clearToken();
    // 触发重新登录
    window.dispatchEvent(new CustomEvent('auth:logout'));
    throw new Error('Unauthorized');
  }
  
  const data = await response.json();
  
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  
  return data;
}

// 登录
export async function login(username: string, password: string): Promise<{ success: boolean; user?: User; error?: string }> {
  try {
    const basePath = getApiBasePath();
    const response = await fetch(`${basePath}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    
    const data = await response.json();
    
    if (data.success && data.token) {
      setToken(data.token);
      setStoredUser(data.user);
      return { success: true, user: data.user };
    }
    
    return { success: false, error: data.error || 'Login failed' };
  } catch (e) {
    return { success: false, error: String(e) };
  }
}

// 登出
export async function logout(): Promise<void> {
  try {
    await apiFetch('/api/auth/logout', { method: 'POST' });
  } catch {
    // 忽略错误
  }
  clearToken();
}

// 获取当前用户信息
export async function getCurrentUser(): Promise<AuthResponse | null> {
  try {
    const data = await apiFetch<AuthResponse>('/api/auth/me');
    if (data.user) {
      setStoredUser(data.user);
    }
    return data;
  } catch {
    return null;
  }
}

// 检查是否已登录
export function isLoggedIn(): boolean {
  return !!getToken();
}

// 检查是否是管理员
export function isAdmin(): boolean {
  const user = getStoredUser();
  return user?.role === 'admin';
}
