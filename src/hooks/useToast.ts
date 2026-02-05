/**
 * Toast Hook - 非阻塞提示系统
 */

import { useState, useCallback } from 'react';

export type ToastType = 'success' | 'warning' | 'error' | 'info';

export interface Toast {
  id: string;
  message: string;
  type: ToastType;
  duration: number;
}

// 全局 Toast 状态管理
let globalSetToasts: React.Dispatch<React.SetStateAction<Toast[]>> | null = null;
let toastIdCounter = 0;

export function showGlobalToast(
  message: string,
  type: ToastType = 'info',
  duration: number = 2000
): void {
  const id = `toast_${++toastIdCounter}`;
  const toast: Toast = { id, message, type, duration };
  
  if (globalSetToasts) {
    globalSetToasts(prev => [...prev, toast]);
    
    // 自动移除
    setTimeout(() => {
      if (globalSetToasts) {
        globalSetToasts(prev => prev.filter(t => t.id !== id));
      }
    }, duration);
  } else {
    // 降级到 console
    console.log(`[Toast ${type}] ${message}`);
  }
}

export function useToast() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  
  // 注册全局 setter
  globalSetToasts = setToasts;
  
  const addToast = useCallback((
    message: string,
    type: ToastType = 'info',
    duration: number = 2000
  ) => {
    showGlobalToast(message, type, duration);
  }, []);
  
  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);
  
  return {
    toasts,
    addToast,
    removeToast
  };
}
