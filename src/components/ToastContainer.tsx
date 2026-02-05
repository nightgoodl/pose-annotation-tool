/**
 * Toast 容器组件 - 显示非阻塞提示
 */

import { useToast, type ToastType } from '../hooks/useToast';

const typeStyles: Record<ToastType, string> = {
  success: 'bg-green-600 border-green-500',
  warning: 'bg-yellow-600 border-yellow-500',
  error: 'bg-red-600 border-red-500',
  info: 'bg-blue-600 border-blue-500'
};

const typeIcons: Record<ToastType, string> = {
  success: '✓',
  warning: '⚠',
  error: '✕',
  info: 'ℹ'
};

export function ToastContainer() {
  const { toasts, removeToast } = useToast();
  
  if (toasts.length === 0) return null;
  
  return (
    <div className="fixed bottom-4 right-4 z-[9999] flex flex-col gap-2 pointer-events-none">
      {toasts.map(toast => (
        <div
          key={toast.id}
          className={`
            ${typeStyles[toast.type]}
            border rounded-lg px-4 py-2 shadow-lg
            text-white text-sm font-medium
            flex items-center gap-2
            pointer-events-auto cursor-pointer
            animate-slide-in
          `}
          onClick={() => removeToast(toast.id)}
          style={{
            animation: 'slideIn 0.2s ease-out'
          }}
        >
          <span className="text-lg">{typeIcons[toast.type]}</span>
          <span>{toast.message}</span>
        </div>
      ))}
      
      <style>{`
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateX(100%);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
      `}</style>
    </div>
  );
}

export default ToastContainer;
