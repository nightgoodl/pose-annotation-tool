/**
 * 多视角标注 - 控制面板组件
 */

import { useMVAnnotationStore } from '../stores/mvAnnotationStore';

interface MVControlPanelProps {
  onSaveAndNext?: () => Promise<void>;
  onSkipAndNext?: () => Promise<void>;
}

export function MVControlPanel({ onSaveAndNext, onSkipAndNext }: MVControlPanelProps) {
  const workflowState = useMVAnnotationStore((state) => state.workflowState);
  const pointPairs = useMVAnnotationStore((state) => state.pointPairs);
  
  const resetAlignment = useMVAnnotationStore((state) => state.resetAlignment);
  const runAlignment = useMVAnnotationStore((state) => state.runAlignment);
  const isSavingNext = useMVAnnotationStore((state) => state.isSavingNext);
  const remainingCount = useMVAnnotationStore((state) => state.remainingCount);
  
  return (
    <div className="h-full bg-gray-800 rounded-lg p-4 flex flex-col gap-4 overflow-y-auto">
      {/* 标题 */}
      <div className="text-white font-semibold border-b border-gray-600 pb-2">
        多视角对齐控制
      </div>
      
      {/* 点数提示 */}
      {workflowState === 'annotation' && (
        <div className="text-xs text-gray-500 text-center">
          点数: {pointPairs.length}{pointPairs.length >= 3 ? ' (自动对齐)' : ''}
        </div>
      )}
      
      {/* 操作按钮 */}
      {workflowState === 'annotation' && (
        <div className="space-y-2">
          {/* 重置关键点 */}
          {pointPairs.length > 0 && (
            <button
              onClick={resetAlignment}
              disabled={isSavingNext}
              className="w-full px-4 py-2 rounded font-medium transition-colors bg-yellow-700 hover:bg-yellow-600 text-white disabled:bg-gray-600 disabled:text-gray-400 disabled:cursor-not-allowed"
            >
              重置关键点 ({pointPairs.length})
            </button>
          )}
          
          {/* RANSAC 对齐（手动触发，≥5点可用） */}
          <button
            onClick={() => runAlignment(true)}
            disabled={isSavingNext || pointPairs.length < 5}
            className={`w-full px-4 py-2 rounded font-medium transition-colors ${
              isSavingNext || pointPairs.length < 5
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-blue-700 hover:bg-blue-600 text-white'
            }`}
          >
            RANSAC 对齐{pointPairs.length < 5 ? ` (需${5 - pointPairs.length}点)` : ''}
          </button>
          
          {/* 保存并处理下一个 - 需要>=3点 */}
          {onSaveAndNext && (
            <button
              onClick={onSaveAndNext}
              disabled={isSavingNext || pointPairs.length < 3}
              className={`w-full px-4 py-3 rounded font-semibold transition-colors ${
                isSavingNext || pointPairs.length < 3
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-500 text-white'
              }`}
            >
              {isSavingNext ? '处理中...' : pointPairs.length < 3 ? `保存并下一个 (需${3 - pointPairs.length}点)` : '保存并处理下一个'}
              {remainingCount !== null && !isSavingNext && pointPairs.length >= 3 && (
                <span className="ml-2 px-1.5 py-0.5 bg-white/20 rounded-full text-xs">
                  剩余 {remainingCount}
                </span>
              )}
            </button>
          )}
          
          {/* 放弃并处理下一个 */}
          {onSkipAndNext && (
            <button
              onClick={onSkipAndNext}
              disabled={isSavingNext}
              className={`w-full px-4 py-2 rounded font-medium transition-colors ${
                isSavingNext
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : 'bg-gray-600 hover:bg-gray-500 text-gray-200'
              }`}
            >
              {isSavingNext ? '处理中...' : '放弃并处理下一个'}
            </button>
          )}
        </div>
      )}
      
      {/* 快捷键提示 */}
      <div className="border-t border-gray-600 pt-3 text-xs text-gray-500">
        <div className="font-medium mb-1">快捷键:</div>
        <div>Ctrl+Enter - 保存并下一个</div>
        <div>Ctrl+Shift+Enter - 放弃并下一个</div>
      </div>
    </div>
  );
}

export default MVControlPanel;
