/**
 * 控制面板 - 点对管理、对齐操作、导出
 */

import React from 'react';
import { 
  Trash2, 
  Play, 
  RotateCcw, 
  Save, 
  Eye, 
  EyeOff,
  Crosshair,
  Layers
} from 'lucide-react';
import { useAnnotationStore } from '../stores/annotationStore';

export function ControlPanel() {
  const workflowState = useAnnotationStore((state) => state.workflowState);
  const category = useAnnotationStore((state) => state.category);
  const pointPairs = useAnnotationStore((state) => state.pointPairs);
  const selectedPairId = useAnnotationStore((state) => state.selectedPairId);
  const pendingLocalPoint = useAnnotationStore((state) => state.pendingLocalPoint);
  const pendingWorldPoint = useAnnotationStore((state) => state.pendingWorldPoint);
  const calculatedPose = useAnnotationStore((state) => state.calculatedPose);
  const calculatedScale = useAnnotationStore((state) => state.calculatedScale);
  const alignmentError = useAnnotationStore((state) => state.alignmentError);
  const currentIoU = useAnnotationStore((state) => state.currentIoU);
  const showGhostWireframe = useAnnotationStore((state) => state.showGhostWireframe);
  const maskOpacity = useAnnotationStore((state) => state.maskOpacity);
  
  const removePointPair = useAnnotationStore((state) => state.removePointPair);
  const selectPointPair = useAnnotationStore((state) => state.selectPointPair);
  const clearPointPairs = useAnnotationStore((state) => state.clearPointPairs);
  const runAlignment = useAnnotationStore((state) => state.runAlignment);
  const resetAlignment = useAnnotationStore((state) => state.resetAlignment);
  const exportAnnotation = useAnnotationStore((state) => state.exportAnnotation);
  const savePose = useAnnotationStore((state) => state.savePose);
  const setShowGhostWireframe = useAnnotationStore((state) => state.setShowGhostWireframe);
  const setMaskOpacity = useAnnotationStore((state) => state.setMaskOpacity);
  
  const handleSave = async () => {
    const result = await savePose();
    if (result.success) {
      alert(`Pose已保存到: ${result.pose_path}`);
    } else {
      alert(`保存失败: ${result.error}`);
    }
  };
  
  const handleExport = () => {
    const result = exportAnnotation();
    if (result) {
      console.log('Exported annotation:', result);
      // 复制到剪贴板
      navigator.clipboard.writeText(JSON.stringify(result, null, 2))
        .then(() => alert('标注结果已复制到剪贴板'))
        .catch((err) => console.error('复制失败:', err));
    }
  };
  
  const canAlign = pointPairs.length >= 3;
  const isAnnotating = workflowState === 'annotation';
  
  return (
    <div className="h-full bg-gray-800 rounded-lg p-4 flex flex-col gap-4 overflow-y-auto">
      {/* 状态信息 */}
      <div className="bg-gray-700 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-gray-300 mb-2">状态</h3>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-400">分类:</span>
            <span className={`font-medium ${
              category === 'valid' ? 'text-green-400' :
              category === 'fixed' ? 'text-blue-400' :
              category === 'invalid' ? 'text-red-400' :
              'text-gray-400'
            }`}>
              {category === 'valid' ? '有效' :
               category === 'fixed' ? '固定装' :
               category === 'invalid' ? '无效' : '未分类'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">点对数:</span>
            <span className="text-white">{pointPairs.length}</span>
          </div>
          {/* IoU评价 - 主要评价标准 */}
          {currentIoU !== null && (
            <div className="flex justify-between">
              <span className="text-gray-400">IoU:</span>
              <span className={`font-bold ${currentIoU > 0.5 ? 'text-green-400' : currentIoU > 0.3 ? 'text-yellow-400' : 'text-red-400'}`}>
                {(currentIoU * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {calculatedPose && pointPairs.length >= 3 && (
            <>
              <div className="flex justify-between">
                <span className="text-gray-400">缩放:</span>
                <span className="text-white">{calculatedScale.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">RMSE:</span>
                <span className="text-gray-300 text-xs">
                  {alignmentError.toFixed(4)}m
                </span>
              </div>
            </>
          )}
          {/* 对齐质量评价 - 基于IoU */}
          {currentIoU !== null && (
            <div className="mt-2 p-2 bg-gray-600 rounded text-xs">
              <div className="text-gray-300 mb-1">对齐质量:</div>
              <div className={`font-bold ${currentIoU > 0.5 ? 'text-green-400' : currentIoU > 0.3 ? 'text-yellow-400' : 'text-red-400'}`}>
                {currentIoU > 0.5 ? '✓ 优秀' : currentIoU > 0.3 ? '⚠ 一般' : '✗ 较差'}
              </div>
            </div>
          )}
          {calculatedPose && currentIoU === null && (
            <div className="mt-2 p-2 bg-gray-600 rounded text-xs">
              <div className="text-gray-300 mb-1">对齐状态:</div>
              <div className="text-yellow-400">
                {pointPairs.length < 3 ? '使用初始pose' : '计算IoU中...'}
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* 待匹配点 */}
      {isAnnotating && (pendingLocalPoint || pendingWorldPoint) && (
        <div className="bg-orange-900/30 border border-orange-700 rounded-lg p-3">
          <h3 className="text-sm font-semibold text-orange-400 mb-2 flex items-center gap-2">
            <Crosshair className="w-4 h-4" />
            待匹配点
          </h3>
          <div className="space-y-1 text-xs">
            {pendingLocalPoint && (
              <div className="text-orange-300">
                模型点: ({pendingLocalPoint.x.toFixed(3)}, {pendingLocalPoint.y.toFixed(3)}, {pendingLocalPoint.z.toFixed(3)})
              </div>
            )}
            {pendingWorldPoint && (
              <div className="text-orange-300">
                世界点: ({pendingWorldPoint.point.x.toFixed(3)}, {pendingWorldPoint.point.y.toFixed(3)}, {pendingWorldPoint.point.z.toFixed(3)})
              </div>
            )}
            <div className="text-gray-400 mt-1">
              {pendingLocalPoint && !pendingWorldPoint && '请在左侧图像上点击对应位置'}
              {!pendingLocalPoint && pendingWorldPoint && '请在右侧模型上点击对应位置'}
            </div>
          </div>
        </div>
      )}
      
      {/* 点对列表 */}
      {isAnnotating && (
        <div className="flex-1 min-h-0">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
              <Layers className="w-4 h-4" />
              点对列表
            </h3>
            {pointPairs.length > 0 && (
              <button
                onClick={clearPointPairs}
                className="text-xs text-red-400 hover:text-red-300"
              >
                清空
              </button>
            )}
          </div>
          
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {pointPairs.length === 0 ? (
              <div className="text-xs text-gray-500 text-center py-4">
                暂无点对，请在两侧视图中点击对应特征点
              </div>
            ) : (
              pointPairs.map((pair, index) => (
                <div
                  key={pair.id}
                  className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
                    pair.id === selectedPairId
                      ? 'bg-blue-600/30 border border-blue-500'
                      : 'bg-gray-700 hover:bg-gray-600'
                  }`}
                  onClick={() => selectPointPair(pair.id)}
                >
                  <span className="text-xs font-bold text-gray-400 w-4">
                    {index + 1}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="text-xs text-gray-300 truncate">
                      L: ({pair.localPoint.x.toFixed(2)}, {pair.localPoint.y.toFixed(2)}, {pair.localPoint.z.toFixed(2)})
                    </div>
                    <div className="text-xs text-gray-400 truncate">
                      W: ({pair.worldPoint.x.toFixed(2)}, {pair.worldPoint.y.toFixed(2)}, {pair.worldPoint.z.toFixed(2)})
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removePointPair(pair.id);
                    }}
                    className="p-1 hover:bg-red-600/50 rounded"
                  >
                    <Trash2 className="w-3 h-3 text-red-400" />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      )}
      
      {/* 操作按钮 */}
      {isAnnotating && (
        <div className="space-y-2">
          <button
            onClick={() => {
              console.log('Running alignment with', pointPairs.length, 'point pairs');
              runAlignment();
            }}
            disabled={!canAlign}
            className={`w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg font-medium transition-colors ${
              canAlign
                ? 'bg-green-600 hover:bg-green-500 text-white'
                : 'bg-gray-600 text-gray-400 cursor-not-allowed'
            }`}
          >
            <Play className="w-4 h-4" />
            {canAlign ? '对齐 (Align)' : `需要 ${3 - pointPairs.length} 对点`}
          </button>
          
          <button
            onClick={resetAlignment}
            className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg font-medium bg-gray-600 hover:bg-gray-500 text-white transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            重置
          </button>
          
          <button
            onClick={handleSave}
            className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors"
          >
            <Save className="w-4 h-4" />
            保存Pose
          </button>
        </div>
      )}
      
      {/* 显示设置 */}
      <div className="border-t border-gray-700 pt-4">
        <h3 className="text-sm font-semibold text-gray-300 mb-3">显示设置</h3>
        
        <div className="space-y-3">
          <button
            onClick={() => setShowGhostWireframe(!showGhostWireframe)}
            className="w-full flex items-center justify-between py-2 px-3 rounded bg-gray-700 hover:bg-gray-600 transition-colors"
          >
            <span className="text-sm text-gray-300">幽灵线框</span>
            {showGhostWireframe ? (
              <Eye className="w-4 h-4 text-green-400" />
            ) : (
              <EyeOff className="w-4 h-4 text-gray-500" />
            )}
          </button>
          
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-xs text-gray-400">Mask 透明度</span>
              <span className="text-xs text-gray-300">{Math.round(maskOpacity * 100)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={maskOpacity}
              onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
            />
          </div>
        </div>
      </div>
      
      {/* 快捷键提示 */}
      <div className="border-t border-gray-700 pt-4 text-xs text-gray-500">
        <div className="font-medium mb-1">快捷键:</div>
        <div>点击模型/图像 - 添加点</div>
        <div>至少3对点后可对齐</div>
      </div>
    </div>
  );
}

export default ControlPanel;
