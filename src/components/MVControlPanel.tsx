/**
 * 多视角标注 - 控制面板组件
 */

import { useMVAnnotationStore } from '../stores/mvAnnotationStore';
import { showGlobalToast } from '../hooks/useToast';

export function MVControlPanel() {
  const workflowState = useMVAnnotationStore((state) => state.workflowState);
  const category = useMVAnnotationStore((state) => state.category);
  const pointPairs = useMVAnnotationStore((state) => state.pointPairs);
  const calculatedScale = useMVAnnotationStore((state) => state.calculatedScale);
  const alignmentError = useMVAnnotationStore((state) => state.alignmentError);
  const frameIoUs = useMVAnnotationStore((state) => state.frameIoUs);
  const averageIoU = useMVAnnotationStore((state) => state.averageIoU);
  const showGhostWireframe = useMVAnnotationStore((state) => state.showGhostWireframe);
  const maskOpacity = useMVAnnotationStore((state) => state.maskOpacity);
  const useNvdiffrastRender = useMVAnnotationStore((state) => state.useNvdiffrastRender);
  
  const removePointPair = useMVAnnotationStore((state) => state.removePointPair);
  const clearPointPairs = useMVAnnotationStore((state) => state.clearPointPairs);
  const runAlignment = useMVAnnotationStore((state) => state.runAlignment);
  const resetAlignment = useMVAnnotationStore((state) => state.resetAlignment);
  const setShowGhostWireframe = useMVAnnotationStore((state) => state.setShowGhostWireframe);
  const setMaskOpacity = useMVAnnotationStore((state) => state.setMaskOpacity);
  const setUseNvdiffrastRender = useMVAnnotationStore((state) => state.setUseNvdiffrastRender);
  const savePose = useMVAnnotationStore((state) => state.savePose);
  const currentInput = useMVAnnotationStore((state) => state.currentInput);
  
  // 统计各帧的点数
  const framePointCounts = new Map<string, number>();
  pointPairs.forEach(p => {
    framePointCounts.set(p.frame_id, (framePointCounts.get(p.frame_id) || 0) + 1);
  });
  
  const handleSave = async () => {
    const result = await savePose();
    if (result.success) {
      showGlobalToast('保存成功', 'success', 1500);
    } else {
      showGlobalToast(`保存失败: ${result.error}`, 'error', 3000);
    }
  };
  
  return (
    <div className="h-full bg-gray-800 rounded-lg p-4 flex flex-col gap-4 overflow-y-auto">
      {/* 标题 */}
      <div className="text-white font-semibold border-b border-gray-600 pb-2">
        多视角对齐控制
      </div>
      
      {/* 状态信息 */}
      <div className="bg-gray-700 rounded p-3">
        <div className="text-sm text-gray-300 mb-2">状态</div>
        <div className="text-xs text-gray-400 space-y-1">
          <div>工作流: <span className="text-white">{
            workflowState === 'classification' ? '分类中' :
            workflowState === 'annotation' ? '标注中' : '审核中'
          }</span></div>
          <div>状态: <span className="text-white">{
            category === 'invalid' ? '无效数据' :
            category === 'valid' ? '标注中' :
            category === 'fixed' ? '已对齐' : '待分类'
          }</span></div>
        </div>
      </div>
      
      {/* 显示控制 - 移到上方便于快速访问 */}
      <div className="bg-gray-700 rounded p-3">
        <div className="text-sm text-gray-300 mb-2">显示设置</div>
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <span className="w-16">Mask</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={maskOpacity}
              onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
              className="flex-1"
            />
            <span className="w-8 text-right">{(maskOpacity * 100).toFixed(0)}%</span>
          </div>
          <label className="flex items-center gap-2 text-xs text-gray-400">
            <input
              type="checkbox"
              checked={showGhostWireframe}
              onChange={(e) => setShowGhostWireframe(e.target.checked)}
              className="rounded"
            />
            显示mesh投影 (仅缩略图)
          </label>
          {showGhostWireframe && (
            <label className="flex items-center gap-2 text-xs text-gray-400 ml-4">
              <input
                type="checkbox"
                checked={useNvdiffrastRender}
                onChange={(e) => setUseNvdiffrastRender(e.target.checked)}
                className="rounded"
              />
              GPU渲染 (nvdiffrast)
            </label>
          )}
        </div>
      </div>
      
      {/* 对齐计算 */}
      {workflowState === 'annotation' && (
        <div className="bg-gray-700 rounded p-3">
          <div className="text-sm text-gray-300 mb-2">对齐计算</div>
          
          <div className="flex gap-2 mb-2">
            <button
              onClick={() => runAlignment(false)}
              disabled={pointPairs.length < 3}
              className="flex-1 px-3 py-2 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded text-sm"
            >
              普通对齐
            </button>
            <button
              onClick={() => runAlignment(true)}
              disabled={pointPairs.length < 5}
              className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded text-sm"
              title="RANSAC需要至少5个点"
            >
              RANSAC
            </button>
            <button
              onClick={resetAlignment}
              className="px-3 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded text-sm"
            >
              重置
            </button>
          </div>
          <div className="text-xs text-gray-500 mb-2">
            点数: {pointPairs.length} (普通≥3, RANSAC≥5)
          </div>
          
          {alignmentError > 0 && (
            <div className="text-xs text-gray-400 space-y-1 bg-gray-800 rounded p-2">
              <div>缩放: <span className="text-white">{calculatedScale.toFixed(4)}</span></div>
              <div>误差: <span className="text-white">{alignmentError.toFixed(4)}</span></div>
              <div>平均IoU: <span className={`font-bold ${
                averageIoU > 0.5 ? 'text-green-400' : averageIoU > 0.3 ? 'text-yellow-400' : averageIoU > 0 ? 'text-red-400' : 'text-white'
              }`}>
                {averageIoU > 0 ? `${(averageIoU * 100).toFixed(1)}%` : '计算中...'}
              </span></div>
              {frameIoUs.size > 0 && (
                <div className="mt-1 pt-1 border-t border-gray-700">
                  <div className="text-gray-500 mb-0.5">各帧IoU:</div>
                  <div className="flex flex-wrap gap-1">
                    {Array.from(frameIoUs.entries()).map(([fid, iou]) => (
                      <span key={fid} className={`text-xs px-1 py-0.5 rounded ${
                        iou > 0.5 ? 'bg-green-600' : iou > 0.3 ? 'bg-yellow-600' : 'bg-red-600'
                      } text-white`}>
                        ...{fid.slice(-4)}: {(iou * 100).toFixed(0)}%
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
      
      {/* 关键点对列表 */}
      {workflowState === 'annotation' && (
        <div className="bg-gray-700 rounded p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-300">
              关键点对 ({pointPairs.length})
            </span>
            {pointPairs.length > 0 && (
              <button
                onClick={clearPointPairs}
                className="text-xs text-red-400 hover:text-red-300"
              >
                清空
              </button>
            )}
          </div>
          
          {/* 帧统计 */}
          {framePointCounts.size > 0 && (
            <div className="text-xs text-gray-400 mb-2 flex flex-wrap gap-1">
              {Array.from(framePointCounts.entries()).map(([fid, count]) => (
                <span key={fid} className="bg-gray-600 px-1.5 py-0.5 rounded">
                  ...{fid.slice(-4)}: {count}
                </span>
              ))}
            </div>
          )}
          
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {pointPairs.map((pair, index) => (
              <div key={pair.id} className="flex items-center justify-between bg-gray-600 rounded px-2 py-1 text-xs">
                <span className="text-white truncate flex-1">
                  #{index + 1} [F:{pair.frame_id.slice(-4)}]
                </span>
                <button
                  onClick={() => removePointPair(pair.id)}
                  className="text-red-400 hover:text-red-300 ml-2 shrink-0"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
          
          {pointPairs.length === 0 && (
            <div className="text-xs text-gray-500 text-center py-4">
              先在模型上点击，再在图像上点击对应位置
            </div>
          )}
        </div>
      )}
      
      {/* 保存按钮 */}
      {workflowState === 'annotation' && pointPairs.length >= 3 && (
        <button
          onClick={handleSave}
          className="px-4 py-3 bg-green-600 hover:bg-green-500 text-white rounded font-semibold"
        >
          保存Pose
        </button>
      )}
    </div>
  );
}

export default MVControlPanel;
