/**
 * 守门人分类面板 (Gatekeeper Classification Panel)
 * 
 * 功能：
 * - 在标点操作前强制用户对物体进行分类
 * - 三个选项：有效(Valid)、固定装(Fixed)、无效(Invalid)
 * - 选择后解锁或跳过标注流程
 * 
 * 修改：改为顶部按钮栏，不遮挡视图
 */

import { CheckCircle, Building, XCircle } from 'lucide-react';
import { useAnnotationStore } from '../stores/annotationStore';

export function ClassificationPanel() {
  const workflowState = useAnnotationStore((state) => state.workflowState);
  const classifyAsValid = useAnnotationStore((state) => state.classifyAsValid);
  const classifyAsFixed = useAnnotationStore((state) => state.classifyAsFixed);
  const classifyAsInvalid = useAnnotationStore((state) => state.classifyAsInvalid);
  
  if (workflowState !== 'classification') {
    return null;
  }
  
  return (
    <div className="absolute top-12 left-0 right-0 z-30 bg-yellow-900/90 backdrop-blur-sm border-b border-yellow-600 px-4 py-3">
      <div className="flex items-center justify-between max-w-4xl mx-auto">
        <div className="text-yellow-200 text-sm font-medium">
          请先分类此物体：
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={classifyAsValid}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-green-600 hover:bg-green-500 text-white font-medium transition-colors"
          >
            <CheckCircle className="w-4 h-4" />
            <span>有效 (A)</span>
          </button>
          
          <button
            onClick={classifyAsFixed}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium transition-colors"
          >
            <Building className="w-4 h-4" />
            <span>固定装 (B)</span>
          </button>
          
          <button
            onClick={classifyAsInvalid}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white font-medium transition-colors"
          >
            <XCircle className="w-4 h-4" />
            <span>无效 (C)</span>
          </button>
        </div>
      </div>
    </div>
  );
}

export default ClassificationPanel;
