/**
 * 部署分析页面
 * 配置和运行 LLM 部署评估任务
 */

import React from 'react'
import { Loader2, CheckCircle, XCircle, StopCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { BaseCard } from '@/components/common/BaseCard'
import { PageHeader } from '@/components/ui/page-header'
import { DeploymentAnalysisPanel } from '@/components/ConfigPanel/DeploymentAnalysis'
import { useWorkbench } from '@/contexts/WorkbenchContext'

export const DeploymentAnalysis: React.FC = () => {
  const { analysis, topology } = useWorkbench()

  // 运行状态
  const loading = analysis.deploymentAnalysisData?.loading ?? false
  const errorMsg = analysis.deploymentAnalysisData?.errorMsg ?? null
  const searchProgress = analysis.deploymentAnalysisData?.searchProgress
  const onCancelEvaluation = analysis.deploymentAnalysisData?.onCancelEvaluation

  return (
    <div className="h-full w-full bg-gradient-to-b from-gray-50 to-white flex flex-col">
      {/* 标题栏 */}
      <PageHeader title="部署分析" />

      {/* 主内容区 */}
      <div className="flex-1 overflow-auto p-8 bg-gradient-to-b from-gray-50 to-white">
        <div>
          {/* 配置面板 */}
          <div>
            <DeploymentAnalysisPanel
              topology={topology.topology}
              onTrafficResultChange={analysis.setTrafficResult}
              onAnalysisDataChange={analysis.setDeploymentAnalysisData}
              history={analysis.analysisHistory}
              onAddToHistory={analysis.handleAddToHistory}
              onDeleteHistory={analysis.handleDeleteHistory}
              onClearHistory={analysis.handleClearHistory}
            />
          </div>

          {/* 下方：运行状态卡片 */}
          {(loading || errorMsg || (searchProgress && searchProgress.stage !== 'idle' && searchProgress.stage !== 'completed')) && (
            <BaseCard title="运行状态" className="mt-4">
              {/* 错误提示 */}
              {errorMsg && (
                <Alert variant="destructive" className="mb-4">
                  <XCircle className="h-4 w-4" />
                  <AlertTitle>分析失败</AlertTitle>
                  <AlertDescription>{errorMsg}</AlertDescription>
                </Alert>
              )}

              {/* 搜索进度 */}
              {loading && searchProgress && searchProgress.stage !== 'idle' && (
                <div>
                  {/* 阶段 1: 生成候选方案 */}
                  <div className="flex items-center gap-3 mb-3">
                    {searchProgress.stage === 'generating' ? (
                      <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                    ) : (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    )}
                    <span className="text-sm">
                      生成候选方案: <span className="font-semibold">{searchProgress.totalCandidates}</span> 个
                    </span>
                  </div>

                  {/* 阶段 2: 后端评估 */}
                  {searchProgress.stage !== 'generating' && (
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-3">
                          {searchProgress.stage === 'evaluating' ? (
                            <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                          ) : (
                            <CheckCircle className="h-4 w-4 text-green-500" />
                          )}
                          <span className="text-sm">
                            后端评估: <span className="font-semibold">{searchProgress.evaluated}</span> / <span className="font-semibold">{searchProgress.totalCandidates}</span>
                            {searchProgress.stage === 'evaluating' && (
                              <span className="ml-2 text-xs text-gray-500">（5 并发）</span>
                            )}
                          </span>
                        </div>
                        {/* 取消按钮 */}
                        {searchProgress.stage === 'evaluating' && onCancelEvaluation && (
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={onCancelEvaluation}
                          >
                            <StopCircle className="h-4 w-4 mr-1" />
                            取消
                          </Button>
                        )}
                      </div>
                      {searchProgress.stage === 'evaluating' && (
                        <Progress
                          value={Math.round((searchProgress.evaluated / searchProgress.totalCandidates) * 100)}
                          className="h-2"
                        />
                      )}
                    </div>
                  )}

                  {/* 已取消 */}
                  {searchProgress.stage === 'cancelled' && (
                    <Alert variant="warning" className="mt-3">
                      <AlertTitle>评估已取消</AlertTitle>
                      <AlertDescription>
                        已完成 {searchProgress.evaluated} / {searchProgress.totalCandidates} 个方案的评估
                      </AlertDescription>
                    </Alert>
                  )}
                </div>
              )}

              {/* 简单加载中状态（无进度信息） */}
              {loading && (!searchProgress || searchProgress.stage === 'idle') && (
                <div className="text-center py-5">
                  <Loader2 className="h-8 w-8 animate-spin text-blue-500 mx-auto" />
                  <div className="mt-3 text-gray-400">正在分析...</div>
                </div>
              )}
            </BaseCard>
          )}
        </div>
      </div>
    </div>
  )
}
