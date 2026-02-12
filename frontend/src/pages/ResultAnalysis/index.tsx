/**
 * 结果分析页面
 * 展示实验详情和结果详细分析
 */

import React, { useEffect, useState } from 'react'
import { ArrowLeft, RefreshCw, Download, Loader2 } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { InfoTooltip } from '@/components/ui/info-tooltip'
import { BaseCard } from '@/components/common/BaseCard'
import { getExperimentDetail, Experiment } from '@/api/results'
import { formatNumber, formatPercentValue, formatDate } from '@/utils/formatters'

// 统计项组件
interface StatItemProps {
  label: string
  value: string | number
  precision?: number
}

const StatItem: React.FC<StatItemProps> = ({ label, value, precision = 2 }) => (
  <div className="p-4 bg-gray-50 rounded">
    <div className="text-gray-400 text-xs mb-1">{label}</div>
    <div className="text-lg font-semibold text-gray-900">
      {typeof value === 'number' ? formatNumber(value, precision) : value}
    </div>
  </div>
)

interface ResultAnalysisProps {
  experimentId?: number
  onBack?: () => void
}

export const ResultAnalysis: React.FC<ResultAnalysisProps> = ({ experimentId, onBack }) => {
  void onBack  // Props may not always be provided
  const [experiment, setExperiment] = useState<Experiment | null>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')

  // 加载实验详情
  const loadExperiment = async () => {
    if (!experimentId) return
    setLoading(true)
    try {
      const data = await getExperimentDetail(experimentId)
      setExperiment(data)
    } catch (error) {
      toast.error('加载实验详情失败')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  // 首次加载
  useEffect(() => {
    loadExperiment()
  }, [experimentId])

  // 导出结果
  const handleExport = () => {
    toast.info('导出功能开发中...')
  }

  if (!experiment) {
    return (
      <div className="h-full flex flex-col">
        <div className="px-6 py-4 border-b border-gray-100 bg-white">
          {onBack && (
            <Button variant="ghost" onClick={onBack}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              返回
            </Button>
          )}
        </div>
        <div className="flex-1 flex items-center justify-center">
          {loading && <Loader2 className="h-8 w-8 animate-spin text-blue-500" />}
        </div>
      </div>
    )
  }

  // 从第一个任务的 config_snapshot 中提取配置
  const firstTask = experiment.tasks && experiment.tasks.length > 0 ? experiment.tasks[0] : null
  const modelConfig = firstTask?.config_snapshot?.model as Record<string, unknown> || {}
  const inferenceConfig = firstTask?.config_snapshot?.inference as Record<string, unknown> || {}

  const progress =
    experiment.total_tasks > 0
      ? Math.round((experiment.completed_tasks / experiment.total_tasks) * 100)
      : 0

  return (
      <div className="h-full flex flex-col">
        {/* 标题栏 */}
        <div className="px-6 py-4 border-b border-gray-100 bg-white flex justify-between items-center">
          <div className="flex items-center gap-3">
            {onBack && (
              <Button variant="ghost" onClick={onBack}>
                <ArrowLeft className="h-4 w-4 mr-2" />
                返回
              </Button>
            )}
            <div>
              <div className="text-lg font-semibold text-gray-900">
                {experiment.name}
              </div>
              <span className="text-sm text-gray-500">{experiment.description || '无描述'}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <InfoTooltip content="刷新">
              <Button variant="outline" size="icon" onClick={loadExperiment}>
                <RefreshCw className="h-4 w-4" />
              </Button>
            </InfoTooltip>
            <Button variant="outline" onClick={handleExport}>
              <Download className="h-4 w-4 mr-2" />
              导出
            </Button>
          </div>
        </div>

        {/* 内容区 */}
        <div className="flex-1 overflow-auto p-6">
          {/* 进度提示 */}
          {progress < 100 && (
            <Alert className="mb-4">
              <AlertDescription>
                实验进度: {progress}% ({experiment.completed_tasks}/{experiment.total_tasks} 任务完成)
              </AlertDescription>
            </Alert>
          )}

          {/* 标签页 */}
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList>
              <TabsTrigger value="overview">概览</TabsTrigger>
              <TabsTrigger value="tasks">任务列表 ({experiment.tasks?.length || 0})</TabsTrigger>
              <TabsTrigger value="results">结果分析</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="pt-4">
              {/* 任务进度 */}
              <BaseCard title="任务进度" className="mb-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-gray-400 text-xs mb-1">已完成任务</div>
                    <div className="text-2xl font-semibold">
                      {experiment.completed_tasks}
                      <span className="text-sm text-gray-400 ml-1">/ {experiment.total_tasks}</span>
                    </div>
                  </div>
                  <div className="py-4">
                    <div className="mb-2 text-sm">进度: {progress}%</div>
                    <div className="w-full h-2 bg-gray-200 rounded">
                      <div
                        className={`h-full rounded transition-all ${progress === 100 ? 'bg-green-500' : 'bg-blue-500'}`}
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  </div>
                </div>
              </BaseCard>

              {/* 模型配置 */}
              <BaseCard title="模型配置" className="mb-4">
                <div className="grid grid-cols-3 gap-4">
                  <StatItem label="模型名称" value={modelConfig.model_name as string || '-'} />
                  <StatItem label="隐藏层尺寸" value={modelConfig.hidden_size as number || '-'} />
                  <StatItem label="层数" value={modelConfig.num_layers as number || '-'} />
                  <StatItem label="注意力头数" value={modelConfig.num_attention_heads as number || '-'} />
                  <StatItem label="KV 头数" value={modelConfig.num_kv_heads as number || '-'} />
                  <StatItem label="数据类型" value={modelConfig.dtype as string || '-'} />
                </div>
              </BaseCard>

              {/* 推理配置 */}
              <BaseCard title="推理配置" className="mb-4">
                <div className="grid grid-cols-3 gap-4">
                  <StatItem label="批次大小" value={inferenceConfig.batch_size as number || '-'} />
                  <StatItem label="输入序列长度" value={inferenceConfig.input_seq_length as number || '-'} />
                  <StatItem label="输出序列长度" value={inferenceConfig.output_seq_length as number || '-'} />
                </div>
              </BaseCard>

              {/* 创建时间 */}
              <BaseCard title="元数据" className="mb-4">
                <div className="grid grid-cols-2 gap-4">
                  <StatItem label="创建时间" value={formatDate(experiment.created_at)} />
                  <StatItem label="最后更新" value={formatDate(experiment.updated_at)} />
                </div>
              </BaseCard>
            </TabsContent>

            <TabsContent value="tasks" className="pt-4">
              {experiment.tasks && experiment.tasks.length > 0 ? (
                <div className="grid grid-cols-[repeat(auto-fill,minmax(250px,1fr))] gap-4">
                  {experiment.tasks.map((task) => (
                    <div
                      key={task.task_id}
                      className="p-4 bg-white rounded-lg border hover:shadow-md transition-shadow cursor-pointer"
                      onClick={() => toast.info(`任务 ID: ${task.task_id}`)}
                    >
                      <div className="mb-2 font-semibold">{task.task_id}</div>
                      <div className="mb-1 text-xs">
                        <span className="text-gray-500">状态:</span>
                        <span className="ml-2 text-blue-500">{task.status}</span>
                      </div>
                      <div className="mb-1 text-xs">
                        <span className="text-gray-500">进度:</span>
                        <span className="ml-2">{formatPercentValue(task.progress, 1)}</span>
                      </div>
                      <div className="mb-1 text-xs">
                        <span className="text-gray-500">创建时间:</span>
                        <span className="ml-2">{formatDate(task.created_at)}</span>
                      </div>
                      {task.message && (
                        <div className="text-xs">
                          <span className="text-gray-500">消息:</span>
                          <div className="mt-1 text-gray-600">{task.message}</div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <Alert>
                  <AlertDescription>暂无任务</AlertDescription>
                </Alert>
              )}
            </TabsContent>

            <TabsContent value="results" className="pt-4">
              <Alert>
                <AlertTitle>结果分析功能开发中</AlertTitle>
                <AlertDescription>
                  将展示详细的性能指标、甘特图、通信开销分析等内容
                </AlertDescription>
              </Alert>
            </TabsContent>
          </Tabs>
        </div>
      </div>
  )
}
