/**
 * 任务详情面板组件
 * 展示任务的详细信息，包括基础信息、配置快照、搜索统计和性能指标
 */

import React from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { BarChart3, ChevronDown, ChevronRight } from 'lucide-react'
import type { EvaluationTask } from '@/api/results'

interface TaskDetailPanelProps {
  task: EvaluationTask
  onAnalyze: () => void
}

// 信息行组件
const InfoRow: React.FC<{
  label: string
  value: React.ReactNode
  fullWidth?: boolean
}> = ({ label, value, fullWidth = false }) => (
  <div className={`flex ${fullWidth ? 'flex-col gap-1' : 'items-center justify-between'} py-1.5 ${fullWidth ? '' : 'gap-4'}`}>
    <span className="text-sm text-text-muted min-w-[120px]">{label}</span>
    <span className="text-sm font-medium text-text-primary break-all">{value}</span>
  </div>
)

// 折叠面板组件
const CollapsibleSection: React.FC<{
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
  count?: number
}> = ({ title, children, defaultOpen = false, count }) => {
  const [isOpen, setIsOpen] = React.useState(defaultOpen)

  const handleToggle = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsOpen(!isOpen)
  }

  return (
    <Card className="bg-white/80 backdrop-blur-sm">
      <CardHeader
        className="cursor-pointer py-3 px-4 hover:bg-gray-50/50 transition-colors"
        onClick={handleToggle}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {isOpen ? (
              <ChevronDown className="h-4 w-4 text-gray-500" />
            ) : (
              <ChevronRight className="h-4 w-4 text-gray-500" />
            )}
            <CardTitle className="text-base">
              {title}
              {count !== undefined && (
                <span className="ml-2 text-sm text-gray-500 font-normal">({count})</span>
              )}
            </CardTitle>
          </div>
        </div>
      </CardHeader>
      {isOpen && (
        <CardContent className="pt-0 pb-4 px-4 bg-white/50">
          {children}
        </CardContent>
      )}
    </Card>
  )
}

// 任务状态映射
const STATUS_MAP: Record<string, { label: string; variant: 'default' | 'secondary' | 'destructive' | 'outline' | 'success' | 'warning' }> = {
  'pending': { label: '等待中', variant: 'secondary' },
  'running': { label: '运行中', variant: 'default' },
  'completed': { label: '已完成', variant: 'success' },
  'failed': { label: '失败', variant: 'destructive' },
  'cancelled': { label: '已取消', variant: 'warning' },
}

// 格式化数值
const formatNumber = (value: any, decimals = 2): string => {
  if (value === undefined || value === null) return '-'
  if (typeof value !== 'number') {
    // 尝试转换为数字
    const num = Number(value)
    if (isNaN(num)) return String(value)
    value = num
  }
  if (Number.isInteger(value)) return String(value)
  return value.toFixed(decimals)
}

// 格式化百分比
const formatPercent = (value: number | undefined, decimals = 2): string => {
  if (value === undefined || value === null) return '-'
  return `${(value * 100).toFixed(decimals)}%`
}

// 格式化日期
const formatDate = (dateStr: string | undefined): string => {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString('zh-CN')
}

// 格式化配置对象
const formatConfigValue = (value: any): string => {
  if (value === null || value === undefined) return '-'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'object') return JSON.stringify(value, null, 2)
  return String(value)
}

export const TaskDetailPanel: React.FC<TaskDetailPanelProps> = ({ task, onAnalyze }) => {
  const statusInfo = STATUS_MAP[task.status] || { label: task.status, variant: 'outline' as const }
  const result = task.result

  // 提取配置快照的关键信息
  const configSnapshot = task.config_snapshot || {}
  const modelConfig = configSnapshot.model || {}
  const hardwareConfig = configSnapshot.hardware || {}
  const inferenceConfig = configSnapshot.inference || {}
  const topologyConfig = configSnapshot.topology || {}

  // 统计配置项数量
  const configItemCount =
    Object.keys(modelConfig).length +
    Object.keys(hardwareConfig).length +
    Object.keys(inferenceConfig).length +
    (topologyConfig && Object.keys(topologyConfig).length > 0 ? 1 : 0)

  // 搜索统计数据
  const searchStats = task.search_stats || {}
  const searchStatsCount = Object.keys(searchStats).length

  return (
    <div className="space-y-3">
      {/* 顶部操作栏 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h4 className="text-sm font-semibold text-text-primary">任务详情</h4>
          <Badge variant={statusInfo.variant}>{statusInfo.label}</Badge>
        </div>
        <Button onClick={onAnalyze} className="gap-2">
          <BarChart3 className="h-4 w-4" />
          性能分析
        </Button>
      </div>

      {/* 基础信息 */}
      <CollapsibleSection title="基础信息" defaultOpen={false} count={6}>
        <div className="space-y-0.5">
          <InfoRow label="Benchmark" value={task.benchmark_name || '-'} />
          <InfoRow label="拓扑配置" value={task.topology_config_name || '-'} />
          <InfoRow label="任务状态" value={statusInfo.label} />
          <InfoRow label="任务ID" value={task.task_id} />
          <InfoRow label="结果ID" value={task.result_id || '-'} />
          <InfoRow label="创建时间" value={formatDate(task.created_at)} />
        </div>
      </CollapsibleSection>

      {/* 性能指标 */}
      {result && (
        <CollapsibleSection title="性能指标" defaultOpen={false} count={8}>
          <div className="space-y-0.5">
            <InfoRow label="综合得分" value={formatNumber(result.score)} />
            <InfoRow label="芯片数" value={result.chips || '-'} />
            <InfoRow label="吞吐量 (TPS)" value={formatNumber(result.throughput)} />
            <InfoRow label="单芯片吞吐 (TPS/Chip)" value={formatNumber(result.tps_per_chip)} />
            <InfoRow label="TPOT (ms)" value={formatNumber(result.tpot, 4)} />
            <InfoRow label="TTFT (ms)" value={formatNumber(result.ttft, 4)} />
            <InfoRow label="MFU" value={formatPercent(result.mfu)} />
            <InfoRow label="MBU" value={formatPercent(result.mbu)} />
          </div>
        </CollapsibleSection>
      )}

      {/* 搜索统计 */}
      {searchStatsCount > 0 && (
        <CollapsibleSection title="搜索统计" count={searchStatsCount}>
          <div className="space-y-0.5">
            {Object.entries(searchStats).map(([key, value]) => (
              <InfoRow
                key={key}
                label={key.replace(/_/g, ' ')}
                value={formatNumber(value as number)}
              />
            ))}
          </div>
        </CollapsibleSection>
      )}

      {/* 配置快照 */}
      {configItemCount > 0 && (
        <CollapsibleSection title="配置快照" count={configItemCount}>
          <div className="space-y-4">
            {/* 模型配置 */}
            {Object.keys(modelConfig).length > 0 && (
              <div>
                <h5 className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">模型配置</h5>
                <div className="space-y-0.5 pl-2 border-l-2 border-blue-200">
                  {Object.entries(modelConfig).map(([key, value]) => (
                    <InfoRow
                      key={key}
                      label={key}
                      value={formatConfigValue(value)}
                      fullWidth={typeof value === 'object'}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* 硬件配置 */}
            {Object.keys(hardwareConfig).length > 0 && (
              <div>
                <h5 className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">硬件配置</h5>
                <div className="space-y-0.5 pl-2 border-l-2 border-green-200">
                  {Object.entries(hardwareConfig).map(([key, value]) => (
                    <InfoRow
                      key={key}
                      label={key}
                      value={formatConfigValue(value)}
                      fullWidth={typeof value === 'object'}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* 推理配置 */}
            {Object.keys(inferenceConfig).length > 0 && (
              <div>
                <h5 className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">推理配置</h5>
                <div className="space-y-0.5 pl-2 border-l-2 border-purple-200">
                  {Object.entries(inferenceConfig).map(([key, value]) => (
                    <InfoRow
                      key={key}
                      label={key}
                      value={formatConfigValue(value)}
                      fullWidth={typeof value === 'object'}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* 拓扑配置 */}
            {topologyConfig && Object.keys(topologyConfig).length > 0 && (
              <div>
                <h5 className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">拓扑配置</h5>
                <div className="space-y-0.5 pl-2 border-l-2 border-orange-200">
                  <InfoRow
                    label="拓扑结构"
                    value={
                      <pre className="text-xs bg-gray-50 p-2 rounded overflow-auto max-h-[200px]">
                        {JSON.stringify(topologyConfig, null, 2)}
                      </pre>
                    }
                    fullWidth
                  />
                </div>
              </div>
            )}
          </div>
        </CollapsibleSection>
      )}
    </div>
  )
}

export default TaskDetailPanel
