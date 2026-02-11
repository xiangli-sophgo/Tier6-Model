/**
 * 任务详情面板组件
 * 展示任务的详细信息，包括基础信息、配置快照、搜索统计和性能指标
 */

import React from 'react'
import { Button } from '@/components/ui/button'
import { BaseCard } from '@/components/common/BaseCard'
import { BarChart3 } from 'lucide-react'
import type { EvaluationTask } from '@/api/results'
import { formatPercent } from '@/utils/formatters'

interface TaskDetailPanelProps {
  task: EvaluationTask
  onAnalyze: () => void
}


// 信息项组件（单个条目）
const InfoItem: React.FC<{
  label: string
  value: React.ReactNode
}> = ({ label, value }) => (
  <div className="flex flex-col gap-1 py-2 px-3 bg-gray-50/50 rounded border border-gray-100">
    <span className="text-xs text-text-muted">{label}</span>
    <span className="text-sm font-medium text-text-primary break-all">{value}</span>
  </div>
)

// 信息网格组件（多列布局）
const InfoGrid: React.FC<{
  items: Array<{ label: string; value: React.ReactNode }>
  columns?: number
}> = ({ items, columns = 5 }) => (
  <div className={`grid gap-2`} style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
    {items.map((item, index) => (
      <InfoItem key={index} label={item.label} value={item.value} />
    ))}
  </div>
)

// 配置分区组件（带左边框和标题）
const ConfigSection: React.FC<{
  title: string
  color: string
  children: React.ReactNode
  level?: number
}> = ({ title, color, children, level = 0 }) => (
  <div className={level > 0 ? 'ml-4' : ''}>
    <h5
      className={`text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide px-2 py-1 border-l-2`}
      style={{
        backgroundColor: `${color}15`,
        borderLeftColor: color,
      }}
    >
      {title}
    </h5>
    {children}
  </div>
)

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
    const num = Number(value)
    if (isNaN(num)) return String(value)
    value = num
  }
  if (Number.isInteger(value)) return String(value)
  return value.toFixed(decimals)
}

// 格式化日期
const formatDate = (dateStr: string | undefined): string => {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString('zh-CN')
}

// 格式化配置值
const formatConfigValue = (value: any): string => {
  if (value === null || value === undefined) return '-'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

// 渲染简单字段（非对象）
const renderSimpleFields = (
  config: Record<string, unknown>,
  excludeKeys: string[] = []
): Array<{ label: string; value: React.ReactNode }> => {
  return Object.entries(config)
    .filter(([key, value]) => typeof value !== 'object' && !excludeKeys.includes(key))
    .map(([key, value]) => ({
      label: key,
      value: formatConfigValue(value),
    }))
}

// 递归渲染嵌套对象字段（支持任意深度）
const renderNestedFields = (
  config: Record<string, unknown>,
  color: string,
  excludeKeys: string[] = [],
  depth: number = 0
): React.ReactNode => {
  const nestedEntries = Object.entries(config).filter(
    ([key, value]) => typeof value === 'object' && value !== null && !Array.isArray(value) && !excludeKeys.includes(key)
  )

  if (nestedEntries.length === 0) return null

  return nestedEntries.map(([key, value]) => {
    const obj = value as Record<string, unknown>
    // 分离简单字段和嵌套对象字段
    const simpleItems = renderSimpleFields(obj)
    const hasNested = Object.values(obj).some(v => typeof v === 'object' && v !== null && !Array.isArray(v))

    return (
      <div key={key} className="mt-3 ml-4">
        <h6
          className="text-xs font-medium text-text-secondary mb-2 px-2 py-1 border-l-2"
          style={{
            backgroundColor: `${color}${depth === 0 ? '10' : '08'}`,
            borderLeftColor: color,
          }}
        >
          {key}
        </h6>
        {simpleItems.length > 0 && <InfoGrid items={simpleItems} />}
        {hasNested && renderNestedFields(obj, color, [], depth + 1)}
      </div>
    )
  })
}

export const TaskDetailPanel: React.FC<TaskDetailPanelProps> = ({ task, onAnalyze }) => {
  // statusInfo 保留用于将来扩展状态显示
  const _statusInfo = STATUS_MAP[task.status] || { label: task.status, variant: 'outline' as const }
  void _statusInfo // 避免未使用警告
  const result = task.result

  // ============================================
  // 提取配置快照（兼容新旧格式）
  // ============================================
  const configSnapshot = task.config_snapshot || {}

  // 新格式: benchmark_config.model / benchmark_config.inference
  // 旧格式: model / inference
  const benchmarkConfig = configSnapshot.benchmark_config as Record<string, unknown> | undefined
  const modelConfig = (benchmarkConfig?.model || configSnapshot.model || {}) as Record<string, unknown>
  const inferenceConfig = (benchmarkConfig?.inference || configSnapshot.inference || {}) as Record<string, unknown>

  // 新格式: topology_config
  // 旧格式: topology
  const topologyConfig = (configSnapshot.topology_config || configSnapshot.topology || {}) as Record<string, unknown>

  // 从拓扑配置中提取硬件参数（新格式: chips/interconnect 在顶层, 旧格式: hardware_params 包装）
  const hardwareParams = topologyConfig.hardware_params as Record<string, unknown> | undefined
  const chipsConfig = (topologyConfig.chips || hardwareParams?.chips) as Record<string, Record<string, unknown>> | undefined
  const interconnectLinks = (topologyConfig.interconnect as any)?.links || hardwareParams?.interconnect
  const interconnectConfig = interconnectLinks as Record<string, Record<string, unknown>> | undefined

  // 通信配置（新格式: interconnect.comm_params, 旧格式: comm_latency_config）
  const commLatencyConfig = ((topologyConfig.interconnect as any)?.comm_params || topologyConfig.comm_latency_config || hardwareParams?.comm_latency_config) as Record<string, unknown> | undefined

  // ============================================
  // 计算拓扑规模
  // ============================================
  const podCount = (topologyConfig.pod_count as number) || 1
  const racksPerPod = (topologyConfig.racks_per_pod as number) || 1
  const rackConfig = topologyConfig.rack_config as { boards?: Array<{ count?: number; chips?: Array<{ count?: number; name?: string }> }> } | undefined

  let totalBoards = 0
  let totalChips = 0
  let chipName = '-'

  if (rackConfig?.boards) {
    for (const board of rackConfig.boards) {
      totalBoards += board.count || 1
      if (board.chips) {
        for (const chip of board.chips) {
          totalChips += (board.count || 1) * (chip.count || 1)
          if (chipName === '-' && chip.name) {
            chipName = chip.name
          }
        }
      }
    }
    totalBoards *= racksPerPod
    totalChips *= racksPerPod * podCount
  } else {
    // 兼容旧格式：从 pods 数组统计
    const pods = topologyConfig.pods as any[] | undefined
    if (pods) {
      pods.forEach((pod: any) => {
        const racks = pod.racks || []
        racks.forEach((rack: any) => {
          const boards = rack.boards || []
          totalBoards += boards.length
          boards.forEach((board: any) => {
            const chips = board.chips || []
            totalChips += chips.length
            if (chipName === '-' && chips.length > 0 && chips[0].name) {
              chipName = chips[0].name
            }
          })
        })
      })
    }
  }

  // 统计配置项数量
  const configItemCount =
    Object.keys(modelConfig).length +
    Object.keys(inferenceConfig).length +
    (topologyConfig && Object.keys(topologyConfig).length > 0 ? 1 : 0)

  return (
    <div className="space-y-3">
      {/* 顶部操作栏 */}
      <div className="flex items-center">
        <Button onClick={onAnalyze} className="gap-2">
          <BarChart3 className="h-4 w-4" />
          性能分析
        </Button>
      </div>

      {/* 基础信息 */}
      <BaseCard
        title="基础信息"
        collapsible={true}
        defaultExpanded={false}
        collapsibleCount={4}
        glassmorphism={true}
        gradient={true}
      >
        <InfoGrid
          items={[
            { label: 'Benchmark', value: task.benchmark_name || '-' },
            { label: '拓扑配置', value: task.topology_config_name || '-' },
            { label: '任务ID', value: task.task_id },
            { label: '创建时间', value: formatDate(task.created_at) },
          ]}
        />
      </BaseCard>

      {/* 配置参数 */}
      {(configItemCount > 0 || result?.parallelism) && (
        <BaseCard
          title="配置参数"
          collapsible={true}
          defaultExpanded={false}
          collapsibleCount={configItemCount + (result?.parallelism ? 6 : 0)}
          glassmorphism={true}
          gradient={true}
        >
          <div className="space-y-4">
            {/* 并行策略 */}
            {(() => {
              const parallelismConfig = (task.manual_parallelism || result?.parallelism) as Record<string, unknown> | undefined
              if (!parallelismConfig) return null
              return (
                <ConfigSection title="并行策略" color="#6366f1">
                  <InfoGrid items={renderSimpleFields(parallelismConfig)} />
                  {renderNestedFields(parallelismConfig, '#6366f1')}
                </ConfigSection>
              )
            })()}

            {/* 模型配置 */}
            {Object.keys(modelConfig).length > 0 && (
              <ConfigSection title="模型配置" color="#3b82f6">
                <InfoGrid items={renderSimpleFields(modelConfig)} />
                {renderNestedFields(modelConfig, '#3b82f6')}
              </ConfigSection>
            )}

            {/* 推理配置 */}
            {Object.keys(inferenceConfig).length > 0 && (
              <ConfigSection title="推理配置" color="#8b5cf6">
                <InfoGrid items={renderSimpleFields(inferenceConfig)} />
                {renderNestedFields(inferenceConfig, '#8b5cf6')}
              </ConfigSection>
            )}

            {/* 拓扑配置 */}
            {topologyConfig && Object.keys(topologyConfig).length > 0 && (
              <ConfigSection title="拓扑配置" color="#f97316">
                {/* 集群规模 */}
                <InfoGrid
                  items={[
                    { label: 'Pod', value: podCount },
                    { label: 'Rack', value: racksPerPod * podCount },
                    { label: 'Board', value: totalBoards },
                    { label: 'Chip', value: totalChips },
                  ]}
                />

                {/* 芯片规格 */}
                {chipsConfig && Object.keys(chipsConfig).length > 0 && (
                  <div className="mt-3 ml-4">
                    {Object.entries(chipsConfig).map(([chipKey, chipSpec]) => (
                      <div key={chipKey} className="mb-3">
                        <h6
                          className="text-xs font-medium text-text-secondary mb-2 px-2 py-1 border-l-2"
                          style={{
                            backgroundColor: '#f9731610',
                            borderLeftColor: '#f97316',
                          }}
                        >
                          芯片: {chipKey}
                        </h6>
                        <InfoGrid items={renderSimpleFields(chipSpec)} />
                        {renderNestedFields(chipSpec, '#f97316')}
                      </div>
                    ))}
                  </div>
                )}

                {/* 互联配置 */}
                {interconnectConfig && Object.keys(interconnectConfig).length > 0 && (
                  <div className="mt-3 ml-4">
                    <h6
                      className="text-xs font-medium text-text-secondary mb-2 px-2 py-1 border-l-2"
                      style={{
                        backgroundColor: '#f9731610',
                        borderLeftColor: '#f97316',
                      }}
                    >
                      互联配置
                    </h6>
                    <InfoGrid
                      items={Object.entries(interconnectConfig).map(([level, spec]) => ({
                        label: level,
                        value: `${spec.bandwidth_gbps || '-'} GB/s, ${spec.latency_us || '-'} μs`,
                      }))}
                      columns={4}
                    />
                  </div>
                )}

                {/* 通信配置 */}
                {commLatencyConfig && Object.keys(commLatencyConfig).length > 0 && (
                  <div className="mt-3 ml-4">
                    <h6
                      className="text-xs font-medium text-text-secondary mb-2 px-2 py-1 border-l-2"
                      style={{
                        backgroundColor: '#f9731610',
                        borderLeftColor: '#f97316',
                      }}
                    >
                      通信配置
                    </h6>
                    <InfoGrid
                      items={Object.entries(commLatencyConfig)
                        .filter(([_, value]) => typeof value !== 'object')
                        .map(([key, value]) => ({
                          label: key,
                          value: formatConfigValue(value),
                        }))}
                    />
                  </div>
                )}
              </ConfigSection>
            )}
          </div>
        </BaseCard>
      )}

      {/* 性能指标 */}
      {result && (
        <BaseCard
          title="性能指标"
          collapsible={true}
          defaultExpanded={false}
          collapsibleCount={11 + (result.cost ? 5 : 0)}
          glassmorphism={true}
          gradient={true}
        >
          <InfoGrid
            items={[
              { label: '综合得分', value: formatNumber(result.score) },
              { label: '芯片数', value: result.chips || '-' },
              { label: '吞吐量 (TPS)', value: formatNumber(result.tps) },
              { label: '单芯片吞吐 (TPS/Chip)', value: formatNumber(result.tps_per_chip) },
              { label: '单请求吞吐 (TPS/Batch)', value: formatNumber(result.tps_per_batch) },
              { label: 'TPOT (ms)', value: formatNumber(result.tpot, 4) },
              { label: 'TTFT (ms)', value: formatNumber(result.ttft, 4) },
              { label: 'MFU', value: formatPercent(result.mfu) },
              { label: 'MBU', value: formatPercent(result.mbu) },
              { label: '显存占用 (GB)', value: formatNumber(result.dram_occupy / (1024 ** 3), 2) },
              { label: '计算量 (TFLOPs)', value: formatNumber(result.flops / 1e12, 2) },
              ...(result.cost ? [
                { label: '总成本 ($)', value: formatNumber(result.cost.total_cost, 2) },
                { label: '服务器成本 ($)', value: formatNumber(result.cost.server_cost, 2) },
                { label: '互联成本 ($)', value: formatNumber(result.cost.interconnect_cost, 2) },
                { label: '单芯成本 ($)', value: formatNumber(result.cost.cost_per_chip, 2) },
              ] : []),
            ]}
          />
        </BaseCard>
      )}
    </div>
  )
}

export default TaskDetailPanel
