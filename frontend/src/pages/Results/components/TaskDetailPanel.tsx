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
}> = ({ items, columns = 4 }) => (
  <div className={`grid gap-2`} style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
    {items.map((item, index) => (
      <InfoItem key={index} label={item.label} value={item.value} />
    ))}
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
      <CollapsibleSection title="基础信息" defaultOpen={false} count={4}>
        <InfoGrid
          items={[
            { label: 'Benchmark', value: task.benchmark_name || '-' },
            { label: '拓扑配置', value: task.topology_config_name || '-' },
            { label: '任务ID', value: task.task_id },
            { label: '创建时间', value: formatDate(task.created_at) },
          ]}
        />
      </CollapsibleSection>

      {/* 配置参数 */}
      {(configItemCount > 0 || result?.parallelism) && (
        <CollapsibleSection title="配置参数" count={configItemCount + (result?.parallelism ? 6 : 0)}>
          <div className="space-y-4">
            {/* 并行策略 */}
            {result?.parallelism && (
              <div>
                <h5 className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide px-2 py-1 bg-indigo-50 border-l-2 border-indigo-400">并行策略</h5>
                <InfoGrid
                  items={[
                    { label: 'DP (数据并行)', value: result.parallelism.dp || '-' },
                    { label: 'TP (张量并行)', value: result.parallelism.tp || '-' },
                    { label: 'PP (流水线并行)', value: result.parallelism.pp || '-' },
                    { label: 'EP (专家并行)', value: result.parallelism.ep || '-' },
                    { label: 'SP (序列并行)', value: result.parallelism.sp || '-' },
                    { label: 'MoE_TP (MoE张量并行)', value: result.parallelism.moe_tp || '-' },
                  ]}
                />
              </div>
            )}

            {/* 模型配置 */}
            {Object.keys(modelConfig).length > 0 && (
              <div>
                <h5 className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide px-2 py-1 bg-blue-50 border-l-2 border-blue-400">模型配置</h5>
                <InfoGrid
                  items={Object.entries(modelConfig)
                    .filter(([_, value]) => typeof value !== 'object')
                    .map(([key, value]) => ({
                      label: key,
                      value: formatConfigValue(value),
                    }))}
                />
                {/* 嵌套对象配置展开显示 */}
                {Object.entries(modelConfig)
                  .filter(([_, value]) => typeof value === 'object' && value !== null)
                  .map(([key, value]) => (
                    <div key={key} className="mt-3">
                      <h6 className="text-xs font-medium text-text-secondary mb-2">{key}</h6>
                      <InfoGrid
                        items={Object.entries(value as Record<string, unknown>).map(([subKey, subValue]) => ({
                          label: subKey,
                          value: formatConfigValue(subValue),
                        }))}
                      />
                    </div>
                  ))}
              </div>
            )}

            {/* 硬件配置 */}
            {Object.keys(hardwareConfig).length > 0 && (
              <div>
                <h5 className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide px-2 py-1 bg-green-50 border-l-2 border-green-400">硬件配置</h5>
                <InfoGrid
                  items={Object.entries(hardwareConfig)
                    .filter(([_, value]) => typeof value !== 'object')
                    .map(([key, value]) => ({
                      label: key,
                      value: formatConfigValue(value),
                    }))}
                />
                {/* 嵌套对象配置展开显示 */}
                {Object.entries(hardwareConfig)
                  .filter(([_, value]) => typeof value === 'object' && value !== null)
                  .map(([key, value]) => (
                    <div key={key} className="mt-3">
                      <h6 className="text-xs font-medium text-text-secondary mb-2">{key}</h6>
                      <InfoGrid
                        items={Object.entries(value as Record<string, unknown>).map(([subKey, subValue]) => ({
                          label: subKey,
                          value: formatConfigValue(subValue),
                        }))}
                      />
                    </div>
                  ))}
              </div>
            )}

            {/* 推理配置 */}
            {Object.keys(inferenceConfig).length > 0 && (
              <div>
                <h5 className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide px-2 py-1 bg-purple-50 border-l-2 border-purple-400">推理配置</h5>
                <InfoGrid
                  items={Object.entries(inferenceConfig)
                    .filter(([_, value]) => typeof value !== 'object')
                    .map(([key, value]) => ({
                      label: key,
                      value: formatConfigValue(value),
                    }))}
                />
                {/* 嵌套对象配置展开显示 */}
                {Object.entries(inferenceConfig)
                  .filter(([_, value]) => typeof value === 'object' && value !== null)
                  .map(([key, value]) => (
                    <div key={key} className="mt-3">
                      <h6 className="text-xs font-medium text-text-secondary mb-2">{key}</h6>
                      <InfoGrid
                        items={Object.entries(value as Record<string, unknown>).map(([subKey, subValue]) => ({
                          label: subKey,
                          value: formatConfigValue(subValue),
                        }))}
                      />
                    </div>
                  ))}
              </div>
            )}

            {/* 拓扑配置 */}
            {topologyConfig && Object.keys(topologyConfig).length > 0 && (
              <div>
                <h5 className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide px-2 py-1 bg-orange-50 border-l-2 border-orange-400">拓扑配置</h5>
                {(() => {
                  const topology = topologyConfig as any

                  // 方式1：从hardware_config读取（如果有）
                  const hardwareConfig = topology.hardware_config

                  // 方式2：从拓扑结构提取
                  const pods = topology.pods || []
                  const numPods = pods.length

                  let numRacks = 0
                  let numBoards = 0
                  let numChips = 0
                  let chipInfo: any = null
                  let boardConnections: any = null
                  let rackConnections: any = null
                  let podConnections: any = null

                  // 统计数量并提取第一个chip信息和连接信息
                  pods.forEach((pod: any) => {
                    const racks = pod.racks || []
                    numRacks += racks.length

                    // 提取pod级别的连接
                    if (!podConnections && pod.connections && pod.connections.length > 0) {
                      podConnections = pod.connections[0]
                    }

                    racks.forEach((rack: any) => {
                      const boards = rack.boards || []
                      numBoards += boards.length

                      // 提取rack级别的连接
                      if (!rackConnections && rack.connections && rack.connections.length > 0) {
                        rackConnections = rack.connections[0]
                      }

                      boards.forEach((board: any) => {
                        const chips = board.chips || []
                        numChips += chips.length

                        // 获取第一个chip的信息
                        if (!chipInfo && chips.length > 0) {
                          chipInfo = chips[0]
                        }

                        // 提取board级别的连接
                        if (!boardConnections && board.connections && board.connections.length > 0) {
                          boardConnections = board.connections[0]
                        }
                      })
                    })
                  })

                  // 优先使用hardware_config中的芯片信息
                  const chip = hardwareConfig?.chip || chipInfo
                  const node = hardwareConfig?.node
                  const cluster = hardwareConfig?.cluster

                  // 构建显示项
                  const topologyItems = [
                    { label: 'Pod数量', value: numPods },
                    { label: 'Rack数量', value: numRacks },
                    { label: 'Board数量', value: numBoards },
                    { label: 'Chip总数', value: numChips },
                  ]

                  const chipItems = chip ? [
                    { label: '芯片类型', value: chip.chip_type || chip.name || '-' },
                    { label: '算力 (FP16)', value: chip.compute_tflops_fp16 ? `${chip.compute_tflops_fp16} TFLOPs` : '-' },
                    { label: '显存', value: chip.memory_gb ? `${chip.memory_gb} GB` : '-' },
                    { label: 'HBM带宽', value: chip.memory_bandwidth_gbps ? `${chip.memory_bandwidth_gbps} GB/s` : (chip.memory_bandwidth_tbps ? `${chip.memory_bandwidth_tbps * 1000} GB/s` : '-') },
                  ] : []

                  // 构建互联配置项（支持多种数据源）
                  const interconnectItems = []

                  // 板内互联（优先从hardware_config.node读取）
                  if (node?.intra_bandwidth_gbps || boardConnections) {
                    interconnectItems.push({
                      label: '板内互联',
                      value: node ?
                        `NVLink (${node.intra_bandwidth_gbps || '-'} GB/s, ${node.intra_latency_us ? node.intra_latency_us * 1000 : '-'} ns)` :
                        `${boardConnections.link_type || 'NVLink'} (${boardConnections.bandwidth_gbps || boardConnections.bandwidth || '-'} GB/s, ${boardConnections.latency_ns || boardConnections.latency || '-'} ns)`
                    })
                  }

                  // 机架内互联
                  if (cluster?.intra_bandwidth_gbps || rackConnections) {
                    interconnectItems.push({
                      label: '机架内互联',
                      value: cluster ?
                        `InfiniBand (${cluster.intra_bandwidth_gbps || '-'} GB/s, ${cluster.intra_latency_us ? cluster.intra_latency_us * 1000 : '-'} ns)` :
                        `${rackConnections.link_type || 'InfiniBand'} (${rackConnections.bandwidth_gbps || rackConnections.bandwidth || '-'} GB/s, ${rackConnections.latency_ns || rackConnections.latency || '-'} ns)`
                    })
                  }

                  // 跨机架互联
                  if (cluster?.inter_bandwidth_gbps || podConnections) {
                    interconnectItems.push({
                      label: '跨机架互联',
                      value: cluster ?
                        `InfiniBand (${cluster.inter_bandwidth_gbps || '-'} GB/s, ${cluster.inter_latency_us ? cluster.inter_latency_us * 1000 : '-'} ns)` :
                        `${podConnections.link_type || 'InfiniBand'} (${podConnections.bandwidth_gbps || podConnections.bandwidth || '-'} GB/s, ${podConnections.latency_ns || podConnections.latency || '-'} ns)`
                    })
                  }

                  return (
                    <div className="space-y-3">
                      {/* 集群规模 */}
                      <div>
                        <h6 className="text-xs font-medium text-text-secondary mb-2">集群规模</h6>
                        <InfoGrid items={topologyItems} />
                      </div>

                      {/* 芯片规格 */}
                      {chipItems.length > 0 && (
                        <div>
                          <h6 className="text-xs font-medium text-text-secondary mb-2">芯片规格</h6>
                          <InfoGrid items={chipItems} />
                        </div>
                      )}

                      {/* 互联配置 */}
                      {interconnectItems.length > 0 && (
                        <div>
                          <h6 className="text-xs font-medium text-text-secondary mb-2">互联配置</h6>
                          <InfoGrid items={interconnectItems} columns={2} />
                        </div>
                      )}
                    </div>
                  )
                })()}
              </div>
            )}
          </div>
        </CollapsibleSection>
      )}

      {/* 性能指标 */}
      {result && (
        <CollapsibleSection title="性能指标" defaultOpen={false} count={11 + (result.cost ? 5 : 0)}>
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
                { label: '单位成本 ($/M tokens)', value: formatNumber(result.cost.cost_per_million_tokens, 4) },
                { label: '服务器成本 ($)', value: formatNumber(result.cost.server_cost, 2) },
                { label: '互联成本 ($)', value: formatNumber(result.cost.interconnect_cost, 2) },
                { label: '单芯成本 ($)', value: formatNumber(result.cost.cost_per_chip, 2) },
              ] : []),
            ]}
          />
        </CollapsibleSection>
      )}
    </div>
  )
}

export default TaskDetailPanel
