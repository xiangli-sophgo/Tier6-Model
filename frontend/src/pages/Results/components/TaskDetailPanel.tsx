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

// ============================================
// 字段中文名称映射
// ============================================
const FIELD_LABELS: Record<string, string> = {
  // 模型配置
  model_name: '模型名称',
  model_type: '模型类型',
  hidden_size: '隐藏层维度',
  num_layers: '层数',
  num_attention_heads: '注意力头数',
  num_kv_heads: 'KV 头数',
  intermediate_size: 'FFN 中间层',
  vocab_size: '词表大小',
  dtype: '数据类型',
  weight_dtype: '权重数据类型',
  activation_dtype: '激活数据类型',
  max_seq_length: '最大序列长度',
  attention_type: '注意力类型',
  norm_type: '归一化类型',

  // MoE 配置
  moe_config: 'MoE 配置',
  num_experts: '专家数量',
  experts_per_token: '每 Token 激活专家数',
  num_shared_experts: '共享专家数',
  router_topk_policy: '路由 TopK 策略',

  // MLA 配置
  mla_config: 'MLA 配置',
  kv_lora_rank: 'KV LoRA 秩',
  q_lora_rank: 'Q LoRA 秩',
  qk_rope_dim: 'QK RoPE 维度',
  qk_nope_dim: 'QK Non-RoPE 维度',
  v_head_dim: 'V 头维度',

  // 推理配置
  batch_size: '批次大小',
  input_seq_length: '输入序列长度',
  output_seq_length: '输出序列长度',
  num_micro_batches: '微批次数量',

  // 硬件配置
  name: '名称',
  num_cores: '计算核心数',
  compute_tflops_fp8: 'FP8 算力 (TFLOPS)',
  compute_tflops_bf16: 'BF16 算力 (TFLOPS)',
  memory_capacity_gb: '显存容量 (GB)',
  memory_bandwidth_gbps: '显存带宽 (GB/s)',
  memory_bandwidth_utilization: '显存带宽利用率',
  lmem_capacity_mb: 'LMEM 容量 (MB)',
  lmem_bandwidth_gbps: 'LMEM 带宽 (GB/s)',
  cube_m: 'Cube M 维度',
  cube_k: 'Cube K 维度',
  cube_n: 'Cube N 维度',
  sram_size_kb: 'SRAM 大小 (KB)',
  sram_utilization: 'SRAM 利用率',
  lane_num: 'Lane 数量',
  align_bytes: '对齐字节数',
  compute_dma_overlap_rate: '计算-搬运重叠率',

  // 互联配置
  c2c: '芯片间 (C2C)',
  b2b: '板间 (B2B)',
  r2r: '机架间 (R2R)',
  p2p: 'Pod 间 (P2P)',
  bandwidth_gbps: '带宽 (GB/s)',
  latency_us: '延迟 (μs)',

  // 通信配置
  allreduce_algorithm: 'AllReduce 算法',
  alltoall_algorithm: 'AllToAll 算法',
  enable_compute_comm_overlap: '计算-通信重叠',
  network_efficiency: '网络效率',

  // 拓扑配置
  pod_count: 'Pod 数量',
  racks_per_pod: '每 Pod Rack 数',
  count: '数量',
  chips: '芯片配置',
  preset_id: '预设 ID',
}

// 获取字段的中文名称
const getFieldLabel = (key: string): string => {
  return FIELD_LABELS[key] || key
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
      label: getFieldLabel(key),
      value: formatConfigValue(value),
    }))
}

// 渲染嵌套对象字段
const renderNestedFields = (
  config: Record<string, unknown>,
  color: string,
  excludeKeys: string[] = []
): React.ReactNode => {
  const nestedEntries = Object.entries(config).filter(
    ([key, value]) => typeof value === 'object' && value !== null && !excludeKeys.includes(key)
  )

  if (nestedEntries.length === 0) return null

  return nestedEntries.map(([key, value]) => (
    <div key={key} className="mt-3 ml-4">
      <h6
        className="text-xs font-medium text-text-secondary mb-2 px-2 py-1 border-l-2"
        style={{
          backgroundColor: `${color}10`,
          borderLeftColor: color,
        }}
      >
        {getFieldLabel(key)}
      </h6>
      <InfoGrid
        items={Object.entries(value as Record<string, unknown>).map(([subKey, subValue]) => ({
          label: getFieldLabel(subKey),
          value: formatConfigValue(subValue),
        }))}
      />
    </div>
  ))
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

  // 从拓扑配置中提取硬件参数
  const hardwareParams = topologyConfig.hardware_params as Record<string, unknown> | undefined
  const chipsConfig = hardwareParams?.chips as Record<string, Record<string, unknown>> | undefined
  const interconnectConfig = hardwareParams?.interconnect as Record<string, Record<string, unknown>> | undefined

  // 通信配置（可能在 topology_config 或 hardware_params 中）
  const commLatencyConfig = (topologyConfig.comm_latency_config || hardwareParams?.comm_latency_config) as Record<string, unknown> | undefined

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
            {result?.parallelism && (
              <ConfigSection title="并行策略" color="#6366f1">
                <InfoGrid
                  items={[
                    { label: 'DP (数据并行)', value: result.parallelism.dp || '-' },
                    { label: 'TP (张量并行)', value: result.parallelism.tp || '-' },
                    { label: 'PP (流水线并行)', value: result.parallelism.pp || '-' },
                    { label: 'EP (专家并行)', value: result.parallelism.ep || '-' },
                    { label: 'SP (序列并行)', value: result.parallelism.sp || '-' },
                    { label: 'MoE_TP', value: result.parallelism.moe_tp || '-' },
                  ]}
                />
              </ConfigSection>
            )}

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
                        <InfoGrid
                          items={[
                            { label: 'FP8 算力', value: chipSpec.compute_tflops_fp8 ? `${chipSpec.compute_tflops_fp8} TFLOPS` : '-' },
                            { label: 'BF16 算力', value: chipSpec.compute_tflops_bf16 ? `${chipSpec.compute_tflops_bf16} TFLOPS` : '-' },
                            { label: '显存容量', value: chipSpec.memory_capacity_gb ? `${chipSpec.memory_capacity_gb} GB` : '-' },
                            { label: '显存带宽', value: chipSpec.memory_bandwidth_gbps ? `${chipSpec.memory_bandwidth_gbps} GB/s` : '-' },
                          ]}
                        />
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
                        label: getFieldLabel(level),
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
                          label: getFieldLabel(key),
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
