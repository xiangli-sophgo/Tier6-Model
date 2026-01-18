/**
 * 分析结果展示组件
 *
 * - 首页显示历史记录列表
 * - 点击历史记录查看详情
 * - 支持返回历史记录列表
 */

import React, { useState, useCallback } from 'react'
import {
  Typography,
  Spin,
  Tag,
  Tooltip,
  Button,
  Table,
  Popconfirm,
  Empty,
} from 'antd'
import {
  InfoCircleOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  HistoryOutlined,
  DeleteOutlined,
  ClearOutlined,
  ExportOutlined,
  ThunderboltOutlined,
  DashboardOutlined,
  ClockCircleOutlined,
  AimOutlined,
} from '@ant-design/icons'
import { PlanAnalysisResult, HardwareConfig, LLMModelConfig, InferenceConfig, DEFAULT_SCORE_WEIGHTS } from '../../../utils/llmDeployment/types'
import { generateBenchmarkName, parseBenchmarkParts } from '../../../utils/llmDeployment/benchmarkNaming'
import { AnalysisHistoryItem, AnalysisViewMode } from '../shared'
import { colors } from './ConfigSelectors'
import { BaseCard } from '../../common/BaseCard'
import { MetricDetailCard } from './components/MetricDetailCard'
import { ModelInfoCard } from './components/ModelInfoCard'
import { ParallelismInfo, ParallelismCard, type ParallelismType } from './components/ParallelismInfo'

const { Text } = Typography

// ============================================
// 历史记录列表组件
// ============================================

interface HistoryListProps {
  history: AnalysisHistoryItem[]
  onLoad: (item: AnalysisHistoryItem) => void
  onDelete: (id: string) => void
  onClear: () => void
}

const HistoryList: React.FC<HistoryListProps> = ({
  history,
  onLoad,
  onDelete,
  onClear,
}) => {
  // 导出JSON
  const handleExportJSON = () => {
    const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `llm-deployment-history-${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (history.length === 0) {
    return (
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description="暂无历史记录"
        style={{ padding: '40px 0' }}
      >
        <Text type="secondary" style={{ fontSize: 12 }}>
          点击左侧"运行分析"开始第一次分析
        </Text>
      </Empty>
    )
  }

  const columns = [
    {
      title: 'Benchmark',
      key: 'benchmark',
      width: 260,
      ellipsis: true,
      render: (_: unknown, record: AnalysisHistoryItem) => (
        <Text strong style={{ fontSize: 14 }}>
          {generateBenchmarkName(record.modelConfig, record.inferenceConfig)}
        </Text>
      ),
    },
    {
      title: '并行策略',
      key: 'parallelism',
      width: 160,
      render: (_: unknown, record: AnalysisHistoryItem) => (
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          {record.parallelism.dp > 1 && (
            <Tag color="blue" style={{ fontSize: 12, margin: 0 }}>DP{record.parallelism.dp}</Tag>
          )}
          {record.parallelism.tp > 1 && (
            <Tag color="green" style={{ fontSize: 12, margin: 0 }}>TP{record.parallelism.tp}</Tag>
          )}
          {record.parallelism.pp > 1 && (
            <Tag color="orange" style={{ fontSize: 12, margin: 0 }}>PP{record.parallelism.pp}</Tag>
          )}
          {record.parallelism.ep > 1 && (
            <Tag color="purple" style={{ fontSize: 12, margin: 0 }}>EP{record.parallelism.ep}</Tag>
          )}
        </div>
      ),
    },
    {
      title: 'TPS/Chip',
      key: 'tps_chip',
      width: 120,
      align: 'center' as const,
      render: (_: unknown, record: AnalysisHistoryItem) => {
        // TPS/Chip = Total TPS / chips
        const tpsPerChip = record.chips > 0 ? record.throughput / record.chips : 0
        return <span style={{ fontSize: 14 }}>{tpsPerChip.toFixed(0)} tok/s</span>
      },
    },
    {
      title: 'FTL',
      dataIndex: 'ttft',
      key: 'ttft',
      width: 90,
      align: 'center' as const,
      render: (v: number) => <span style={{ fontSize: 14 }}>{v.toFixed(1)} ms</span>,
    },
    {
      title: '',
      key: 'actions',
      width: 40,
      render: (_: unknown, record: AnalysisHistoryItem) => (
        <Popconfirm
          title="删除此记录？"
          onConfirm={(e) => {
            e?.stopPropagation()
            onDelete(record.id)
          }}
          okText="删除"
          cancelText="取消"
        >
          <Button
            type="text"
            size="small"
            icon={<DeleteOutlined />}
            onClick={(e) => e.stopPropagation()}
            style={{ color: '#999' }}
          />
        </Popconfirm>
      ),
    },
  ]

  return (
    <div>
      {/* 标题栏 */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 16,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <HistoryOutlined style={{ fontSize: 18, color: colors.primary }} />
          <Text strong style={{ fontSize: 16 }}>历史记录</Text>
          <Tag color="default">{history.length}</Tag>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <Button
            size="small"
            icon={<ExportOutlined />}
            onClick={handleExportJSON}
          >
            导出
          </Button>
          <Popconfirm
            title="清空所有历史记录？"
            onConfirm={onClear}
            okText="清空"
            cancelText="取消"
          >
            <Button size="small" icon={<ClearOutlined />} danger>
              清空
            </Button>
          </Popconfirm>
        </div>
      </div>

      {/* 历史记录表格 */}
      <Table
        dataSource={history}
        columns={columns}
        rowKey="id"
        size="small"
        pagination={{ pageSize: 10, showSizeChanger: false }}
        onRow={(record) => ({
          onClick: () => onLoad(record),
          style: { cursor: 'pointer' },
        })}
        style={{ marginTop: 8 }}
      />

      <div style={{
        marginTop: 12,
        padding: '8px 12px',
        background: '#f5f5f5',
        borderRadius: 6,
        fontSize: 12,
        color: '#666',
        textAlign: 'center',
      }}>
        💡 点击行查看详细分析结果
      </div>
    </div>
  )
}

// ============================================
// 分析结果展示组件
// ============================================

interface AnalysisResultDisplayProps {
  result: PlanAnalysisResult | null
  topKPlans: PlanAnalysisResult[]
  loading: boolean
  onSelectPlan?: (plan: PlanAnalysisResult) => void
  searchStats?: { evaluated: number; feasible: number; timeMs: number } | null
  errorMsg?: string | null
  // 视图模式（从父组件传入）
  viewMode?: AnalysisViewMode
  onViewModeChange?: (mode: AnalysisViewMode) => void
  // 历史记录相关
  history?: AnalysisHistoryItem[]
  onLoadFromHistory?: (item: AnalysisHistoryItem) => void
  onDeleteHistory?: (id: string) => void
  onClearHistory?: () => void
  // 详情视图功能按钮
  canMapToTopology?: boolean
  onMapToTopology?: () => void
  onClearTraffic?: () => void
  // HeroKPIPanel 需要的数据
  hardware?: HardwareConfig
  model?: LLMModelConfig
  inference?: InferenceConfig
}

type MetricType = 'ttft' | 'tpot' | 'throughput' | 'tps_batch' | 'tps_chip' | 'mfu' | 'mbu' | 'cost' | 'percentiles' | 'bottleneck' | 'e2e' | 'chips' | 'memory' | null

export const AnalysisResultDisplay: React.FC<AnalysisResultDisplayProps> = ({
  result,
  topKPlans,
  loading,
  onSelectPlan,
  searchStats,
  errorMsg,
  viewMode = 'history',
  onViewModeChange: _onViewModeChange,
  history = [],
  onLoadFromHistory,
  onDeleteHistory,
  onClearHistory,
  canMapToTopology,
  onMapToTopology,
  onClearTraffic,
  hardware: _hardware,
  model,
  inference,
}) => {
  const [selectedMetric, setSelectedMetric] = useState<MetricType>(null)
  const [showScoreDetails, setShowScoreDetails] = useState(false)
  const [showBenchmarkDetails, setShowBenchmarkDetails] = useState(false)
  const [selectedParallelism, setSelectedParallelism] = useState<ParallelismType | null>(null)

  // 各章节折叠状态
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    deployment: true,
    model: true,
    performance: true,
    suggestions: true,
    candidates: true,
  })

  // 从历史记录加载（父组件会自动切换到详情视图）
  const handleLoadFromHistory = useCallback((item: AnalysisHistoryItem) => {
    onLoadFromHistory?.(item)
  }, [onLoadFromHistory])

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 40 }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>
          <Text type="secondary">正在搜索最优方案...</Text>
        </div>
      </div>
    )
  }

  if (errorMsg) {
    return (
      <div style={{ padding: 16 }}>
        <div style={{ textAlign: 'center', padding: 20, background: '#fff2f0', borderRadius: 8, border: '1px solid #ffccc7' }}>
          <WarningOutlined style={{ fontSize: 24, color: '#ff4d4f', marginBottom: 8 }} />
          <div style={{ color: '#ff4d4f', fontWeight: 500 }}>{errorMsg}</div>
        </div>
        {searchStats && (
          <div style={{ marginTop: 12, padding: 8, background: '#f5f5f5', borderRadius: 6 }}>
            <Text type="secondary" style={{ fontSize: 11 }}>
              搜索统计: 评估 {searchStats.evaluated} 个方案，{searchStats.feasible} 个可行，耗时 {searchStats.timeMs.toFixed(0)}ms
            </Text>
          </div>
        )}
      </div>
    )
  }

  // 历史列表视图
  if (viewMode === 'history') {
    return (
      <div style={{ padding: 4 }}>
        <HistoryList
          history={history}
          onLoad={handleLoadFromHistory}
          onDelete={onDeleteHistory || (() => {})}
          onClear={onClearHistory || (() => {})}
        />
      </div>
    )
  }

  // 详情视图但没有结果（回退到历史列表）
  if (!result) {
    return (
      <div style={{ padding: 4 }}>
        <HistoryList
          history={history}
          onLoad={handleLoadFromHistory}
          onDelete={onDeleteHistory || (() => {})}
          onClear={onClearHistory || (() => {})}
        />
      </div>
    )
  }

  const { plan, memory, latency, throughput, score, suggestions, is_feasible, infeasibility_reason } = result

  // 指标卡片样式
  const metricCardStyle = (isSelected: boolean): React.CSSProperties => ({
    padding: '12px 10px',
    background: isSelected ? colors.primaryLight : '#fff',
    borderRadius: 8,
    cursor: 'pointer',
    border: isSelected ? `2px solid ${colors.primary}` : `1px solid ${colors.border}`,
    transition: 'all 0.2s ease',
    boxShadow: isSelected ? `0 2px 8px rgba(94, 106, 210, 0.15)` : '0 1px 2px rgba(0, 0, 0, 0.04)',
  })

  return (
    <div>
      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 一、部署方案 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <div style={{ marginBottom: 16 }}>
        <BaseCard
          title="部署方案"
          accentColor="#5E6AD2"
          collapsible
          expanded={expandedSections.deployment}
          onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, deployment: expanded }))}
        >
          {/* 并行策略卡片 */}
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <ParallelismCard
              type="dp"
              value={plan.parallelism.dp}
              selected={selectedParallelism === 'dp'}
              onClick={() => setSelectedParallelism(selectedParallelism === 'dp' ? null : 'dp')}
            />
            <ParallelismCard
              type="tp"
              value={plan.parallelism.tp}
              selected={selectedParallelism === 'tp'}
              onClick={() => setSelectedParallelism(selectedParallelism === 'tp' ? null : 'tp')}
            />
            <ParallelismCard
              type="pp"
              value={plan.parallelism.pp}
              selected={selectedParallelism === 'pp'}
              onClick={() => setSelectedParallelism(selectedParallelism === 'pp' ? null : 'pp')}
            />
            {plan.parallelism.ep > 1 && (
              <ParallelismCard
                type="ep"
                value={plan.parallelism.ep}
                selected={selectedParallelism === 'ep'}
                onClick={() => setSelectedParallelism(selectedParallelism === 'ep' ? null : 'ep')}
              />
            )}
            {plan.parallelism.sp > 1 && (
              <ParallelismCard
                type="sp"
                value={plan.parallelism.sp}
                selected={selectedParallelism === 'sp'}
                onClick={() => setSelectedParallelism(selectedParallelism === 'sp' ? null : 'sp')}
              />
            )}
          </div>

          {/* 芯片数和搜索统计 */}
          <div style={{ fontSize: 13, color: colors.textSecondary, marginBottom: 8 }}>
            <span>总芯片数: <b style={{ color: colors.text }}>{plan.total_chips}</b></span>
            {searchStats && (
              <span style={{ marginLeft: 16 }}>
                搜索: {searchStats.evaluated} 方案 · {searchStats.feasible} 可行 · {searchStats.timeMs.toFixed(0)}ms
              </span>
            )}
            <span style={{ marginLeft: 16, color: '#bbb' }}>点击策略卡片查看详情</span>
          </div>

          {/* 硬件拓扑配置 */}
          {_hardware && (
            <div style={{
              marginBottom: 12,
              padding: '10px 12px',
              background: '#f8f9fa',
              borderRadius: 8,
              border: '1px solid #e8e8e8',
            }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, fontSize: 12 }}>
                {/* Chip配置 */}
                <div>
                  <span style={{ color: '#999' }}>Chip: </span>
                  <b style={{ color: colors.text }}>{_hardware.chip.chip_type}</b>
                  <span style={{ color: '#bbb', marginLeft: 4 }}>
                    ({_hardware.chip.compute_tflops_fp16} TFLOPs, {_hardware.chip.memory_gb}GB, {_hardware.chip.memory_bandwidth_gbps} GB/s)
                  </span>
                </div>
                {/* Board配置 */}
                <div>
                  <span style={{ color: '#999' }}>Board: </span>
                  <b style={{ color: colors.text }}>{_hardware.node.chips_per_node} Chips/Board</b>
                  <span style={{ color: '#bbb', marginLeft: 4 }}>
                    (NVLink {_hardware.node.intra_node_bandwidth_gbps} GB/s)
                  </span>
                </div>
                {/* 总Board数：根据总芯片数和每Board芯片数计算 */}
                <div>
                  <span style={{ color: '#999' }}>总计: </span>
                  <b style={{ color: colors.text }}>{Math.ceil(plan.total_chips / _hardware.node.chips_per_node)} Boards</b>
                  <span style={{ color: '#bbb', marginLeft: 4 }}>
                    (Board间 {_hardware.cluster.inter_node_bandwidth_gbps} GB/s)
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Benchmark 标识 (可点击展开) */}
          {inference && model && (
            <div style={{ marginBottom: 0 }}>
              <div
                style={{
                  fontSize: 13,
                  color: colors.textSecondary,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                }}
                onClick={() => setShowBenchmarkDetails(!showBenchmarkDetails)}
              >
                <span>Benchmark: </span>
                <b style={{ color: colors.text, marginLeft: 8 }}>{generateBenchmarkName(model, inference)}</b>
                <span style={{ marginLeft: 8, fontSize: 10, color: '#bbb' }}>
                  {showBenchmarkDetails ? '▲ 收起' : '▼ 展开'}
                </span>
              </div>
              {showBenchmarkDetails && (
                <div style={{
                  marginTop: 12,
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: 12,
                }}>
                  {parseBenchmarkParts(model, inference).map((part, idx) => (
                    <div key={idx} style={{
                      padding: '12px 16px',
                      background: '#fafafa',
                      borderRadius: 8,
                      border: '1px solid #e8e8e8',
                      minWidth: 100,
                    }}>
                      <div style={{ color: colors.primary, fontWeight: 600, fontSize: 18, marginBottom: 4, textAlign: 'center' }}>{part.key}</div>
                      <div style={{ fontSize: 13 }}>
                        <span style={{ color: '#999' }}>{part.label}：</span>
                        <span style={{ color: colors.text, fontWeight: 500 }}>{part.value}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* 并行策略详细介绍 */}
          {selectedParallelism && (
            <div style={{ marginBottom: 12 }}>
              <ParallelismInfo type={selectedParallelism} />
            </div>
          )}

          {/* 拓扑映射操作 */}
          {canMapToTopology && (
            <div style={{
              marginTop: 12,
              paddingTop: 12,
              borderTop: `1px dashed ${colors.borderLight}`,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <Text style={{ fontSize: 11, color: colors.textSecondary }}>
                将并行策略映射到拓扑视图，查看通信流量分布
              </Text>
              <div style={{ display: 'flex', gap: 6 }}>
                <Button
                  size="small"
                  type="primary"
                  onClick={onMapToTopology}
                  style={{ fontSize: 11 }}
                >
                  映射到拓扑
                </Button>
                <Button
                  size="small"
                  onClick={onClearTraffic}
                  style={{ fontSize: 11 }}
                >
                  清除映射
                </Button>
              </div>
            </div>
          )}
        </BaseCard>
      </div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 二、模型架构 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      {model && (
        <div style={{ marginBottom: 16 }}>
          <BaseCard
            title="模型架构"
            accentColor="#13c2c2"
            collapsible
            expanded={expandedSections.model}
            onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, model: expanded }))}
          >
            <ModelInfoCard model={model} inference={inference} />
          </BaseCard>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 三、性能分析 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <div style={{ marginBottom: 16 }}>
        <BaseCard
          title="性能分析"
          accentColor="#52c41a"
          collapsible
          expanded={expandedSections.performance}
          onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, performance: expanded }))}
        >
        <>
        {/* 延迟指标 */}
        <Text style={{ fontSize: 13, fontWeight: 500, color: colors.text, display: 'block', marginBottom: 8 }}>延迟</Text>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 12 }}>
          <div style={{ ...metricCardStyle(selectedMetric === 'ttft'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'ttft' ? null : 'ttft')}>
            <InfoCircleOutlined style={{ position: 'absolute', top: 8, right: 8, fontSize: 10, color: selectedMetric === 'ttft' ? colors.primary : '#d9d9d9' }} />
            <Text style={{ fontSize: 13, color: colors.textSecondary }}>FTL</Text>
            <div style={{ fontSize: 18, fontWeight: 600, color: colors.text, marginTop: 4 }}>
              {latency.prefill_total_latency_ms.toFixed(1)} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>ms</span>
            </div>
          </div>
          <div style={{ ...metricCardStyle(selectedMetric === 'tpot'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'tpot' ? null : 'tpot')}>
            <InfoCircleOutlined style={{ position: 'absolute', top: 8, right: 8, fontSize: 10, color: selectedMetric === 'tpot' ? colors.primary : '#d9d9d9' }} />
            <Text style={{ fontSize: 13, color: colors.textSecondary }}>TPOT</Text>
            <div style={{ fontSize: 18, fontWeight: 600, color: colors.text, marginTop: 4 }}>
              {latency.decode_per_token_latency_ms.toFixed(2)} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>ms</span>
            </div>
          </div>
          <div style={{ ...metricCardStyle(selectedMetric === 'e2e'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'e2e' ? null : 'e2e')}>
            <InfoCircleOutlined style={{ position: 'absolute', top: 8, right: 8, fontSize: 10, color: selectedMetric === 'e2e' ? colors.primary : '#d9d9d9' }} />
            <Text style={{ fontSize: 13, color: colors.textSecondary }}>E2E</Text>
            <div style={{ fontSize: 18, fontWeight: 600, color: colors.text, marginTop: 4 }}>
              {(latency.end_to_end_latency_ms / 1000).toFixed(2)} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>s</span>
            </div>
          </div>
          <div style={{ ...metricCardStyle(selectedMetric === 'percentiles'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'percentiles' ? null : 'percentiles')}>
            <InfoCircleOutlined style={{ position: 'absolute', top: 8, right: 8, fontSize: 10, color: selectedMetric === 'percentiles' ? colors.primary : '#d9d9d9' }} />
            <Text style={{ fontSize: 13, color: colors.textSecondary }}>P99</Text>
            <div style={{ fontSize: 18, fontWeight: 600, color: latency.ttft_percentiles && latency.ttft_percentiles.p99 > 450 ? colors.error : colors.text, marginTop: 4 }}>
              {latency.ttft_percentiles ? latency.ttft_percentiles.p99.toFixed(0) : '-'} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>ms</span>
            </div>
          </div>
        </div>

        {/* 吞吐与效率 */}
        <Text style={{ fontSize: 13, fontWeight: 500, color: colors.text, display: 'block', marginBottom: 8 }}>吞吐与效率</Text>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 8 }}>
          <Tooltip title="Total TPS = TPS_chip × NumChips，集群总吞吐">
            <div style={{ ...metricCardStyle(selectedMetric === 'throughput'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'throughput' ? null : 'throughput')}>
              <InfoCircleOutlined style={{ position: 'absolute', top: 8, right: 8, fontSize: 10, color: selectedMetric === 'throughput' ? colors.primary : '#d9d9d9' }} />
              <Text style={{ fontSize: 13, color: colors.textSecondary }}>Total TPS</Text>
              <div style={{ fontSize: 18, fontWeight: 600, color: colors.text, marginTop: 4 }}>
                {throughput.tokens_per_second.toFixed(0)} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>tok/s</span>
              </div>
            </div>
          </Tooltip>
          <Tooltip title="TPS per Batch = 1000 / TPOT(ms)，用户体验指标，SLO约束 ≥10">
            <div style={{ ...metricCardStyle(selectedMetric === 'tps_batch'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'tps_batch' ? null : 'tps_batch')}>
              <InfoCircleOutlined style={{ position: 'absolute', top: 8, right: 8, fontSize: 10, color: selectedMetric === 'tps_batch' ? colors.primary : '#d9d9d9' }} />
              <Text style={{ fontSize: 13, color: colors.textSecondary }}>TPS/Batch</Text>
              <div style={{ fontSize: 18, fontWeight: 600, color: throughput.tps_per_batch >= 10 ? colors.text : colors.error, marginTop: 4 }}>
                {throughput.tps_per_batch.toFixed(1)} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>tok/s</span>
              </div>
            </div>
          </Tooltip>
          <Tooltip title="TPS per Chip = B × TPS_batch，成本效益优化目标">
            <div style={{ ...metricCardStyle(selectedMetric === 'tps_chip'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'tps_chip' ? null : 'tps_chip')}>
              <InfoCircleOutlined style={{ position: 'absolute', top: 8, right: 8, fontSize: 10, color: selectedMetric === 'tps_chip' ? colors.primary : '#d9d9d9' }} />
              <Text style={{ fontSize: 13, color: colors.textSecondary }}>TPS/Chip</Text>
              <div style={{ fontSize: 18, fontWeight: 600, color: colors.text, marginTop: 4 }}>
                {throughput.tps_per_chip.toFixed(0)} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>tok/s</span>
              </div>
            </div>
          </Tooltip>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8, marginBottom: 12 }}>
          <div style={{ ...metricCardStyle(selectedMetric === 'mfu'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'mfu' ? null : 'mfu')}>
            <InfoCircleOutlined style={{ position: 'absolute', top: 8, right: 8, fontSize: 10, color: selectedMetric === 'mfu' ? colors.primary : '#d9d9d9' }} />
            <Text style={{ fontSize: 13, color: colors.textSecondary }}>MFU</Text>
            <div style={{ fontSize: 18, fontWeight: 600, color: colors.text, marginTop: 4 }}>
              {(throughput.model_flops_utilization * 100).toFixed(1)} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>%</span>
            </div>
          </div>
          <div style={{ ...metricCardStyle(selectedMetric === 'mbu'), textAlign: 'center', position: 'relative' }} onClick={() => setSelectedMetric(selectedMetric === 'mbu' ? null : 'mbu')}>
            <InfoCircleOutlined style={{ position: 'absolute', top: 8, right: 8, fontSize: 10, color: selectedMetric === 'mbu' ? colors.primary : '#d9d9d9' }} />
            <Text style={{ fontSize: 13, color: colors.textSecondary }}>MBU</Text>
            <div style={{ fontSize: 18, fontWeight: 600, color: colors.text, marginTop: 4 }}>
              {(throughput.memory_bandwidth_utilization * 100).toFixed(1)} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>%</span>
            </div>
          </div>
        </div>

        {/* 资源利用 */}
        <Text style={{ fontSize: 13, fontWeight: 500, color: colors.text, display: 'block', marginBottom: 8 }}>资源利用</Text>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 12 }}>
          {/* 显存占用 */}
          <div
            style={{ ...metricCardStyle(selectedMetric === 'memory'), textAlign: 'center', position: 'relative' }}
            onClick={() => setSelectedMetric(selectedMetric === 'memory' ? null : 'memory')}
          >
            <InfoCircleOutlined style={{ position: 'absolute', top: 8, right: 8, fontSize: 10, color: selectedMetric === 'memory' ? colors.primary : '#d9d9d9' }} />
            <Text style={{ fontSize: 13, color: colors.textSecondary }}>显存占用</Text>
            <div style={{ fontSize: 18, fontWeight: 600, color: memory.is_memory_sufficient ? colors.text : colors.error, marginTop: 4 }}>
              {memory.total_per_chip_gb.toFixed(1)} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>/ 80G</span>
            </div>
            {/* <div style={{ fontSize: 10, color: colors.textSecondary, marginTop: 4 }}> */}
              {/* 模型{memory.model_memory_gb.toFixed(1)} · KV{memory.kv_cache_memory_gb.toFixed(1)} · 激活{memory.activation_memory_gb.toFixed(1)} */}
            {/* </div> */}
          </div>
          {/* 推理成本 */}
          <div
            style={{ ...metricCardStyle(selectedMetric === 'cost'), textAlign: 'center', position: 'relative' }}
            onClick={() => setSelectedMetric(selectedMetric === 'cost' ? null : 'cost')}
          >
            <InfoCircleOutlined style={{ position: 'absolute', top: 8, right: 8, fontSize: 10, color: selectedMetric === 'cost' ? colors.primary : '#d9d9d9' }} />
            <Text style={{ fontSize: 13, color: colors.textSecondary }}>推理成本</Text>
            <div style={{ fontSize: 18, fontWeight: 600, color: colors.text, marginTop: 4 }}>
              ${result.cost ? result.cost.cost_per_million_tokens.toFixed(3) : '-'} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>/M</span>
            </div>
          </div>
        </div>

        {/* 综合评分 + 瓶颈分析 */}
        <div style={{ display: 'flex', alignItems: 'stretch', gap: 12, marginTop: 16, paddingTop: 16, borderTop: `1px dashed ${colors.borderLight}` }}>
          {/* 综合评分 */}
          <div
            style={{
              padding: '12px 20px',
              background: is_feasible ? '#f6ffed' : '#fff2f0',
              border: `1.5px solid ${is_feasible ? '#b7eb8f' : '#ffccc7'}`,
              borderRadius: 8,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}
            onClick={() => setShowScoreDetails(!showScoreDetails)}
          >
            {is_feasible ? (
              <CheckCircleOutlined style={{ color: colors.success, fontSize: 18 }} />
            ) : (
              <Tooltip title={infeasibility_reason}>
                <WarningOutlined style={{ color: colors.error, fontSize: 18 }} />
              </Tooltip>
            )}
            <div>
              <Text strong style={{ fontSize: 24, color: is_feasible ? colors.success : colors.error, lineHeight: 1 }}>
                {score.overall_score.toFixed(1)}
              </Text>
              <span style={{ fontSize: 13, color: colors.textSecondary, marginLeft: 4 }}>分</span>
            </div>
            <div style={{ fontSize: 12, color: colors.textSecondary }}>
              综合评分 {showScoreDetails ? '▲' : '▼'}
            </div>
          </div>

          {/* 瓶颈分析 */}
          <div
            style={{
              flex: 1,
              padding: '12px 16px',
              background: selectedMetric === 'bottleneck' ? colors.warningLight : '#fafafa',
              borderRadius: 8,
              cursor: 'pointer',
              border: selectedMetric === 'bottleneck' ? `2px solid ${colors.warning}` : `1px solid ${colors.border}`,
              transition: 'all 0.2s ease',
            }}
            onClick={() => setSelectedMetric(selectedMetric === 'bottleneck' ? null : 'bottleneck')}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
              <Tag color={
                latency.bottleneck_type === 'compute' ? 'orange' :
                latency.bottleneck_type === 'memory' ? 'blue' :
                latency.bottleneck_type === 'communication' ? 'purple' :
                latency.bottleneck_type === 'balanced' ? 'green' : 'default'
              } style={{ margin: 0 }}>
                {latency.bottleneck_type === 'compute' ? '算力瓶颈' :
                 latency.bottleneck_type === 'memory' ? '访存瓶颈' :
                 latency.bottleneck_type === 'communication' ? '通信瓶颈' :
                 latency.bottleneck_type === 'balanced' ? '均衡状态' : latency.bottleneck_type}
              </Tag>
              {latency.bottleneck_analysis && (
                <Text style={{ fontSize: 11, color: colors.textSecondary }}>
                  {latency.bottleneck_analysis.dominant_phase === 'prefill' ? 'Prefill主导' : 'Decode主导'}
                </Text>
              )}
            </div>
            {latency.bottleneck_analysis && (
              <>
                <div style={{ display: 'flex', height: 6, borderRadius: 3, overflow: 'hidden', background: '#e8e8e8' }}>
                  {(() => {
                    const analysis = latency.bottleneck_analysis.dominant_phase === 'prefill'
                      ? latency.bottleneck_analysis.prefill
                      : latency.bottleneck_analysis.decode;
                    return (
                      <>
                        <div style={{ width: `${analysis.compute_ratio * 100}%`, background: '#faad14' }} />
                        <div style={{ width: `${analysis.memory_ratio * 100}%`, background: '#1890ff' }} />
                        <div style={{ width: `${analysis.comm_ratio * 100}%`, background: '#722ed1' }} />
                      </>
                    );
                  })()}
                </div>
                <div style={{ display: 'flex', gap: 12, marginTop: 4, fontSize: 10, color: colors.textSecondary }}>
                  {(() => {
                    const analysis = latency.bottleneck_analysis.dominant_phase === 'prefill'
                      ? latency.bottleneck_analysis.prefill
                      : latency.bottleneck_analysis.decode;
                    return (
                      <>
                        <span><span style={{ display: 'inline-block', width: 6, height: 6, background: '#faad14', borderRadius: 1, marginRight: 3, verticalAlign: 'middle' }} />计算{(analysis.compute_ratio * 100).toFixed(0)}%</span>
                        <span><span style={{ display: 'inline-block', width: 6, height: 6, background: '#1890ff', borderRadius: 1, marginRight: 3, verticalAlign: 'middle' }} />访存{(analysis.memory_ratio * 100).toFixed(0)}%</span>
                        <span><span style={{ display: 'inline-block', width: 6, height: 6, background: '#722ed1', borderRadius: 1, marginRight: 3, verticalAlign: 'middle' }} />通信{(analysis.comm_ratio * 100).toFixed(0)}%</span>
                      </>
                    );
                  })()}
                </div>
              </>
            )}
          </div>
        </div>

        {/* 评分详情展开区域 */}
        {showScoreDetails && (
          <div style={{ marginTop: 12, padding: 12, background: '#fafafa', borderRadius: 8 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 12 }}>
              <div style={{ textAlign: 'center', padding: 8, background: '#f0f5ff', borderRadius: 6 }}>
                <ClockCircleOutlined style={{ color: '#1890ff', fontSize: 14 }} />
                <div style={{ fontSize: 16, fontWeight: 600, color: '#1890ff', margin: '4px 0' }}>{score.latency_score.toFixed(0)}</div>
                <div style={{ fontSize: 10, color: colors.textSecondary }}>延迟 {(DEFAULT_SCORE_WEIGHTS.latency * 100).toFixed(0)}%</div>
              </div>
              <div style={{ textAlign: 'center', padding: 8, background: '#f6ffed', borderRadius: 6 }}>
                <ThunderboltOutlined style={{ color: '#52c41a', fontSize: 14 }} />
                <div style={{ fontSize: 16, fontWeight: 600, color: '#52c41a', margin: '4px 0' }}>{score.throughput_score.toFixed(0)}</div>
                <div style={{ fontSize: 10, color: colors.textSecondary }}>吞吐 {(DEFAULT_SCORE_WEIGHTS.throughput * 100).toFixed(0)}%</div>
              </div>
              <div style={{ textAlign: 'center', padding: 8, background: '#fff7e6', borderRadius: 6 }}>
                <DashboardOutlined style={{ color: '#faad14', fontSize: 14 }} />
                <div style={{ fontSize: 16, fontWeight: 600, color: '#faad14', margin: '4px 0' }}>{score.efficiency_score.toFixed(0)}</div>
                <div style={{ fontSize: 10, color: colors.textSecondary }}>效率 {(DEFAULT_SCORE_WEIGHTS.efficiency * 100).toFixed(0)}%</div>
              </div>
              <div style={{ textAlign: 'center', padding: 8, background: '#f9f0ff', borderRadius: 6 }}>
                <AimOutlined style={{ color: '#722ed1', fontSize: 14 }} />
                <div style={{ fontSize: 16, fontWeight: 600, color: '#722ed1', margin: '4px 0' }}>{score.balance_score.toFixed(0)}</div>
                <div style={{ fontSize: 10, color: colors.textSecondary }}>均衡 {(DEFAULT_SCORE_WEIGHTS.balance * 100).toFixed(0)}%</div>
              </div>
            </div>
            <div style={{ fontSize: 11, color: colors.textSecondary, textAlign: 'center', fontFamily: 'monospace' }}>
              综合 = {(DEFAULT_SCORE_WEIGHTS.latency * 100).toFixed(0)}%×延迟 + {(DEFAULT_SCORE_WEIGHTS.throughput * 100).toFixed(0)}%×吞吐 + {(DEFAULT_SCORE_WEIGHTS.efficiency * 100).toFixed(0)}%×效率 + {(DEFAULT_SCORE_WEIGHTS.balance * 100).toFixed(0)}%×均衡
            </div>
          </div>
        )}

        {/* 指标详情展示 - 内嵌在性能分析中 */}
        {selectedMetric && (
          <div style={{ marginTop: 16, paddingTop: 16, borderTop: `1px dashed ${colors.borderLight}` }}>
            <MetricDetailCard metric={selectedMetric} result={result} />
          </div>
        )}
        </>
        </BaseCard>
      </div>

      {/* 优化建议 */}
      {suggestions.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <BaseCard
            title="优化建议"
            accentColor="#722ed1"
            collapsible
            expanded={expandedSections.suggestions}
            onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, suggestions: expanded }))}
          >
            {suggestions.slice(0, 3).map((s, i) => (
              <div key={i} style={{
                padding: 10,
                background: '#fff',
                borderRadius: 8,
                marginBottom: 8,
                borderLeft: `3px solid ${s.priority <= 2 ? colors.error : s.priority <= 3 ? colors.warning : colors.primary}`,
                border: `1px solid ${colors.border}`,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <Text style={{ fontSize: 12, color: colors.text, flex: 1 }}>{s.description}</Text>
                  <Tag
                    style={{
                      fontSize: 9,
                      padding: '0 6px',
                      borderRadius: 4,
                      border: 'none',
                      background: s.priority <= 2 ? colors.errorLight : s.priority <= 3 ? colors.warningLight : colors.primaryLight,
                      color: s.priority <= 2 ? colors.error : s.priority <= 3 ? colors.warning : colors.primary,
                      marginLeft: 8,
                    }}
                  >
                    P{s.priority}
                  </Tag>
                </div>
                <Text style={{ fontSize: 10, color: colors.textSecondary, marginTop: 4, display: 'block' }}>预期: {s.expected_improvement}</Text>
              </div>
            ))}
          </BaseCard>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 八、候选方案 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      {topKPlans.length > 1 && (
        <div style={{ marginBottom: 16 }}>
          <BaseCard
            title="候选方案"
            subtitle={`${topKPlans.length}个`}
            accentColor="#1890ff"
            collapsible
            expanded={expandedSections.candidates}
            onExpandChange={(expanded) => setExpandedSections(prev => ({ ...prev, candidates: expanded }))}
          >
            <div style={{ maxHeight: 200, overflow: 'auto' }}>
            {topKPlans.map((p, i) => {
              const isSelected = p.plan.plan_id === result?.plan.plan_id
              return (
                <div
                  key={p.plan.plan_id}
                  onClick={() => onSelectPlan?.(p)}
                  style={{
                    padding: 10,
                    background: isSelected ? colors.primaryLight : '#fff',
                    borderRadius: 8,
                    marginBottom: 6,
                    cursor: 'pointer',
                    border: isSelected ? `2px solid ${colors.primary}` : `1px solid ${colors.border}`,
                    transition: 'all 0.2s ease',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{
                        fontSize: 11,
                        fontWeight: 600,
                        color: isSelected ? colors.primary : colors.textSecondary,
                        minWidth: 20,
                      }}>
                        #{i + 1}
                      </span>
                      <div style={{ display: 'flex', gap: 3 }}>
                        <span style={{ fontSize: 10, color: colors.textSecondary }}>DP{p.plan.parallelism.dp}</span>
                        <span style={{ fontSize: 10, color: colors.textSecondary }}>·</span>
                        <span style={{ fontSize: 10, color: colors.textSecondary }}>TP{p.plan.parallelism.tp}</span>
                        <span style={{ fontSize: 10, color: colors.textSecondary }}>·</span>
                        <span style={{ fontSize: 10, color: colors.textSecondary }}>PP{p.plan.parallelism.pp}</span>
                        {p.plan.parallelism.ep > 1 && (
                          <>
                            <span style={{ fontSize: 10, color: colors.textSecondary }}>·</span>
                            <span style={{ fontSize: 10, color: colors.textSecondary }}>EP{p.plan.parallelism.ep}</span>
                          </>
                        )}
                      </div>
                    </div>
                    <Text style={{ fontSize: 14, fontWeight: 600, color: isSelected ? colors.primary : colors.text }}>
                      {p.score.overall_score.toFixed(1)}
                    </Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 10, color: colors.textSecondary }}>
                    <span>{p.latency.prefill_total_latency_ms.toFixed(1)}ms</span>
                    <span>{p.throughput.tokens_per_second.toFixed(0)} tok/s</span>
                    <span>{(p.throughput.model_flops_utilization * 100).toFixed(1)}%</span>
                  </div>
                </div>
              )
            })}
            </div>
          </BaseCard>
        </div>
      )}

    </div>
  )
}

export default AnalysisResultDisplay
