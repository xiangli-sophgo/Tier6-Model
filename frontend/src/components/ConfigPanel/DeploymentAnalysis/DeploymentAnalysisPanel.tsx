/**
 * LLM 部署分析面板
 *
 * 提供模型配置、推理配置、硬件配置、并行策略配置和分析结果展示
 */

import React, { useState, useCallback, useRef } from 'react'
import {
  Typography,
  Button,
  Select,
  Spin,
  message,
  InputNumber,
  Tooltip,
} from 'antd'
import {
  PlayCircleOutlined,
  SearchOutlined,
  WarningOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons'
import {
  LLMModelConfig,
  InferenceConfig,
  HardwareConfig,
  ParallelismStrategy,
  PlanAnalysisResult,
  SearchConstraints,
  TopologyTrafficResult,
  ProtocolConfig,
  NetworkInfraConfig,
  DEFAULT_PROTOCOL_CONFIG,
  DEFAULT_NETWORK_CONFIG,
} from '../../../utils/llmDeployment/types'
import { HierarchicalTopology } from '../../../types'
import {
  MODEL_PRESETS,
  INFERENCE_PRESETS,
  getBackendChipPresets,
} from '../../../utils/llmDeployment/presets'
import { simulateBackend, searchWithFixedChips, InfeasibleResult, SearchResult } from '../../../utils/llmDeployment'
import { adaptSimulationResult } from '../../../utils/llmDeployment/resultAdapter'
import { analyzeTopologyTraffic } from '../../../utils/llmDeployment/trafficMapper'
import {
  extractChipGroupsFromConfig,
  generateHardwareConfigFromPanelConfig,
  ChipGroupInfo,
} from '../../../utils/llmDeployment/topologyHardwareExtractor'
import { RackConfig, DeploymentAnalysisData, AnalysisHistoryItem, AnalysisViewMode } from '../shared'
import {
  BenchmarkConfigSelector,
  colors,
  configRowStyle,
} from './ConfigSelectors'
import { BaseCard } from '../../common/BaseCard'
import { ParallelismConfigPanel } from './ParallelismConfigPanel'
import { AnalysisResultDisplay } from './AnalysisResultDisplay'

const { Text } = Typography

// ============================================
// 主面板组件
// ============================================

interface DeploymentAnalysisPanelProps {
  topology?: HierarchicalTopology | null
  onTrafficResultChange?: (result: TopologyTrafficResult | null) => void
  onAnalysisDataChange?: (data: DeploymentAnalysisData | null) => void
  rackConfig?: RackConfig
  podCount?: number
  racksPerPod?: number
  // 历史记录 (由 WorkbenchContext 统一管理)
  history?: AnalysisHistoryItem[]
  onAddToHistory?: (item: Omit<AnalysisHistoryItem, 'id' | 'timestamp'>) => void
  onDeleteHistory?: (id: string) => void
  onClearHistory?: () => void
}

export const DeploymentAnalysisPanel: React.FC<DeploymentAnalysisPanelProps> = ({
  topology,
  onTrafficResultChange,
  onAnalysisDataChange,
  rackConfig,
  podCount = 1,
  racksPerPod = 1,
  // 历史记录 props
  history = [],
  onAddToHistory,
  onDeleteHistory,
  onClearHistory,
}) => {
  // 模型配置状态
  const [modelConfig, setModelConfig] = useState<LLMModelConfig>(
    MODEL_PRESETS['deepseek-v3']
  )

  // 推理配置状态
  const [inferenceConfig, setInferenceConfig] = useState<InferenceConfig>(
    INFERENCE_PRESETS['standard']
  )

  // 从拓扑配置提取的芯片组
  const [chipGroups, setChipGroups] = useState<ChipGroupInfo[]>([])
  const [selectedChipType, setSelectedChipType] = useState<string | undefined>()

  // 硬件配置状态（初始为 null，等待后端芯片预设加载完成）
  const [hardwareConfig, setHardwareConfig] = useState<HardwareConfig | null>(null)

  // 初始化硬件配置：等待后端芯片预设加载完成
  React.useEffect(() => {
    const backendPresets = getBackendChipPresets()
    // 如果后端预设已加载且硬件配置还未初始化
    if (Object.keys(backendPresets).length > 0 && !hardwareConfig) {
      // 从拓扑配置中提取硬件配置
      if (rackConfig && rackConfig.boards.length > 0 && topology?.connections) {
        const groups = extractChipGroupsFromConfig(rackConfig.boards)
        if (groups.length > 0) {
          const firstChipType = groups[0].presetId || groups[0].chipType
          const config = generateHardwareConfigFromPanelConfig(
            podCount,
            racksPerPod,
            rackConfig.boards.map(b => ({ chips: b.chips, count: b.count })),
            topology.connections,
            firstChipType
          )
          if (config) {
            setHardwareConfig(config)
          }
        }
      }
    }
  }, [hardwareConfig, rackConfig, topology, podCount, racksPerPod])

  // 序列化 rackConfig 用于深度比较
  const rackConfigJson = React.useMemo(() =>
    rackConfig ? JSON.stringify(rackConfig) : '',
    [rackConfig]
  )

  // 当拓扑配置变化时，提取芯片组信息并更新硬件配置
  React.useEffect(() => {
    if (!rackConfig || rackConfig.boards.length === 0) {
      setChipGroups([])
      return
    }

    const groups = extractChipGroupsFromConfig(rackConfig.boards)
    setChipGroups(groups)

    // 默认选择第一个芯片类型
    if (groups.length > 0 && !selectedChipType) {
      setSelectedChipType(groups[0].presetId || groups[0].chipType)
    }

    // 立即更新硬件配置
    if (groups.length > 0) {
      const currentSelectedType = selectedChipType || groups[0].presetId || groups[0].chipType
      // 使用 generateHardwareConfigFromPanelConfig 从连接配置中提取带宽和延迟
      const connections = topology?.connections || []
      const config = generateHardwareConfigFromPanelConfig(
        podCount,
        racksPerPod,
        rackConfig.boards.map(b => ({ chips: b.chips, count: b.count })),
        connections,
        currentSelectedType
      )
      if (config) {
        setHardwareConfig(config)
      }
    }
  }, [rackConfigJson, podCount, racksPerPod, topology?.connections])

  // 当选择的芯片类型变化时，更新硬件配置
  React.useEffect(() => {
    if (rackConfig && chipGroups.length > 0 && selectedChipType) {
      // 使用 generateHardwareConfigFromPanelConfig 从连接配置中提取带宽和延迟
      const connections = topology?.connections || []
      const config = generateHardwareConfigFromPanelConfig(
        podCount,
        racksPerPod,
        rackConfig.boards.map(b => ({ chips: b.chips, count: b.count })),
        connections,
        selectedChipType
      )
      if (config) {
        setHardwareConfig(config)
      }
    }
  }, [selectedChipType, rackConfig, chipGroups, podCount, racksPerPod, topology?.connections])

  // 并行策略状态
  const [parallelismMode, setParallelismMode] = useState<'manual' | 'auto'>('auto')
  const [manualStrategy, setManualStrategy] = useState<ParallelismStrategy>({
    dp: 1, tp: 8, pp: 1, ep: 1, sp: 1,
  })
  const [searchConstraints, setSearchConstraints] = useState<SearchConstraints>({
    max_chips: 8,
    tp_within_node: true,
  })

  // 运行时配置 (协议和网络参数)
  const [protocolConfig, setProtocolConfig] = useState<ProtocolConfig>({ ...DEFAULT_PROTOCOL_CONFIG })
  const [networkConfig, setNetworkConfig] = useState<NetworkInfraConfig>({ ...DEFAULT_NETWORK_CONFIG })

  // 分析结果状态
  const [analysisResult, setAnalysisResult] = useState<PlanAnalysisResult | null>(null)
  const [topKPlans, setTopKPlans] = useState<PlanAnalysisResult[]>([])
  const [infeasiblePlans, setInfeasiblePlans] = useState<InfeasibleResult[]>([])
  const [searchStats, setSearchStats] = useState<{ evaluated: number; feasible: number; timeMs: number } | null>(null)
  const [loading, setLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // 搜索进度状态
  const [searchProgress, setSearchProgress] = useState<{
    stage: 'idle' | 'generating' | 'evaluating' | 'completed' | 'cancelled'
    totalCandidates: number
    currentEvaluating: number
    evaluated: number
  }>({
    stage: 'idle',
    totalCandidates: 0,
    currentEvaluating: 0,
    evaluated: 0,
  })

  // 取消控制器
  const abortControllerRef = useRef<AbortController | null>(null)

  // 视图模式状态（历史列表 或 详情）
  const [viewMode, setViewMode] = useState<AnalysisViewMode>('history')

  // 当前显示的分析结果对应的配置（区分于配置面板的当前选择）
  const [displayModelConfig, setDisplayModelConfig] = useState<LLMModelConfig | null>(null)
  const [displayInferenceConfig, setDisplayInferenceConfig] = useState<InferenceConfig | null>(null)

  // 映射到拓扑的回调
  const handleMapToTopology = React.useCallback(() => {
    if (!analysisResult || !topology || !onTrafficResultChange) return
    try {
      const strategy = analysisResult.plan.parallelism
      const trafficResult = analyzeTopologyTraffic(
        topology,
        strategy,
        analysisResult.communication
      )
      onTrafficResultChange(trafficResult)
    } catch (error) {
      console.error('流量映射失败:', error)
      onTrafficResultChange(null)
    }
  }, [analysisResult, topology, onTrafficResultChange])

  // 从历史记录加载
  const handleLoadFromHistory = useCallback((item: AnalysisHistoryItem) => {
    setModelConfig(item.modelConfig)
    setInferenceConfig(item.inferenceConfig)
    setHardwareConfig(item.hardwareConfig)
    setManualStrategy(item.parallelism)
    setParallelismMode(item.searchMode === 'auto' ? 'auto' : 'manual')
    setAnalysisResult(item.result)
    setTopKPlans(item.topKPlans && item.topKPlans.length > 0 ? item.topKPlans : [item.result])
    // 设置显示配置（分析时使用的配置，不随配置面板变化）
    setDisplayModelConfig(item.modelConfig)
    setDisplayInferenceConfig(item.inferenceConfig)
    setViewMode('detail')  // 切换到详情视图
    const plansCount = item.topKPlans?.length ?? 1
    message.success(`已加载历史记录${plansCount > 1 ? `（含 ${plansCount} 个候选方案）` : ''}`)
  }, [])

  // 删除历史记录 (使用 props 回调)
  const handleDeleteHistory = useCallback((id: string) => {
    onDeleteHistory?.(id)
    message.success('已删除')
  }, [onDeleteHistory])

  // 清空历史记录 (使用 props 回调)
  const handleClearHistory = useCallback(() => {
    onClearHistory?.()
    message.success('已清空历史记录')
  }, [onClearHistory])

  // 取消评估
  const handleCancelAnalysis = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
      setSearchProgress(prev => ({ ...prev, stage: 'cancelled' }))
      setLoading(false)
      message.warning({ content: '评估已取消', key: 'search', duration: 2 })
    }
  }, [])

  // 当分析状态变化时，通知父组件
  React.useEffect(() => {
    if (onAnalysisDataChange && hardwareConfig) {
      onAnalysisDataChange({
        result: analysisResult,
        topKPlans,
        infeasiblePlans,
        hardware: hardwareConfig,
        // 使用显示配置（分析时的配置），而不是配置面板当前选择
        model: displayModelConfig || modelConfig,
        inference: displayInferenceConfig || inferenceConfig,
        loading,
        errorMsg,
        searchStats,
        searchProgress,
        onCancelEvaluation: handleCancelAnalysis,
        onSelectPlan: (plan) => setAnalysisResult(plan),
        onMapToTopology: handleMapToTopology,
        onClearTraffic: () => onTrafficResultChange?.(null),
        canMapToTopology: !!(analysisResult && topology && onTrafficResultChange),
        // 视图模式
        viewMode,
        onViewModeChange: setViewMode,
        // 历史记录相关
        history,
        onLoadFromHistory: handleLoadFromHistory,
        onDeleteHistory: handleDeleteHistory,
        onClearHistory: handleClearHistory,
      })
    }
  }, [analysisResult, topKPlans, infeasiblePlans, hardwareConfig, displayModelConfig, displayInferenceConfig, modelConfig, inferenceConfig, loading, errorMsg, searchStats, searchProgress, onAnalysisDataChange, handleMapToTopology, handleCancelAnalysis, topology, onTrafficResultChange, viewMode, history, handleLoadFromHistory, handleDeleteHistory, handleClearHistory])

  // 计算最大可用芯片数
  const maxChips = hardwareConfig ? hardwareConfig.node.chips_per_node * hardwareConfig.cluster.num_nodes : 0

  // 运行分析
  const handleRunAnalysis = useCallback(async () => {
    if (!hardwareConfig) return // 等待硬件配置加载
    if (!topology) {
      setErrorMsg('拓扑配置未加载，请先配置拓扑')
      return
    }

    // 取消之前的评估
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    abortControllerRef.current = new AbortController()

    setAnalysisResult(null)
    setTopKPlans([])
    setInfeasiblePlans([])
    setSearchStats(null)
    setErrorMsg(null)
    setLoading(true)
    const startTime = Date.now()

    // 实时结果收集器
    const realtimeFeasible: SearchResult[] = []
    const realtimeInfeasible: InfeasibleResult[] = []

    try {
      let result: PlanAnalysisResult | null = null
      // 使用局部变量捕获新的topKPlans，避免React状态更新延迟导致保存到历史时为空
      let newTopKPlans: PlanAnalysisResult[] = []

      if (parallelismMode === 'manual') {
        // 手动模式：直接调用后端模拟
        message.loading({ content: '正在调用后端模拟...', key: 'simulate', duration: 0 })
        const simulation = await simulateBackend(
          topology,
          modelConfig,
          inferenceConfig,
          manualStrategy,
          hardwareConfig
        )
        message.success({ content: '模拟完成', key: 'simulate', duration: 2 })

        // 适配为 PlanAnalysisResult 格式
        result = adaptSimulationResult(simulation, modelConfig, inferenceConfig, manualStrategy, hardwareConfig)
        setAnalysisResult(result)
        newTopKPlans = [result]
        setTopKPlans(newTopKPlans)
      } else {
        // 自动模式：搜索多个方案
        setSearchProgress({ stage: 'generating', totalCandidates: 0, currentEvaluating: 0, evaluated: 0 })

        const searchResults = await searchWithFixedChips(
          topology,
          modelConfig,
          inferenceConfig,
          hardwareConfig,
          searchConstraints.max_chips || maxChips,
          {
            maxPlans: 10,
            abortSignal: abortControllerRef.current.signal,
            onCandidatesGenerated: (totalCandidates) => {
              setSearchProgress({
                stage: 'evaluating',
                totalCandidates,
                currentEvaluating: 0,
                evaluated: 0
              })
              message.loading({
                content: `已生成 ${totalCandidates} 个候选方案，开始后端评估...`,
                key: 'search',
                duration: 2
              })
            },
            onProgress: (current, total) => {
              setSearchProgress(prev => ({
                ...prev,
                stage: 'evaluating',
                currentEvaluating: current,
                evaluated: current
              }))
              message.loading({
                content: `正在评估方案 ${current}/${total}（5 并发）...`,
                key: 'search',
                duration: 0
              })
            },
            onResultReady: (resultItem, _index, isFeasible) => {
              // 实时更新结果
              if (isFeasible) {
                const sr = resultItem as SearchResult
                realtimeFeasible.push(sr)
                // 按评分排序后更新
                realtimeFeasible.sort((a, b) => b.score - a.score)
                const plans = realtimeFeasible.slice(0, 10).map(r =>
                  adaptSimulationResult(r.simulation, modelConfig, inferenceConfig, r.parallelism, hardwareConfig)
                )
                setTopKPlans(plans)
                if (plans.length > 0) {
                  setAnalysisResult(plans[0])
                }
              } else {
                realtimeInfeasible.push(resultItem as InfeasibleResult)
                setInfeasiblePlans([...realtimeInfeasible])
              }
            }
          }
        )

        setSearchProgress(prev => ({ ...prev, stage: 'completed' }))
        message.success({ content: '搜索完成', key: 'search', duration: 2 })

        // 最终结果（可能已经通过实时回调更新）
        setInfeasiblePlans(searchResults.infeasible)

        if (searchResults.feasible.length > 0) {
          // 将 SearchResult 转换为 PlanAnalysisResult
          newTopKPlans = searchResults.feasible.map(sr =>
            adaptSimulationResult(sr.simulation, modelConfig, inferenceConfig, sr.parallelism, hardwareConfig)
          )
          result = newTopKPlans[0]
          setAnalysisResult(result)
          setTopKPlans(newTopKPlans)
          setSearchStats({
            evaluated: searchResults.feasible.length + searchResults.infeasible.length,
            feasible: searchResults.feasible.length,
            timeMs: Date.now() - startTime,
          })
        } else {
          setErrorMsg('未找到可行方案')
          setSearchStats({
            evaluated: searchResults.infeasible.length,
            feasible: 0,
            timeMs: Date.now() - startTime,
          })
        }
      }
      // 保存到历史记录并切换到详情视图
      if (result && result.is_feasible) {
        // 设置显示配置（分析时使用的配置，不随配置面板变化）
        setDisplayModelConfig(modelConfig)
        setDisplayInferenceConfig(inferenceConfig)
        // 使用局部变量newTopKPlans而不是状态变量topKPlans
        const currentTopKPlans = parallelismMode === 'auto' ? newTopKPlans : undefined
        const searchMode = parallelismMode === 'auto' ? 'auto' : 'manual'
        // 通过 props 回调添加历史记录
        onAddToHistory?.({
          modelName: modelConfig.model_name,
          parallelism: result.plan.parallelism,
          score: result.score.overall_score,
          ttft: result.latency.prefill_total_latency_ms,
          tpot: result.latency.decode_per_token_latency_ms,
          throughput: result.throughput.tokens_per_second,
          mfu: result.throughput.model_flops_utilization,
          mbu: result.throughput.memory_bandwidth_utilization,
          cost: result.cost?.cost_per_million_tokens ?? null,
          chips: result.plan.total_chips,
          result,
          topKPlans: searchMode === 'auto' ? currentTopKPlans?.slice(0, 5) : undefined,
          searchMode,
          modelConfig,
          inferenceConfig,
          hardwareConfig,
        })
        setViewMode('detail')  // 分析完成后切换到详情视图
        const plansCount = currentTopKPlans ? Math.min(5, currentTopKPlans.length) : 1
        message.success(`分析完成，已保存${plansCount}个方案到历史记录`)
      }
    } catch (error) {
      // 检查是否是取消导致的错误
      if (error instanceof DOMException && error.name === 'AbortError') {
        // 取消不需要额外处理，已在 handleCancelAnalysis 中处理
        return
      }
      console.error('分析失败:', error)
      const msg = error instanceof Error ? error.message : '未知错误'
      setErrorMsg(`搜索失败: ${msg}`)
    } finally {
      setLoading(false)
      abortControllerRef.current = null
    }
  }, [modelConfig, inferenceConfig, hardwareConfig, parallelismMode, manualStrategy, searchConstraints, maxChips, onAddToHistory, topology])

  // 等待后端芯片预设加载完成
  if (!hardwareConfig) {
    return (
      <div style={{ padding: 16, textAlign: 'center' }}>
        <Spin tip="正在加载芯片预设..." />
      </div>
    )
  }

  return (
    <div style={{ padding: 0 }}>
      {/* Benchmark 设置 (模型 + 推理参数) */}
      <div style={{ marginBottom: 12 }}>
        <BaseCard title="Benchmark 设置" accentColor="#5E6AD2" collapsible defaultExpanded>
          <BenchmarkConfigSelector
            modelConfig={modelConfig}
            onModelChange={setModelConfig}
            inferenceConfig={inferenceConfig}
            onInferenceChange={setInferenceConfig}
          />
        </BaseCard>
      </div>

      {/* 部署设置（合并：硬件配置 + 延迟设置 + 并行策略） */}
      <div style={{ marginBottom: 12 }}>
        <BaseCard title="部署设置" accentColor="#722ed1" collapsible defaultExpanded>
          {/* 硬件配置 */}
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#1A1A1A', marginBottom: 10 }}>
              硬件配置
            </div>
            {chipGroups.length === 0 ? (
              <div style={{ padding: 12, background: colors.warningLight, borderRadius: 8, border: '1px solid #ffd591' }}>
                <Text type="warning">
                  <WarningOutlined style={{ marginRight: 6 }} />
                  请先在「Board层级」中配置芯片类型
                </Text>
              </div>
            ) : (
              <>
                {chipGroups.length > 1 && (
                  <div style={{ ...configRowStyle, marginBottom: 8 }}>
                    <Text>分析芯片类型</Text>
                    <Select
                      size="small"
                      value={selectedChipType}
                      onChange={setSelectedChipType}
                      style={{ width: 140 }}
                      options={chipGroups.map(g => ({
                        value: g.presetId || g.chipType,
                        label: `${g.chipType} (${g.totalCount * podCount * racksPerPod}个)`,
                      }))}
                    />
                  </div>
                )}

                {/* 拓扑结构概览 */}
                <div style={{ padding: 10, background: colors.successLight, borderRadius: 8, fontSize: 12, border: '1px solid #b7eb8f', marginBottom: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                    <Text><CheckCircleOutlined style={{ color: colors.success, marginRight: 4 }} />拓扑配置</Text>
                    <Text>共 <b>{hardwareConfig.node.chips_per_node * hardwareConfig.cluster.num_nodes}</b> 个芯片</Text>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px', color: colors.textSecondary }}>
                    <span>Pod: {podCount} 个</span>
                    <span>Rack: {racksPerPod * podCount} 个</span>
                    <span>Board: {rackConfig ? rackConfig.boards.reduce((sum, b) => sum + b.count, 0) * racksPerPod * podCount : 0} 个</span>
                    <span>Chip: {hardwareConfig.node.chips_per_node * hardwareConfig.cluster.num_nodes} 个</span>
                  </div>
                </div>

                {/* 芯片信息 */}
                <div style={{ padding: 10, background: '#f0f5ff', borderRadius: 8, fontSize: 12, border: '1px solid #adc6ff' }}>
                  <div style={{ fontWeight: 600, marginBottom: 6, color: '#1A1A1A' }}>芯片: {hardwareConfig.chip.chip_type}</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px', color: colors.textSecondary }}>
                    <span>算力: {hardwareConfig.chip.compute_tflops_fp16} TFLOPs</span>
                    <span>显存: {hardwareConfig.chip.memory_gb} GB</span>
                    <span>显存带宽: {hardwareConfig.chip.memory_bandwidth_gbps.toFixed(1)} GB/s</span>
                    <span>核心数: {hardwareConfig.chip.num_cores}</span>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* 分隔线 */}
          <div style={{ height: 1, background: '#E5E5E5', marginBottom: 16 }} />

          {/* 延迟设置 */}
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#1A1A1A', marginBottom: 10 }}>
              延迟设置
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div style={configRowStyle}>
                <Tooltip title="Tensor Parallelism Round Trip Time: 张量并行通信的往返延迟，包括节点内 NVLink/PCIe 通信开销">
                  <Text style={{ fontSize: 12, cursor: 'help', borderBottom: '1px dashed #d9d9d9' }}>TP RTT (µs)</Text>
                </Tooltip>
                <InputNumber
                  size="small"
                  min={0}
                  max={10}
                  step={0.05}
                  value={protocolConfig.rtt_tp_us}
                  onChange={(v) => setProtocolConfig(prev => ({ ...prev, rtt_tp_us: v ?? 0.35 }))}
                  style={{ width: 70 }}
                />
              </div>
              <div style={configRowStyle}>
                <Tooltip title="Expert Parallelism Round Trip Time: 专家并行 (MoE) 通信的往返延迟，包括跨节点的专家路由开销">
                  <Text style={{ fontSize: 12, cursor: 'help', borderBottom: '1px dashed #d9d9d9' }}>EP RTT (µs)</Text>
                </Tooltip>
                <InputNumber
                  size="small"
                  min={0}
                  max={10}
                  step={0.05}
                  value={protocolConfig.rtt_ep_us}
                  onChange={(v) => setProtocolConfig(prev => ({ ...prev, rtt_ep_us: v ?? 0.85 }))}
                  style={{ width: 70 }}
                />
              </div>
              <div style={configRowStyle}>
                <Tooltip title="实际可用带宽与理论峰值带宽的比例，考虑了协议开销、拥塞等因素 (典型值: 0.85-0.95)">
                  <Text style={{ fontSize: 12, cursor: 'help', borderBottom: '1px dashed #d9d9d9' }}>带宽利用率</Text>
                </Tooltip>
                <InputNumber
                  size="small"
                  min={0.5}
                  max={1.0}
                  step={0.01}
                  value={protocolConfig.bandwidth_utilization}
                  onChange={(v) => setProtocolConfig(prev => ({ ...prev, bandwidth_utilization: v ?? 0.95 }))}
                  style={{ width: 70 }}
                />
              </div>
              <div style={configRowStyle}>
                <Tooltip title="多卡同步操作的固定开销，如 Barrier、AllReduce 的初始化延迟">
                  <Text style={{ fontSize: 12, cursor: 'help', borderBottom: '1px dashed #d9d9d9' }}>同步延迟 (µs)</Text>
                </Tooltip>
                <InputNumber
                  size="small"
                  min={0}
                  max={10}
                  step={0.1}
                  value={protocolConfig.sync_latency_us}
                  onChange={(v) => setProtocolConfig(prev => ({ ...prev, sync_latency_us: v ?? 0 }))}
                  style={{ width: 70 }}
                />
              </div>
              <div style={configRowStyle}>
                <Tooltip title="网络交换机的数据包转发延迟 (典型值: 0.1-0.5 µs)">
                  <Text style={{ fontSize: 12, cursor: 'help', borderBottom: '1px dashed #d9d9d9' }}>交换机延迟 (µs)</Text>
                </Tooltip>
                <InputNumber
                  size="small"
                  min={0}
                  max={10}
                  step={0.05}
                  value={networkConfig.switch_delay_us}
                  onChange={(v) => {
                    const switchDelay = v ?? 0.25
                    setNetworkConfig(prev => ({
                      ...prev,
                      switch_delay_us: switchDelay,
                      link_delay_us: 2 * switchDelay + 2 * prev.cable_delay_us,
                    }))
                  }}
                  style={{ width: 70 }}
                />
              </div>
              <div style={configRowStyle}>
                <Tooltip title="网络线缆的光/电信号传输延迟，约 5 ns/米 (典型值: 0.01-0.05 µs)">
                  <Text style={{ fontSize: 12, cursor: 'help', borderBottom: '1px dashed #d9d9d9' }}>线缆延迟 (µs)</Text>
                </Tooltip>
                <InputNumber
                  size="small"
                  min={0}
                  max={1}
                  step={0.005}
                  value={networkConfig.cable_delay_us}
                  onChange={(v) => {
                    const cableDelay = v ?? 0.025
                    setNetworkConfig(prev => ({
                      ...prev,
                      cable_delay_us: cableDelay,
                      link_delay_us: 2 * prev.switch_delay_us + 2 * cableDelay,
                    }))
                  }}
                  style={{ width: 70 }}
                />
              </div>
            </div>
            <div style={{ marginTop: 8, fontSize: 12, color: '#999' }}>
              <Tooltip title="完整的端到端链路延迟 = 2 × 交换机延迟 + 2 × 线缆延迟">
                <span style={{ cursor: 'help', borderBottom: '1px dashed #d9d9d9' }}>
                  链路延迟: {networkConfig.link_delay_us.toFixed(3)} µs
                </span>
              </Tooltip>
            </div>
          </div>

          {/* 分隔线 */}
          <div style={{ height: 1, background: '#E5E5E5', marginBottom: 16 }} />

          {/* 并行策略 */}
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#1A1A1A', marginBottom: 10 }}>
              并行策略
            </div>
            <ParallelismConfigPanel
              mode={parallelismMode}
              onModeChange={setParallelismMode}
              manualStrategy={manualStrategy}
              onManualStrategyChange={setManualStrategy}
              searchConstraints={searchConstraints}
              onSearchConstraintsChange={setSearchConstraints}
              maxChips={maxChips}
              modelConfig={modelConfig}
              hardwareConfig={hardwareConfig}
            />
          </div>
        </BaseCard>
      </div>

      {/* 运行按钮 */}
      <Button
        type="primary"
        icon={parallelismMode === 'auto' ? <SearchOutlined /> : <PlayCircleOutlined />}
        onClick={handleRunAnalysis}
        loading={loading}
        block
        size="large"
        style={{
          marginBottom: 16,
          height: 44,
          borderRadius: 8,
          background: colors.primary,
          boxShadow: '0 2px 8px rgba(94, 106, 210, 0.3)',
        }}
      >
        {parallelismMode === 'auto' ? '搜索最优方案' : '运行分析'}
      </Button>

      {/* 分析状态提示 */}
      {(loading || errorMsg) && (
        <div style={{
          marginTop: 12,
          padding: 12,
          background: loading ? '#e6f7ff' : '#fff2f0',
          borderRadius: 8,
          border: `1px solid ${loading ? '#91d5ff' : '#ffccc7'}`,
          textAlign: 'center',
          fontSize: 13,
        }}>
          {loading ? (
            <span><Spin size="small" style={{ marginRight: 8 }} />正在分析...</span>
          ) : (
            <span style={{ color: '#ff4d4f' }}><WarningOutlined style={{ marginRight: 6 }} />{errorMsg}</span>
          )}
        </div>
      )}

    </div>
  )
}

export { AnalysisResultDisplay }
export default DeploymentAnalysisPanel
