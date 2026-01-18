/**
 * LLM 部署分析面板
 *
 * 提供模型配置、推理配置、硬件配置、并行策略配置和分析结果展示
 */

import React, { useState, useCallback } from 'react'
import {
  Typography,
  Button,
  Select,
  Radio,
  Spin,
  message,
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
  ScoreWeights,
  DEFAULT_SCORE_WEIGHTS,
} from '../../../utils/llmDeployment/types'
import { HierarchicalTopology } from '../../../types'
import {
  MODEL_PRESETS,
  INFERENCE_PRESETS,
  createHardwareConfig,
} from '../../../utils/llmDeployment/presets'
import { analyzePlan } from '../../../utils/llmDeployment/planAnalyzer'
import { searchWithFixedChips } from '../../../utils/llmDeployment/planSearcher'
import { analyzeTopologyTraffic } from '../../../utils/llmDeployment/trafficMapper'
import {
  extractChipGroupsFromConfig,
  generateHardwareConfigFromPanelConfig,
  ChipGroupInfo,
} from '../../../utils/llmDeployment/topologyHardwareExtractor'
import { RackConfig, DeploymentAnalysisData, AnalysisHistoryItem, AnalysisViewMode } from '../shared'
import {
  ModelConfigSelector,
  InferenceConfigSelector,
  HardwareConfigSelector,
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

  // 硬件配置来源: 'topology' 使用拓扑配置, 'manual' 手动配置
  const [hardwareSource, setHardwareSource] = useState<'topology' | 'manual'>('topology')

  // 从拓扑配置提取的芯片组
  const [chipGroups, setChipGroups] = useState<ChipGroupInfo[]>([])
  const [selectedChipType, setSelectedChipType] = useState<string | undefined>()

  // 硬件配置状态
  const [hardwareConfig, setHardwareConfig] = useState<HardwareConfig>(() =>
    createHardwareConfig('h100-sxm', 'dgx-h100', 1, 400)
  )

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

    // 如果使用拓扑配置模式，立即更新硬件配置
    if (hardwareSource === 'topology' && groups.length > 0) {
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
  }, [rackConfigJson, hardwareSource, podCount, racksPerPod, topology?.connections])

  // 当选择的芯片类型变化时，更新硬件配置
  React.useEffect(() => {
    if (hardwareSource === 'topology' && rackConfig && chipGroups.length > 0 && selectedChipType) {
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
  }, [selectedChipType, hardwareSource, rackConfig, chipGroups, podCount, racksPerPod, topology?.connections])

  // 并行策略状态
  const [parallelismMode, setParallelismMode] = useState<'manual' | 'auto'>('manual')
  const [manualStrategy, setManualStrategy] = useState<ParallelismStrategy>({
    dp: 1, tp: 8, pp: 1, ep: 1, sp: 1,
  })
  const [searchConstraints, setSearchConstraints] = useState<SearchConstraints>({
    max_chips: 8,
    tp_within_node: true,
  })

  // 评分权重
  const [scoreWeights, setScoreWeights] = useState<ScoreWeights>({ ...DEFAULT_SCORE_WEIGHTS })

  // 分析结果状态
  const [analysisResult, setAnalysisResult] = useState<PlanAnalysisResult | null>(null)
  const [topKPlans, setTopKPlans] = useState<PlanAnalysisResult[]>([])
  const [searchStats, setSearchStats] = useState<{ evaluated: number; feasible: number; timeMs: number } | null>(null)
  const [loading, setLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

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

  // 当分析状态变化时，通知父组件
  React.useEffect(() => {
    if (onAnalysisDataChange) {
      onAnalysisDataChange({
        result: analysisResult,
        topKPlans,
        hardware: hardwareConfig,
        // 使用显示配置（分析时的配置），而不是配置面板当前选择
        model: displayModelConfig || modelConfig,
        inference: displayInferenceConfig || inferenceConfig,
        loading,
        errorMsg,
        searchStats,
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
  }, [analysisResult, topKPlans, hardwareConfig, displayModelConfig, displayInferenceConfig, modelConfig, inferenceConfig, loading, errorMsg, searchStats, onAnalysisDataChange, handleMapToTopology, topology, onTrafficResultChange, viewMode, history, handleLoadFromHistory, handleDeleteHistory, handleClearHistory])

  // 计算最大可用芯片数
  const maxChips = hardwareConfig.node.chips_per_node * hardwareConfig.cluster.num_nodes

  // 运行分析
  const handleRunAnalysis = useCallback(async () => {
    setAnalysisResult(null)
    setTopKPlans([])
    setSearchStats(null)
    setErrorMsg(null)
    setLoading(true)
    try {
      let result: PlanAnalysisResult | null = null
      // 使用局部变量捕获新的topKPlans，避免React状态更新延迟导致保存到历史时为空
      let newTopKPlans: PlanAnalysisResult[] = []
      if (parallelismMode === 'manual') {
        result = analyzePlan(modelConfig, inferenceConfig, manualStrategy, hardwareConfig)
        setAnalysisResult(result)
        newTopKPlans = [result]
        setTopKPlans(newTopKPlans)
      } else {
        const searchResult = searchWithFixedChips(
          modelConfig,
          inferenceConfig,
          hardwareConfig,
          searchConstraints.max_chips || maxChips,
          'balanced',
          scoreWeights
        )
        if (searchResult.top_k_plans.length > 0) {
          result = searchResult.optimal_plan
          setAnalysisResult(result)
          newTopKPlans = searchResult.top_k_plans
          setTopKPlans(newTopKPlans)
          setSearchStats({
            evaluated: searchResult.search_stats.evaluated_count,
            feasible: searchResult.search_stats.feasible_count,
            timeMs: searchResult.search_stats.search_time_ms,
          })
        } else {
          setErrorMsg(`未找到可行方案 (已评估 ${searchResult.search_stats.evaluated_count} 个方案)`)
          setSearchStats({
            evaluated: searchResult.search_stats.evaluated_count,
            feasible: 0,
            timeMs: searchResult.search_stats.search_time_ms,
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
      console.error('分析失败:', error)
      const msg = error instanceof Error ? error.message : '未知错误'
      setErrorMsg(`搜索失败: ${msg}`)
    } finally {
      setLoading(false)
    }
  }, [modelConfig, inferenceConfig, hardwareConfig, parallelismMode, manualStrategy, searchConstraints, maxChips, scoreWeights, onAddToHistory])

  return (
    <div style={{ padding: 0 }}>
      {/* 模型配置 */}
      <div style={{ marginBottom: 12 }}>
        <BaseCard title="模型配置" accentColor="#5E6AD2" collapsible defaultExpanded>
          <ModelConfigSelector value={modelConfig} onChange={setModelConfig} />
        </BaseCard>
      </div>

      {/* 推理配置 */}
      <div style={{ marginBottom: 12 }}>
        <BaseCard title="推理配置" accentColor="#13c2c2" collapsible defaultExpanded>
          <InferenceConfigSelector value={inferenceConfig} onChange={setInferenceConfig} />
        </BaseCard>
      </div>

      {/* 硬件配置 */}
      <div style={{ marginBottom: 12 }}>
        <BaseCard title="硬件配置" accentColor="#52c41a" collapsible defaultExpanded>
          <div style={{ marginBottom: 12 }}>
            <Radio.Group
              size="small"
              value={hardwareSource}
              onChange={(e) => setHardwareSource(e.target.value)}
              buttonStyle="solid"
            >
              <Radio.Button value="topology">使用拓扑配置</Radio.Button>
              <Radio.Button value="manual">手动配置</Radio.Button>
            </Radio.Group>
          </div>

          {hardwareSource === 'topology' ? (
            <div>
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
                    <div style={configRowStyle}>
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

                  <div style={{ padding: 10, background: colors.successLight, borderRadius: 8, fontSize: 12, border: '1px solid #b7eb8f' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                      <Text><CheckCircleOutlined style={{ color: colors.success, marginRight: 4 }} />芯片: <b>{hardwareConfig.chip.chip_type}</b></Text>
                      <Text>共 <b>{hardwareConfig.node.chips_per_node * hardwareConfig.cluster.num_nodes}</b> 个</Text>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px', color: colors.textSecondary }}>
                      <span>节点数: {hardwareConfig.cluster.num_nodes}</span>
                      <span>每节点: {hardwareConfig.node.chips_per_node} 个</span>
                      <span>算力: {hardwareConfig.chip.compute_tflops_fp16} TFLOPs</span>
                      <span>显存: {hardwareConfig.chip.memory_gb}GB</span>
                    </div>
                  </div>
                </>
              )}
            </div>
          ) : (
            <HardwareConfigSelector value={hardwareConfig} onChange={setHardwareConfig} />
          )}
        </BaseCard>
      </div>

      {/* 并行策略 */}
      <div style={{ marginBottom: 12 }}>
        <BaseCard title="并行策略" accentColor="#faad14" collapsible defaultExpanded>
          <ParallelismConfigPanel
            mode={parallelismMode}
            onModeChange={setParallelismMode}
            manualStrategy={manualStrategy}
            onManualStrategyChange={setManualStrategy}
            searchConstraints={searchConstraints}
            onSearchConstraintsChange={setSearchConstraints}
            maxChips={maxChips}
            scoreWeights={scoreWeights}
            onScoreWeightsChange={setScoreWeights}
          />
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
