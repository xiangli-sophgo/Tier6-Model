/**
 * LLM 部署分析面板
 *
 * 提供模型配置、推理配置、硬件配置、并行策略配置和分析结果展示
 */

import React, { useState, useCallback, useRef } from 'react'
import { useWorkbench } from '../../../contexts/WorkbenchContext'
import {
  PlayCircle,
  Search,
  Info,
} from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { NumberInput } from '@/components/ui/number-input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Textarea } from '@/components/ui/textarea'
import {
  LLMModelConfig,
  InferenceConfig,
  HardwareConfig,
  ParallelismStrategy,
  PlanAnalysisResult,
  TopologyTrafficResult,
  CommLatencyConfig,
  DEFAULT_COMM_LATENCY_CONFIG,
} from '../../../utils/llmDeployment/types'
import { HierarchicalTopology } from '../../../types'
import {
  getModelPreset,
  getBackendModelPresets,
} from '../../../utils/llmDeployment/presets'
import { InfeasibleResult } from '../../../utils/llmDeployment'
import { analyzeTopologyTraffic } from '../../../utils/llmDeployment/trafficMapper'
import {
  submitEvaluation,
  getTaskResults,
  cancelTask as cancelBackendTask,
} from '../../../api/tasks'
import {
  extractChipGroupsFromConfig,
  generateHardwareConfigFromPanelConfig,
  ChipGroupInfo,
  extractHardwareSummary,
} from '../../../utils/llmDeployment/topologyHardwareExtractor'
import {
  RackConfig,
  DeploymentAnalysisData,
  AnalysisHistoryItem,
  AnalysisViewMode,
  AnalysisTask,
  loadAnalysisTasks,
  saveAnalysisTasks,
} from '../shared'
import { AnalysisTaskList } from './AnalysisTaskList'
import { listConfigs, saveConfig, SavedConfig } from '../../../api/topology'
import {
  BenchmarkConfigSelector,
  colors,
  generateBenchmarkName,
} from './ConfigSelectors'
import { BaseCard } from '@/components/common/BaseCard'
import { ParallelismConfigPanel } from './ParallelismConfigPanel'
import { AnalysisResultDisplay } from './AnalysisResultDisplay'
import { useTaskWebSocket, TaskUpdate } from '../../../hooks/useTaskWebSocket'
import { TopologyInfoCard } from './TopologyInfoCard'


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
  onAddToHistory: _onAddToHistory,
  onDeleteHistory,
  onClearHistory,
}) => {
  // 获取 WorkbenchContext 用于页面跳转
  const { ui } = useWorkbench()

  // 模型配置状态（使用默认模型或第一个可用模型）
  const [modelConfig, setModelConfig] = useState<LLMModelConfig>(() => {
    const presets = getBackendModelPresets()
    const defaultModelId = 'deepseek-v3'
    if (presets[defaultModelId]) {
      return getModelPreset(defaultModelId)
    }
    // 如果默认模型不存在，使用第一个可用模型
    const firstModelId = Object.keys(presets)[0]
    return firstModelId ? getModelPreset(firstModelId) : {
      model_name: 'Default',
      model_type: 'dense',
      hidden_size: 4096,
      num_layers: 32,
      num_attention_heads: 32,
      num_kv_heads: 32,
      intermediate_size: 11008,
      vocab_size: 32000,
      weight_dtype: 'bf16',
      activation_dtype: 'bf16',
      max_seq_length: 4096,
      norm_type: 'rmsnorm',
    }
  })

  // 推理配置状态（使用默认值）
  const [inferenceConfig, setInferenceConfig] = useState<InferenceConfig>({
    batch_size: 8,
    input_seq_length: 512,
    output_seq_length: 256,
    max_seq_length: 768,
    num_micro_batches: 4,
  })

  // 从拓扑配置提取的芯片组
  const [chipGroups, setChipGroups] = useState<ChipGroupInfo[]>([])
  const [selectedChipType, setSelectedChipType] = useState<string | undefined>()

  // 拓扑配置文件列表
  const [topologyConfigs, setTopologyConfigs] = useState<SavedConfig[]>([])
  const [selectedTopologyConfig, setSelectedTopologyConfig] = useState<string | undefined>()

  // 当前选中的 Benchmark 配置文件名
  const [selectedBenchmark, setSelectedBenchmark] = useState<string | undefined>()
  // 当前使用的拓扑配置（从文件加载或从 props 传入）
  const [localRackConfig, setLocalRackConfig] = useState<RackConfig | undefined>(rackConfig)
  const [localPodCount, setLocalPodCount] = useState(podCount)
  const [localRacksPerPod, setLocalRacksPerPod] = useState(racksPerPod)

  // 保存弹窗状态
  const [saveAsModalOpen, setSaveAsModalOpen] = useState(false)
  const [newConfigName, setNewConfigName] = useState('')
  const [newConfigDesc, setNewConfigDesc] = useState('')
  const [saveLoading, setSaveLoading] = useState(false)

  // 硬件配置状态（从保存的拓扑配置中提取）
  const [hardwareConfig, setHardwareConfig] = useState<HardwareConfig | null>(null)

  // 实验名称（用户自定义，留空则使用 Benchmark 名称）
  const [experimentName, setExperimentName] = useState<string>('')

  // 任务并发数（本次评估使用的 worker 数量）
  const [taskMaxWorkers, setTaskMaxWorkers] = useState<number>(4)

  // Tile 搜索开关（默认开启）
  const [enableTileSearch, setEnableTileSearch] = useState<boolean>(true)

  // 分区搜索开关（默认开启）
  const [enablePartitionSearch, setEnablePartitionSearch] = useState<boolean>(true)

  // 最大模拟 token 数（默认 4）
  const [maxSimulatedTokens, setMaxSimulatedTokens] = useState<number>(4)

  // 加载拓扑配置列表
  React.useEffect(() => {
    const loadTopologyConfigs = async () => {
      try {
        const configs = await listConfigs()
        setTopologyConfigs(configs)
      } catch (error) {
        console.error('加载拓扑配置列表失败:', error)
      }
    }
    loadTopologyConfigs()
  }, [])

  // 当 props 传入的配置变化时，更新本地状态
  React.useEffect(() => {
    if (rackConfig && !selectedTopologyConfig) {
      setLocalRackConfig(rackConfig)
      setLocalPodCount(podCount)
      setLocalRacksPerPod(racksPerPod)
    }
  }, [rackConfig, podCount, racksPerPod, selectedTopologyConfig])

  // 选择拓扑配置文件
  const handleSelectTopologyConfig = useCallback((configName: string | undefined) => {
    setSelectedTopologyConfig(configName)
    if (!configName) {
      // 清除选择，使用 props 传入的配置
      setLocalRackConfig(rackConfig)
      setLocalPodCount(podCount)
      setLocalRacksPerPod(racksPerPod)
      // 重置延迟设置为默认值
      setCommLatencyConfig({ ...DEFAULT_COMM_LATENCY_CONFIG })
      return
    }
    const config = topologyConfigs.find(c => c.name === configName)
    if (config) {
      // 使用保存的配置
      if (config.rack_config) {
        setLocalRackConfig(config.rack_config as RackConfig)
      }
      setLocalPodCount(config.pod_count || 1)
      setLocalRacksPerPod(config.racks_per_pod || 1)
      // 恢复延迟设置
      if (config.comm_latency_config) {
        setCommLatencyConfig(config.comm_latency_config)
      }
      toast.success(`已加载拓扑配置: ${config.name}`)
    }
  }, [topologyConfigs, rackConfig, podCount, racksPerPod])

  // 从拓扑配置中提取硬件配置（不依赖后端）
  React.useEffect(() => {
    // 从拓扑配置提取硬件参数
    if (localRackConfig && localRackConfig.boards.length > 0 && topology?.connections) {
      const groups = extractChipGroupsFromConfig(localRackConfig.boards)
      if (groups.length > 0) {
        const firstChipType = groups[0].presetId || groups[0].chipType
        const config = generateHardwareConfigFromPanelConfig(
          localPodCount,
          localRacksPerPod,
          localRackConfig.boards.map(b => ({ chips: b.chips, count: b.count })),
          topology.connections,
          firstChipType
        )
        if (config) {
          setHardwareConfig(config)
        }
      }
    } else if (!hardwareConfig) {
      // 如果没有拓扑配置，使用默认值（SG2260E 参数）
      console.warn('未找到拓扑配置，使用默认硬件配置')
      const defaultConfig: HardwareConfig = {
        chip: {
          name: 'SG2262',
          num_cores: 64,
          compute_tflops_fp8: 1536,
          compute_tflops_bf16: 768,
          memory_capacity_gb: 64,
          memory_bandwidth_gbps: 11468,
          memory_bandwidth_utilization: 0.85,
          lmem_capacity_mb: 2,
          lmem_bandwidth_gbps: 512,
          // 微架构参数（SG2260E 默认值）
          cube_m: 16,
          cube_k: 32,
          cube_n: 8,
          sram_size_kb: 2048,
          sram_utilization: 0.45,
          lane_num: 16,
          align_bytes: 32,
          compute_dma_overlap_rate: 0.8,
        },
        board: {
          chips_per_board: 8,
          b2b_bandwidth_gbps: 450,
          b2b_latency_us: 0.35,
        },
        rack: {
          boards_per_rack: 4,
          r2r_bandwidth_gbps: 200,
          r2r_latency_us: 2,
        },
        pod: {
          racks_per_pod: 1,
          p2p_bandwidth_gbps: 100,
          p2p_latency_us: 5,
        },
      }
      setHardwareConfig(defaultConfig)
    }
  }, [localRackConfig, topology, localPodCount, localRacksPerPod])

  // 序列化 localRackConfig 用于深度比较
  const rackConfigJson = React.useMemo(() =>
    localRackConfig ? JSON.stringify(localRackConfig) : '',
    [localRackConfig]
  )

  // 当拓扑配置变化时，提取芯片组信息并更新硬件配置
  React.useEffect(() => {
    if (!localRackConfig || localRackConfig.boards.length === 0) {
      setChipGroups([])
      return
    }

    const groups = extractChipGroupsFromConfig(localRackConfig.boards)
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
        localPodCount,
        localRacksPerPod,
        localRackConfig.boards.map(b => ({ chips: b.chips, count: b.count })),
        connections,
        currentSelectedType
      )
      if (config) {
        setHardwareConfig(config)
      }
    }
  }, [rackConfigJson, localPodCount, localRacksPerPod, topology?.connections])

  // 当选择的芯片类型变化时，更新硬件配置
  React.useEffect(() => {
    if (localRackConfig && chipGroups.length > 0 && selectedChipType) {
      // 使用 generateHardwareConfigFromPanelConfig 从连接配置中提取带宽和延迟
      const connections = topology?.connections || []
      const config = generateHardwareConfigFromPanelConfig(
        localPodCount,
        localRacksPerPod,
        localRackConfig.boards.map(b => ({ chips: b.chips, count: b.count })),
        connections,
        selectedChipType
      )
      if (config) {
        setHardwareConfig(config)
      }
    }
  }, [selectedChipType, localRackConfig, chipGroups, localPodCount, localRacksPerPod, topology?.connections])

  // 并行策略状态
  const [parallelismMode, setParallelismMode] = useState<'manual' | 'auto'>('auto')
  const [manualStrategy, setManualStrategy] = useState<ParallelismStrategy>({
    dp: 1, tp: 1, pp: 1, ep: 1, sp: 1, moe_tp: 1,
  })

  // 当模型配置或硬件配置变化时，更新手动策略为满足约束的默认值
  React.useEffect(() => {
    if (!hardwareConfig) return

    const isMoE = modelConfig.model_type === 'moe' && modelConfig.moe_config
    const maxTP = Math.min(128, modelConfig.num_attention_heads, hardwareConfig.board.chips_per_board)

    // 找一个能整除头数的 TP 值
    let validTP = 1
    for (let tp = maxTP; tp >= 1; tp--) {
      if (modelConfig.num_attention_heads % tp === 0) {
        validTP = tp
        break
      }
    }

    if (isMoE) {
      // MoE 模型：确保 DP × TP = MoE_TP × EP
      // 默认使用 dp=1, tp=validTP, moe_tp=validTP, ep=1
      setManualStrategy(prev => ({
        ...prev,
        dp: 1,
        tp: validTP,
        moe_tp: validTP,
        ep: 1,
      }))
    } else {
      // 非 MoE 模型
      setManualStrategy(prev => ({
        ...prev,
        dp: 1,
        tp: validTP,
      }))
    }
  }, [modelConfig, hardwareConfig])

  // 通信延迟配置 (统一配置：协议、网络基础设施、芯片延迟)
  const [commLatencyConfig, setCommLatencyConfig] = useState<CommLatencyConfig>({ ...DEFAULT_COMM_LATENCY_CONFIG })

  // 计算拓扑层级统计
  const topologyStats = React.useMemo(() => {
    const boardCount = localRackConfig
      ? localRackConfig.boards.reduce((sum, b) => sum + b.count, 0) * localRacksPerPod * localPodCount
      : 0
    const chipCount = localRackConfig
      ? localRackConfig.boards.reduce((sum, b) => sum + b.count * b.chips.reduce((cs, c) => cs + c.count, 0), 0) * localRacksPerPod * localPodCount
      : 0
    return {
      podCount: localPodCount,
      rackCount: localRacksPerPod * localPodCount,
      boardCount,
      chipCount,
    }
  }, [localPodCount, localRacksPerPod, localRackConfig])

  // 获取互联参数（从选中的配置或使用默认值）
  const interconnectParams = React.useMemo(() => {
    const selectedConfig = selectedTopologyConfig
      ? topologyConfigs.find(c => c.name === selectedTopologyConfig)
      : null
    if (selectedConfig?.hardware_params?.interconnect) {
      return selectedConfig.hardware_params.interconnect
    }
    // 默认值
    return {
      c2c: { bandwidth_gbps: 448, latency_us: 0.25 },
      b2b: { bandwidth_gbps: 450, latency_us: 0.35 },
      r2r: { bandwidth_gbps: 200, latency_us: 2 },
      p2p: { bandwidth_gbps: 100, latency_us: 5 },
    }
  }, [selectedTopologyConfig, topologyConfigs])

  // 跳转到互联拓扑页面
  const handleNavigateToTopology = useCallback(() => {
    ui.setViewMode('topology')
  }, [ui])

  // 刷新配置列表
  const refreshTopologyConfigs = useCallback(async () => {
    try {
      const configs = await listConfigs()
      setTopologyConfigs(configs)
    } catch (error) {
      console.error('刷新拓扑配置列表失败:', error)
    }
  }, [])

  // 另存为新配置
  const handleSaveAsConfig = useCallback(async () => {
    if (!newConfigName.trim()) {
      toast.warning('请输入配置名称')
      return
    }
    // 检查名称是否已存在
    if (topologyConfigs.some(c => c.name === newConfigName.trim())) {
      toast.error('配置名称已存在，请使用其他名称')
      return
    }
    setSaveLoading(true)
    try {
      const newConfig: SavedConfig = {
        name: newConfigName.trim(),
        description: newConfigDesc.trim() || undefined,
        pod_count: localPodCount,
        racks_per_pod: localRacksPerPod,
        rack_config: localRackConfig ? {
          total_u: localRackConfig.total_u,
          boards: localRackConfig.boards,
        } : undefined,
        comm_latency_config: { ...commLatencyConfig },
      }
      await saveConfig(newConfig)
      await refreshTopologyConfigs()
      setSelectedTopologyConfig(newConfigName.trim())
      setSaveAsModalOpen(false)
      setNewConfigName('')
      setNewConfigDesc('')
      toast.success(`已创建新配置: ${newConfigName.trim()}`)
    } catch (error) {
      console.error('另存为配置失败:', error)
      toast.error('另存为配置失败')
    } finally {
      setSaveLoading(false)
    }
  }, [newConfigName, newConfigDesc, topologyConfigs, selectedTopologyConfig, localPodCount, localRacksPerPod, localRackConfig, commLatencyConfig, refreshTopologyConfigs])

  // 分析结果状态
  const [analysisResult, setAnalysisResult] = useState<PlanAnalysisResult | null>(null)
  const [topKPlans, setTopKPlans] = useState<PlanAnalysisResult[]>([])
  const [infeasiblePlans, _setInfeasiblePlans] = useState<InfeasibleResult[]>([])
  const [searchStats, _setSearchStats] = useState<{ evaluated: number; feasible: number; timeMs: number } | null>(null)
  const [loading, _setLoading] = useState(false)
  const [errorMsg, _setErrorMsg] = useState<string | null>(null)

  // 标记未使用的 setter（保留状态用于兼容性）
  void _setInfeasiblePlans
  void _setSearchStats
  void _setLoading
  void _setErrorMsg

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

  // 取消控制器 Map（支持多个并行任务）
  const abortControllersMap = useRef<Map<string, AbortController>>(new Map())

  // 分析任务列表（本地持久化）
  const [analysisTasks, setAnalysisTasks] = useState<AnalysisTask[]>(() => loadAnalysisTasks())

  // 使用 ref 保存最新的任务列表，以便在 WebSocket 回调中访问（避免闭包问题）
  const analysisTasksRef = useRef<AnalysisTask[]>(analysisTasks)

  // 任务变化时保存到 localStorage 并更新 ref
  React.useEffect(() => {
    saveAnalysisTasks(analysisTasks)
    analysisTasksRef.current = analysisTasks
  }, [analysisTasks])

  // 更新任务状态
  const updateTask = useCallback((taskId: string, updates: Partial<AnalysisTask>) => {
    setAnalysisTasks(prev => prev.map(t => t.id === taskId ? { ...t, ...updates } : t))
  }, [])

  // 添加新任务
  const addTask = useCallback((task: AnalysisTask) => {
    setAnalysisTasks(prev => [task, ...prev])
  }, [])

  // 删除任务
  const deleteTask = useCallback((taskId: string) => {
    // 如果任务正在运行，先取消
    const controller = abortControllersMap.current.get(taskId)
    if (controller) {
      controller.abort()
      abortControllersMap.current.delete(taskId)
    }
    setAnalysisTasks(prev => prev.filter(t => t.id !== taskId))
  }, [])

  // 取消任务
  const cancelTask = useCallback(async (taskId: string) => {
    // 取消后端任务
    try {
      await cancelBackendTask(taskId)
    } catch (error) {
      console.error('取消后端任务失败:', error)
    }

    updateTask(taskId, { status: 'cancelled', endTime: Date.now() })
  }, [updateTask])

  // 清空已完成任务
  const clearCompletedTasks = useCallback(() => {
    setAnalysisTasks(prev => prev.filter(t => t.status === 'running'))
  }, [])

  // 刷新任务列表（从 localStorage 重新加载）
  const refreshTasks = useCallback(() => {
    setAnalysisTasks(loadAnalysisTasks())
  }, [])

  // WebSocket 任务更新处理（实时接收后端进度）
  const handleTaskUpdate = useCallback(async (update: TaskUpdate) => {
    // 直接检查任务是否存在于本地任务列表中（使用 ref 访问最新值）
    const taskExists = analysisTasksRef.current.some(t => t.id === update.task_id)

    if (!taskExists) {
      // 可能是其他页面的任务或已删除的任务，忽略
      return
    }

    const localTaskId = update.task_id

    if (update.status === 'running') {
      // 更新进度和子任务
      const updateData: Partial<AnalysisTask> = {
        progress: { current: Math.round(update.progress), total: 100 },
      }

      // 如果有子任务数据，转换并更新
      if (update.search_stats?.sub_tasks) {
        updateData.subTasks = update.search_stats.sub_tasks.map(st => ({
          candidateIndex: st.candidate_index,
          parallelism: st.parallelism,
          status: st.status,
          progress: st.progress,
          chips: st.chips,
        }))
      }

      updateTask(localTaskId, updateData)
    } else if (update.status === 'completed') {
      // 任务完成，获取结果
      try {
        const results = await getTaskResults(update.task_id)

        if (results.top_k_plans && results.top_k_plans.length > 0) {
          const topPlan = results.top_k_plans[0] as Record<string, unknown>
          const parallelism = (topPlan.parallelism || {}) as ParallelismStrategy

          // 更新任务状态
          updateTask(localTaskId, {
            status: 'completed',
            endTime: Date.now(),
            parallelism: parallelism,
            score: topPlan.score as number,
            ttft: topPlan.ttft as number,
            tpot: topPlan.tpot as number,
            tps: topPlan.tps as number,
            mfu: topPlan.mfu as number,
            mbu: topPlan.mbu as number,
          })
        } else {
          // 没有可行方案，提取失败原因
          let errorMessage = '未找到可行方案'
          if (results.infeasible_plans && results.infeasible_plans.length > 0) {
            const firstInfeasible = results.infeasible_plans[0] as Record<string, unknown>
            const reason = firstInfeasible.infeasible_reason as string
            if (reason) {
              errorMessage = `${errorMessage}: ${reason}`
            }
          }
          updateTask(localTaskId, {
            status: 'failed',
            endTime: Date.now(),
            error: errorMessage,
          })
        }
      } catch (error) {
        console.error('[DeploymentAnalysis] 获取任务结果失败:', error)
        updateTask(localTaskId, {
          status: 'failed',
          endTime: Date.now(),
          error: '获取结果失败',
        })
      }
    } else if (update.status === 'failed') {
      updateTask(localTaskId, {
        status: 'failed',
        endTime: Date.now(),
        error: update.error || '任务执行失败',
      })
    } else if (update.status === 'cancelled') {
      updateTask(localTaskId, {
        status: 'cancelled',
        endTime: Date.now(),
      })
    }
  }, [updateTask])

  // 使用 WebSocket 订阅任务更新
  useTaskWebSocket({
    onTaskUpdate: handleTaskUpdate,
  })

  // 查看任务结果（跳转到结果管理页面）
  const viewTaskResult = useCallback((task: AnalysisTask) => {
    // 跳转到结果管理页面
    ui.setViewMode('results')
    toast.info(`已跳转到结果管理，请查找实验: ${task.experimentName || task.benchmarkName || task.modelName}`)
  }, [ui])

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
    toast.success(`已加载历史记录${plansCount > 1 ? `（含 ${plansCount} 个候选方案）` : ''}`)
  }, [])

  // 删除历史记录 (使用 props 回调)
  const handleDeleteHistory = useCallback((id: string) => {
    onDeleteHistory?.(id)
    toast.success('已删除')
  }, [onDeleteHistory])

  // 清空历史记录 (使用 props 回调)
  const handleClearHistory = useCallback(() => {
    onClearHistory?.()
    toast.success('已清空历史记录')
  }, [onClearHistory])

  // 取消评估（兼容旧接口）
  const handleCancelAnalysis = useCallback(() => {
    // 取消所有正在运行的任务
    abortControllersMap.current.forEach((controller, taskId) => {
      controller.abort()
      updateTask(taskId, { status: 'cancelled', endTime: Date.now() })
    })
    abortControllersMap.current.clear()
    setSearchProgress(prev => ({ ...prev, stage: 'cancelled' }))
  }, [updateTask])

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

  // 计算最大可用芯片数（从拓扑配置中提取实际芯片总数）
  const maxChips = React.useMemo(() => {
    if (!topology) return 0
    const summary = extractHardwareSummary(topology)
    return summary.totalChips
  }, [topology])

  // 运行分析（提交到后端执行）
  const handleRunAnalysis = useCallback(async () => {
    // 验证：必须选择配置文件
    if (!selectedTopologyConfig) {
      toast.error('请先选择拓扑配置文件')
      return
    }
    if (!selectedBenchmark) {
      toast.error('请先选择 Benchmark 配置')
      return
    }

    const strategy = parallelismMode === 'manual' ? manualStrategy : { dp: 1, tp: 1, pp: 1, ep: 1, sp: 1, moe_tp: 1 }

    // 使用用户输入的实验名称，如果为空则使用 Benchmark 名称
    const finalExperimentName = experimentName.trim() || selectedBenchmark

    // 生成临时任务 ID（提交后会更新为后端返回的 ID）
    const tempTaskId = `temp-${Date.now()}`

    // 计算实际使用的芯片数
    const isMoE = !!modelConfig.moe_config
    const actualChips = parallelismMode === 'manual'
      ? (isMoE ? strategy.dp * strategy.tp : strategy.dp * strategy.tp * strategy.ep)
      : maxChips

    // 先创建本地任务记录并显示卡片（立即响应用户操作）
    const newTask: AnalysisTask = {
      id: tempTaskId,
      status: 'running',
      startTime: Date.now(),
      experimentName: finalExperimentName,
      modelName: modelConfig.model_name,
      benchmarkName: selectedBenchmark,
      parallelism: strategy,
      mode: parallelismMode,
      chips: actualChips,
    }
    addTask(newTask)

    try {
      // 提交任务：只传配置文件名，后端自动加载完整配置
      // 注意：selectedBenchmark 和 selectedTopologyConfig 在函数开头已验证非空
      const response = await submitEvaluation({
        experiment_name: finalExperimentName,
        description: selectedBenchmark!,
        benchmark_name: selectedBenchmark!,
        topology_config_name: selectedTopologyConfig!,
        search_mode: parallelismMode,
        manual_parallelism: parallelismMode === 'manual' ? manualStrategy as unknown as Record<string, unknown> : undefined,
        search_constraints: parallelismMode === 'auto' ? { max_chips: maxChips } : undefined,
        max_workers: taskMaxWorkers,
        enable_tile_search: enableTileSearch,
        enable_partition_search: enablePartitionSearch,
        max_simulated_tokens: maxSimulatedTokens,
      })

      const backendTaskId = response.task_id

      // 更新任务 ID 为后端返回的真实 ID
      setAnalysisTasks(prev => prev.map(t =>
        t.id === tempTaskId ? { ...t, id: backendTaskId } : t
      ))

      toast.success('任务已提交')
    } catch (error) {
      console.error('提交任务失败:', error)
      const msg = error instanceof Error ? error.message : '未知错误'
      // 提交失败，更新任务状态为失败
      updateTask(tempTaskId, {
        status: 'failed',
        endTime: Date.now(),
        error: msg,
      })
      toast.error(`提交任务失败: ${msg}`)
    }
  }, [
    experimentName, taskMaxWorkers, modelConfig, parallelismMode, manualStrategy,
    maxChips, addTask, updateTask, setAnalysisTasks, enableTileSearch,
    enablePartitionSearch, maxSimulatedTokens, selectedTopologyConfig, selectedBenchmark
  ])


  // 如果硬件配置未加载，显示提示（不再是加载中）
  if (!hardwareConfig) {
    return (
      <div className="p-10 text-center">
        <span className="block mb-4 text-sm text-gray-500">
          未找到拓扑配置
        </span>
        <span className="text-[13px] text-gray-500">
          请先在「拓扑设置」页面配置芯片和网络拓扑
        </span>
      </div>
    )
  }

  return (
    <>
      <div>
        {/* 上方：Benchmark 设置和部署设置（左右两列） */}
        <div className="grid grid-cols-2 gap-8 mb-4">
          {/* 左列：Benchmark 设置 + 并行策略 */}
          <div>
            <BaseCard title="Benchmark 设置" collapsible defaultExpanded gradient>
              <BenchmarkConfigSelector
                modelConfig={modelConfig}
                onModelChange={setModelConfig}
                inferenceConfig={inferenceConfig}
                onInferenceChange={setInferenceConfig}
                onBenchmarkSelect={setSelectedBenchmark}
              />
            </BaseCard>

            {/* 部署策略卡片 */}
            <BaseCard collapsible title="部署策略" defaultExpanded gradient className="mt-4">
              <ParallelismConfigPanel
                mode={parallelismMode}
                onModeChange={setParallelismMode}
                manualStrategy={manualStrategy}
                onManualStrategyChange={setManualStrategy}
                maxChips={maxChips}
                modelConfig={modelConfig}
                hardwareConfig={hardwareConfig}
              />

              {/* Tile 搜索开关 */}
              <div className="mt-4 mb-2 flex items-center justify-between">
                <div className="flex items-center">
                  <span className="text-[13px] text-gray-600 mr-2">启用 Tile 搜索</span>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-3.5 w-3.5 text-gray-400 cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>开启时使用最优tile搜索以获得最高精度，关闭时使用固定tile大小以显著提升评估速度</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Switch
                  checked={enableTileSearch}
                  onCheckedChange={setEnableTileSearch}
                />
              </div>

              {/* 分区搜索开关 */}
              <div className="mt-3 mb-2 flex items-center justify-between">
                <div className="flex items-center">
                  <span className="text-[13px] text-gray-600 mr-2">启用分区搜索</span>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-3.5 w-3.5 text-gray-400 cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>开启时搜索最优分区策略（极慢，单个GEMM需100+秒），关闭时使用固定分区（推荐，速度提升100倍）</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Switch
                  checked={enablePartitionSearch}
                  onCheckedChange={setEnablePartitionSearch}
                />
              </div>

              {/* 最大模拟 token 数 */}
              <div className="mt-3 mb-2 flex items-center justify-between">
                <div className="flex items-center">
                  <span className="text-[13px] text-gray-600 mr-2">最大模拟 Token 数</span>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-3.5 w-3.5 text-gray-400 cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>Decode 阶段模拟的 token 数量，值越小评估越快但精度略降。推荐：快速评估用 1-2，精确评估用 4-8</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <NumberInput
                  min={1}
                  max={16}
                  value={maxSimulatedTokens}
                  onChange={(value) => setMaxSimulatedTokens(value || 4)}
                  className="w-20"
                />
              </div>

              {/* 实验名称和任务并发数 */}
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                  <div className="mb-1.5 text-[13px] text-gray-600">实验名称</div>
                  <Input
                    placeholder="留空则使用 Benchmark 名称"
                    value={experimentName}
                    onChange={(e) => setExperimentName(e.target.value)}
                  />
                </div>
                <div>
                  <div className="mb-1.5 text-[13px] text-gray-600 flex items-center">
                    任务并发数
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 ml-1 text-gray-400 cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>本次评估使用的 worker 数量（1-16）</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <NumberInput
                    min={1}
                    max={16}
                    value={taskMaxWorkers}
                    onChange={(value) => setTaskMaxWorkers(value || 4)}
                    className="w-full"
                  />
                </div>
              </div>

              {/* 运行按钮 */}
              <Button
                onClick={handleRunAnalysis}
                className="w-full mt-4 h-11 rounded-lg"
                style={{
                  background: colors.primary,
                  boxShadow: '0 2px 8px rgba(94, 106, 210, 0.3)',
                }}
              >
                {parallelismMode === 'auto' ? (
                  <>
                    <Search className="h-4 w-4 mr-2" />
                    开始方案评估
                  </>
                ) : (
                  <>
                    <PlayCircle className="h-4 w-4 mr-2" />
                    运行分析
                  </>
                )}
              </Button>
            </BaseCard>
          </div>

          {/* 右列：拓扑配置（只读展示） */}
          <div>
            <TopologyInfoCard
              topologyConfigs={topologyConfigs}
              selectedConfigName={selectedTopologyConfig}
              onSelectConfig={handleSelectTopologyConfig}
              onNavigateToTopology={handleNavigateToTopology}
              chipGroups={chipGroups}
              selectedChipType={selectedChipType}
              onSelectChipType={setSelectedChipType}
              hardwareConfig={hardwareConfig}
              topologyStats={topologyStats}
              interconnectParams={interconnectParams}
              commLatencyConfig={commLatencyConfig}
            />
          </div>
        </div>

        {/* 另存为弹窗 */}
        <Dialog open={saveAsModalOpen} onOpenChange={setSaveAsModalOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>另存为新配置</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <Label className="block mb-2">配置名称 <span className="text-red-500">*</span></Label>
                <Input
                  value={newConfigName}
                  onChange={(e) => setNewConfigName(e.target.value)}
                  placeholder="请输入配置名称"
                />
              </div>
              <div>
                <Label className="block mb-2">描述 (可选)</Label>
                <Textarea
                  value={newConfigDesc}
                  onChange={(e) => setNewConfigDesc(e.target.value)}
                  placeholder="请输入配置描述"
                  rows={3}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => {
                setSaveAsModalOpen(false)
                setNewConfigName('')
                setNewConfigDesc('')
              }}>
                取消
              </Button>
              <Button onClick={handleSaveAsConfig} disabled={saveLoading}>
                {saveLoading ? '保存中...' : '保存'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* 分析任务列表 */}
        <BaseCard collapsible title="分析任务" defaultExpanded gradient className="mt-4">
          <AnalysisTaskList
            tasks={analysisTasks}
            onViewTask={viewTaskResult}
            onCancelTask={cancelTask}
            onDeleteTask={deleteTask}
            onClearCompleted={clearCompletedTasks}
            onRefresh={refreshTasks}
          />
        </BaseCard>
      </div>
    </>
  )
}

export { AnalysisResultDisplay }
export default DeploymentAnalysisPanel
