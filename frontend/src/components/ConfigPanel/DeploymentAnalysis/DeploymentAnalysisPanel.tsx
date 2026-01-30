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
  AlertTriangle,
  CheckCircle,
  Save,
  Copy,
  RotateCcw,
  Info,
} from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { NumberInput } from '@/components/ui/number-input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
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
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
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
  MODEL_PRESETS,
  INFERENCE_PRESETS,
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
  configRowStyle,
  generateBenchmarkName,
} from './ConfigSelectors'
import { BaseCard } from '../../common/BaseCard'
import { ParallelismConfigPanel } from './ParallelismConfigPanel'
import { AnalysisResultDisplay } from './AnalysisResultDisplay'
import { useTaskWebSocket, TaskUpdate } from '../../../hooks/useTaskWebSocket'


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
  // 获取 WorkbenchContext 用于页面跳转
  const { ui } = useWorkbench()

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

  // 拓扑配置文件列表
  const [topologyConfigs, setTopologyConfigs] = useState<SavedConfig[]>([])
  const [selectedTopologyConfig, setSelectedTopologyConfig] = useState<string | undefined>()
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
          chip_type: 'SG2260E',
          num_cores: 64,
          compute_tflops_fp8: 1536,
          compute_tflops_bf16: 768,
          memory_capacity_gb: 64,
          memory_bandwidth_gbps: 11468,
          memory_bandwidth_utilization: 0.85,
          lmem_capacity_mb: 2,
          lmem_bandwidth_gbps: 512,
          c2c_bandwidth_gbps: 448,
          c2c_latency_us: 0.2,
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

  // 刷新配置列表
  const refreshTopologyConfigs = useCallback(async () => {
    try {
      const configs = await listConfigs()
      setTopologyConfigs(configs)
    } catch (error) {
      console.error('刷新拓扑配置列表失败:', error)
    }
  }, [])

  // 保存当前配置 (更新已选择的配置)
  const handleSaveConfig = useCallback(async () => {
    if (!selectedTopologyConfig) {
      toast.warning('请先选择一个配置文件，或使用「另存为」创建新配置')
      return
    }
    const existingConfig = topologyConfigs.find(c => c.name === selectedTopologyConfig)
    if (!existingConfig) {
      toast.error('配置文件不存在')
      return
    }
    setSaveLoading(true)
    try {
      const updatedConfig: SavedConfig = {
        ...existingConfig,
        comm_latency_config: { ...commLatencyConfig },
      }
      await saveConfig(updatedConfig)
      await refreshTopologyConfigs()
      toast.success(`已保存配置: ${selectedTopologyConfig}`)
    } catch (error) {
      console.error('保存配置失败:', error)
      toast.error('保存配置失败')
    } finally {
      setSaveLoading(false)
    }
  }, [selectedTopologyConfig, topologyConfigs, commLatencyConfig, refreshTopologyConfigs])

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
      // 获取当前拓扑配置的基础数据
      const baseConfig = selectedTopologyConfig
        ? topologyConfigs.find(c => c.name === selectedTopologyConfig)
        : null
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

  // 重置延迟设置为默认值
  const handleResetDelayConfig = useCallback(() => {
    setCommLatencyConfig({ ...DEFAULT_COMM_LATENCY_CONFIG })
    toast.success('已重置延迟设置为默认值')
  }, [])

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
    if (!hardwareConfig) return
    if (!topology) {
      toast.error('拓扑配置未加载，请先配置拓扑')
      return
    }

    const strategy = parallelismMode === 'manual' ? manualStrategy : { dp: 1, tp: 1, pp: 1, ep: 1, sp: 1, moe_tp: 1 }

    // 生成 Benchmark 配置名称
    const benchmarkName = generateBenchmarkName(modelConfig, inferenceConfig)
    // 使用用户输入的实验名称，如果为空则使用 Benchmark 名称
    const finalExperimentName = experimentName.trim() || benchmarkName

    // 生成临时任务 ID（提交后会更新为后端返回的 ID）
    const tempTaskId = `temp-${Date.now()}`

    // 计算实际使用的芯片数
    // MoE 模型: DP × TP（因为 MoE 约束 DP × TP = MoE_TP × EP，Attention 和 MoE 共用芯片）
    // 非 MoE 模型: DP × TP × EP
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
      benchmarkName,
      parallelism: strategy,
      mode: parallelismMode,
      chips: actualChips,
    }
    addTask(newTask)

    // 捕获当前配置
    const currentModelConfig = { ...modelConfig }
    const currentInferenceConfig = { ...inferenceConfig }
    const currentHardwareConfig = { ...hardwareConfig }
    const currentParallelismMode = parallelismMode
    const currentManualStrategy = { ...manualStrategy }

    try {
      // 构建完整的拓扑配置（物理拓扑 + 通信延迟配置 + 硬件配置）
      // 注意：后端 _extract_hardware_config 会检查 topology.hardware_config
      const fullTopology = {
        ...topology,
        comm_latency_config: { ...commLatencyConfig },
        hardware_config: currentHardwareConfig,  // 嵌入硬件配置，确保 chip_type 正确传递
      }

      // 生成拓扑配置名称（如果未选择配置文件，则自动生成）
      let finalTopologyConfigName = selectedTopologyConfig
      if (!finalTopologyConfigName) {
        // 计算拓扑层级信息
        const totalPods = localPodCount
        const totalRacks = localRacksPerPod * localPodCount
        const totalBoards = localRackConfig
          ? localRackConfig.boards.reduce((sum, b) => sum + b.count, 0) * localRacksPerPod * localPodCount
          : 0
        const totalChips = maxChips

        // 格式: P{Pod数}-R{Rack总数}-B{Board总数}-C{Chip总数}
        finalTopologyConfigName = `P${totalPods}-R${totalRacks}-B${totalBoards}-C${totalChips}`
      }

      // 提交任务到后端
      const response = await submitEvaluation({
        experiment_name: finalExperimentName,
        description: benchmarkName,
        topology: fullTopology as unknown as Record<string, unknown>,
        model: currentModelConfig as unknown as Record<string, unknown>,
        hardware: currentHardwareConfig as unknown as Record<string, unknown>,
        inference: currentInferenceConfig as unknown as Record<string, unknown>,
        search_mode: currentParallelismMode,
        manual_parallelism: currentParallelismMode === 'manual' ? currentManualStrategy as unknown as Record<string, unknown> : undefined,
        search_constraints: currentParallelismMode === 'auto' ? { max_chips: maxChips } : undefined,
        max_workers: taskMaxWorkers,
        enable_tile_search: enableTileSearch,
        enable_partition_search: enablePartitionSearch,
        max_simulated_tokens: maxSimulatedTokens,
        benchmark_name: benchmarkName,
        topology_config_name: finalTopologyConfigName,
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
  }, [experimentName, taskMaxWorkers, modelConfig, inferenceConfig, hardwareConfig, parallelismMode, manualStrategy, maxChips, topology, addTask, updateTask, commLatencyConfig, setAnalysisTasks])

  // Collapsible panel state for chip parameters
  const [chipParamsOpen, setChipParamsOpen] = useState(false)
  const [boardParamsOpen, setBoardParamsOpen] = useState(false)
  const [rackParamsOpen, setRackParamsOpen] = useState(false)
  const [podParamsOpen, setPodParamsOpen] = useState(false)
  const [commParamsOpen, setCommParamsOpen] = useState(false)

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
    <TooltipProvider>
      <div>
        {/* 上方：Benchmark 设置和部署设置（左右两列） */}
        <div className="grid grid-cols-2 gap-8 mb-4">
          {/* 左列：Benchmark 设置 + 并行策略 */}
          <div>
            <BaseCard title="Benchmark 设置" accentColor="#5E6AD2" collapsible defaultExpanded>
              <BenchmarkConfigSelector
                modelConfig={modelConfig}
                onModelChange={setModelConfig}
                inferenceConfig={inferenceConfig}
                onInferenceChange={setInferenceConfig}
              />
            </BaseCard>

            {/* 部署策略卡片 */}
            <BaseCard title="部署策略" accentColor="#13c2c2" collapsible defaultExpanded style={{ marginTop: 16 }}>
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
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="h-3.5 w-3.5 text-gray-400 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>开启时使用最优tile搜索以获得最高精度，关闭时使用固定tile大小以显著提升评估速度</TooltipContent>
                  </Tooltip>
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
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="h-3.5 w-3.5 text-gray-400 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>开启时搜索最优分区策略（极慢，单个GEMM需100+秒），关闭时使用固定分区（推荐，速度提升100倍）</TooltipContent>
                  </Tooltip>
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
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="h-3.5 w-3.5 text-gray-400 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>Decode 阶段模拟的 token 数量，值越小评估越快但精度略降。推荐：快速评估用 1-2，精确评估用 4-8</TooltipContent>
                  </Tooltip>
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
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-3.5 w-3.5 ml-1 text-gray-400 cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>本次评估使用的 worker 数量（1-16）</TooltipContent>
                    </Tooltip>
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

          {/* 右列：拓扑设置 */}
          <div>
            <BaseCard title="拓扑设置" accentColor="#722ed1" collapsible defaultExpanded>
            <div className="mb-4">
              {/* 拓扑配置文件选择 */}
              <div style={{ ...configRowStyle, marginBottom: 10 }}>
                <span className="text-gray-500 text-xs">
                  <span className="text-red-500">*</span> 拓扑配置文件
                </span>
                <Select
                  value={selectedTopologyConfig || '__current__'}
                  onValueChange={(v) => handleSelectTopologyConfig(v === '__current__' ? undefined : v)}
                >
                  <SelectTrigger className="w-[180px] h-8">
                    <SelectValue placeholder="使用当前拓扑" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__current__">使用当前拓扑</SelectItem>
                    {topologyConfigs.map(c => (
                      <SelectItem key={c.name} value={c.name}>{c.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {chipGroups.length === 0 ? (
                <div className="p-3 rounded-lg border border-amber-300" style={{ background: colors.warningLight }}>
                  <span className="text-amber-600">
                    <AlertTriangle className="inline h-4 w-4 mr-1.5" />
                    请先在「互联拓扑」中配置芯片类型，或选择已保存的配置文件
                  </span>
                </div>
              ) : (
                <>
                  {chipGroups.length > 1 && (
                    <div style={{ ...configRowStyle, marginBottom: 8 }}>
                      <span className="text-xs">分析芯片类型</span>
                      <Select
                        value={selectedChipType || ''}
                        onValueChange={setSelectedChipType}
                      >
                        <SelectTrigger className="w-[140px] h-8">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {chipGroups.map(g => (
                            <SelectItem key={g.presetId || g.chipType} value={g.presetId || g.chipType}>
                              {g.chipType} ({g.totalCount * localPodCount * localRacksPerPod}个)
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}

                  {/* 拓扑结构概览 */}
                  <div className="p-2.5 rounded-lg text-xs border border-green-300 mb-2" style={{ background: colors.successLight }}>
                    <div className="flex justify-between mb-1.5">
                      <span><CheckCircle className="inline h-3.5 w-3.5 mr-1" style={{ color: colors.success }} />拓扑概览</span>
                      <span>共 <b>{maxChips}</b> 个芯片</span>
                    </div>
                    <div className="grid grid-cols-2 gap-x-3 gap-y-1" style={{ color: colors.textSecondary }}>
                      <span>Pod: {localPodCount} 个</span>
                      <span>Rack: {localRacksPerPod * localPodCount} 个</span>
                      <span>Board: {localRackConfig ? localRackConfig.boards.reduce((sum, b) => sum + b.count, 0) * localRacksPerPod * localPodCount : 0} 个</span>
                      <span>Chip: {maxChips} 个</span>
                    </div>
                  </div>

                  {/* 芯片硬件参数 */}
                  <Collapsible open={chipParamsOpen} onOpenChange={setChipParamsOpen} className="mb-3">
                    <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded bg-gray-100 hover:bg-gray-200 text-sm font-medium">
                      <span>芯片硬件参数: {hardwareConfig.chip.chip_type}</span>
                      <span className="text-gray-500">{chipParamsOpen ? '▲' : '▼'}</span>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="p-2 bg-white border border-t-0 rounded-b">
                      {/* 核心数 + 算力 (合并一行，3列) */}
                      <div className="grid grid-cols-3 gap-3 mb-2">
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">核心数</Label>
                            </TooltipTrigger>
                            <TooltipContent>计算核心数量</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={1}
                            value={hardwareConfig.chip.num_cores}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, num_cores: v ?? 8 }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">FP8 (TFLOPS)</Label>
                            </TooltipTrigger>
                            <TooltipContent>FP8 精度算力</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            value={hardwareConfig.chip.compute_tflops_fp8}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, compute_tflops_fp8: v ?? 0, compute_tflops_bf16: (v ?? 0) / 2 }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">BF16 (TFLOPS)</Label>
                            </TooltipTrigger>
                            <TooltipContent>BF16 精度算力</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            value={hardwareConfig.chip.compute_tflops_bf16}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, compute_tflops_bf16: v ?? 0, compute_tflops_fp8: (v ?? 0) * 2 }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                          />
                        </div>
                      </div>

                      {/* Memory (3列: 容量、带宽、利用率) */}
                      <div className="border-t border-dashed my-2 pt-1.5">
                        <span className="text-xs text-gray-500">Memory</span>
                      </div>
                      <div className="grid grid-cols-3 gap-3 mb-2">
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">容量 (GB)</Label>
                            </TooltipTrigger>
                            <TooltipContent>显存容量</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            value={hardwareConfig.chip.memory_capacity_gb}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, memory_capacity_gb: v ?? 0 }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">带宽 (TB/s)</Label>
                            </TooltipTrigger>
                            <TooltipContent>显存总带宽 (理论峰值)</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            step={0.1}
                            value={Number((hardwareConfig.chip.memory_bandwidth_gbps / 1000).toFixed(1))}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, memory_bandwidth_gbps: (v ?? 0) * 1000 }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">利用率</Label>
                            </TooltipTrigger>
                            <TooltipContent>显存带宽利用率 (0-1)</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            max={1}
                            step={0.01}
                            value={hardwareConfig.chip.memory_bandwidth_utilization}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, memory_bandwidth_utilization: v ?? 0.85 }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                          />
                        </div>
                      </div>

                      {/* LMEM + C2C (4列: LMEM容量、LMEM带宽、C2C带宽、C2C延迟) */}
                      <div className="border-t border-dashed my-2 pt-1.5">
                        <span className="text-xs text-gray-500">LMEM / C2C</span>
                      </div>
                      <div className="grid grid-cols-4 gap-3 mb-2">
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">LMEM (MB)</Label>
                            </TooltipTrigger>
                            <TooltipContent>LMEM 片上缓存容量</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            value={hardwareConfig.chip.lmem_capacity_mb}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, lmem_capacity_mb: v ?? 2 }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">L带宽 (GB/s)</Label>
                            </TooltipTrigger>
                            <TooltipContent>LMEM 缓存带宽</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            value={hardwareConfig.chip.lmem_bandwidth_gbps}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, lmem_bandwidth_gbps: v ?? 512 }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">C2C (GB/s)</Label>
                            </TooltipTrigger>
                            <TooltipContent>芯片间互联带宽（板内）</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            step={0.1}
                            value={hardwareConfig.chip.c2c_bandwidth_gbps}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, c2c_bandwidth_gbps: v ?? 900 }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">延迟 (us)</Label>
                            </TooltipTrigger>
                            <TooltipContent>芯片间互联延迟</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            step={0.01}
                            value={hardwareConfig.chip.c2c_latency_us}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, c2c_latency_us: v ?? 0.01 }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                          />
                        </div>
                      </div>

                      {/* 微架构参数 (更紧凑: 4列布局) */}
                      <div className="border-t border-dashed my-2 pt-1.5">
                        <span className="text-xs text-gray-500">微架构 / GEMM</span>
                      </div>
                      <div className="grid grid-cols-4 gap-3 mb-1.5">
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">Cube M</Label>
                            </TooltipTrigger>
                            <TooltipContent>矩阵单元 M 维度</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={1}
                            value={hardwareConfig.chip.cube_m}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, cube_m: v || undefined }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                            placeholder="自动"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">Cube K</Label>
                            </TooltipTrigger>
                            <TooltipContent>矩阵单元 K 维度</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={1}
                            value={hardwareConfig.chip.cube_k}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, cube_k: v || undefined }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                            placeholder="自动"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">Cube N</Label>
                            </TooltipTrigger>
                            <TooltipContent>矩阵单元 N 维度</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={1}
                            value={hardwareConfig.chip.cube_n}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, cube_n: v || undefined }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                            placeholder="自动"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">Lane 数</Label>
                            </TooltipTrigger>
                            <TooltipContent>SIMD lane 数量</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={1}
                            value={hardwareConfig.chip.lane_num}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, lane_num: v || undefined }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                            placeholder="自动"
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-4 gap-3">
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">SRAM (KB)</Label>
                            </TooltipTrigger>
                            <TooltipContent>每核 SRAM 大小</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            value={hardwareConfig.chip.sram_size_kb}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, sram_size_kb: v || undefined }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                            placeholder="自动"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">SRAM利用</Label>
                            </TooltipTrigger>
                            <TooltipContent>SRAM 可用比例 (0-1)</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            max={1}
                            step={0.01}
                            value={hardwareConfig.chip.sram_utilization}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, sram_utilization: v || undefined }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                            placeholder="自动"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">对齐字节</Label>
                            </TooltipTrigger>
                            <TooltipContent>内存对齐字节数</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={1}
                            value={hardwareConfig.chip.align_bytes}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, align_bytes: v || undefined }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                            placeholder="自动"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">重叠率</Label>
                            </TooltipTrigger>
                            <TooltipContent>计算-搬运重叠率 (0-1)</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            max={1}
                            step={0.01}
                            value={hardwareConfig.chip.compute_dma_overlap_rate}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              chip: { ...prev.chip, compute_dma_overlap_rate: v || undefined }
                            } : prev)}
                            className="w-full h-7 mt-0.5"
                            placeholder="自动"
                          />
                        </div>
                      </div>
                      </div>
                    </CollapsibleContent>
                  </Collapsible>

                  {/* Board 配置 */}
                  <Collapsible open={boardParamsOpen} onOpenChange={setBoardParamsOpen} className="mb-3">
                    <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded bg-gray-100 hover:bg-gray-200 text-sm font-medium">
                      <span>Board 配置 / 板级互联</span>
                      <span className="text-gray-500">{boardParamsOpen ? '▲' : '▼'}</span>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="p-2 bg-white border border-t-0 rounded-b">
                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">每板芯片数</Label>
                            </TooltipTrigger>
                            <TooltipContent>单个 Board 上的芯片数量</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={1}
                            max={128}
                            value={hardwareConfig.board.chips_per_board}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              board: { ...prev.board, chips_per_board: v ?? 8 }
                            } : prev)}
                            className="w-full h-8 mt-1"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">B2B 带宽 (GB/s)</Label>
                            </TooltipTrigger>
                            <TooltipContent>Board-to-Board 互联带宽</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            step={1}
                            value={hardwareConfig.board.b2b_bandwidth_gbps}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              board: { ...prev.board, b2b_bandwidth_gbps: v ?? 450 }
                            } : prev)}
                            className="w-full h-8 mt-1"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">B2B 延迟 (us)</Label>
                            </TooltipTrigger>
                            <TooltipContent>Board-to-Board 互联延迟</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            step={0.01}
                            value={hardwareConfig.board.b2b_latency_us}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              board: { ...prev.board, b2b_latency_us: v ?? 0.35 }
                            } : prev)}
                            className="w-full h-8 mt-1"
                          />
                        </div>
                      </div>
                      </div>
                    </CollapsibleContent>
                  </Collapsible>

                  {/* Rack 配置 */}
                  <Collapsible open={rackParamsOpen} onOpenChange={setRackParamsOpen} className="mb-3">
                    <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded bg-gray-100 hover:bg-gray-200 text-sm font-medium">
                      <span>Rack 配置 / 机架互联</span>
                      <span className="text-gray-500">{rackParamsOpen ? '▲' : '▼'}</span>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="p-2 bg-white border border-t-0 rounded-b">
                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">每架 Board 数</Label>
                            </TooltipTrigger>
                            <TooltipContent>单个 Rack 中的 Board 数量</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={1}
                            max={128}
                            value={hardwareConfig.rack.boards_per_rack}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              rack: { ...prev.rack, boards_per_rack: v ?? 4 }
                            } : prev)}
                            className="w-full h-8 mt-1"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">R2R 带宽 (GB/s)</Label>
                            </TooltipTrigger>
                            <TooltipContent>Rack-to-Rack 互联带宽</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            step={1}
                            value={hardwareConfig.rack.r2r_bandwidth_gbps}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              rack: { ...prev.rack, r2r_bandwidth_gbps: v ?? 200 }
                            } : prev)}
                            className="w-full h-8 mt-1"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">R2R 延迟 (us)</Label>
                            </TooltipTrigger>
                            <TooltipContent>Rack-to-Rack 互联延迟</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            step={0.01}
                            value={hardwareConfig.rack.r2r_latency_us}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              rack: { ...prev.rack, r2r_latency_us: v ?? 2 }
                            } : prev)}
                            className="w-full h-8 mt-1"
                          />
                        </div>
                      </div>
                      </div>
                    </CollapsibleContent>
                  </Collapsible>

                  {/* Pod 配置 */}
                  <Collapsible open={podParamsOpen} onOpenChange={setPodParamsOpen} className="mb-3">
                    <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded bg-gray-100 hover:bg-gray-200 text-sm font-medium">
                      <span>Pod 配置 / 集群互联</span>
                      <span className="text-gray-500">{podParamsOpen ? '▲' : '▼'}</span>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="p-2 bg-white border border-t-0 rounded-b">
                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">每 Pod Rack 数</Label>
                            </TooltipTrigger>
                            <TooltipContent>单个 Pod 中的 Rack 数量</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={1}
                            max={128}
                            value={hardwareConfig.pod.racks_per_pod}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              pod: { ...prev.pod, racks_per_pod: v ?? 1 }
                            } : prev)}
                            className="w-full h-8 mt-1"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">P2P 带宽 (GB/s)</Label>
                            </TooltipTrigger>
                            <TooltipContent>Pod-to-Pod 互联带宽</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            step={1}
                            value={hardwareConfig.pod.p2p_bandwidth_gbps}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              pod: { ...prev.pod, p2p_bandwidth_gbps: v ?? 100 }
                            } : prev)}
                            className="w-full h-8 mt-1"
                          />
                        </div>
                        <div>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Label className="text-xs text-gray-500 cursor-help">P2P 延迟 (us)</Label>
                            </TooltipTrigger>
                            <TooltipContent>Pod-to-Pod 互联延迟</TooltipContent>
                          </Tooltip>
                          <NumberInput
                            min={0}
                            step={0.01}
                            value={hardwareConfig.pod.p2p_latency_us}
                            onChange={(v) => setHardwareConfig(prev => prev ? {
                              ...prev,
                              pod: { ...prev.pod, p2p_latency_us: v ?? 5 }
                            } : prev)}
                            className="w-full h-8 mt-1"
                          />
                        </div>
                      </div>
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                </>
              )}
            </div>

            {/* 延迟配置 - 使用 Collapsible 折叠面板 */}
            <div className="mb-3">
              <Collapsible open={commParamsOpen} onOpenChange={setCommParamsOpen}>
                <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded bg-gray-100 hover:bg-gray-200 text-sm font-medium">
                  <span>互联通信参数</span>
                  <span className="text-gray-500">{commParamsOpen ? '▲' : '▼'}</span>
                </CollapsibleTrigger>
                <CollapsibleContent>
                      <div className="p-2 bg-white border border-t-0 rounded-b">
                  {/* 协议参数 */}
                  <div className="grid grid-cols-4 gap-4">
                    <div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Label className="text-xs text-gray-500 cursor-help">TP RTT (µs)</Label>
                        </TooltipTrigger>
                        <TooltipContent>Tensor Parallelism Round Trip Time: 张量并行通信的往返延迟</TooltipContent>
                      </Tooltip>
                      <NumberInput
                        min={0}
                        max={10}
                        step={0.05}
                        value={commLatencyConfig.rtt_tp_us}
                        onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, rtt_tp_us: v ?? 0.35 }))}
                        className="w-full h-8 mt-1"
                      />
                    </div>
                    <div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Label className="text-xs text-gray-500 cursor-help">EP RTT (µs)</Label>
                        </TooltipTrigger>
                        <TooltipContent>Expert Parallelism Round Trip Time: 专家并行通信的往返延迟</TooltipContent>
                      </Tooltip>
                      <NumberInput
                        min={0}
                        max={10}
                        step={0.05}
                        value={commLatencyConfig.rtt_ep_us}
                        onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, rtt_ep_us: v ?? 0.85 }))}
                        className="w-full h-8 mt-1"
                      />
                    </div>
                    <div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Label className="text-xs text-gray-500 cursor-help">链路带宽利用率</Label>
                        </TooltipTrigger>
                        <TooltipContent>链路带宽利用率: 实际可用带宽与理论峰值带宽的比例 (典型值: 0.85-0.95)</TooltipContent>
                      </Tooltip>
                      <NumberInput
                        min={0.5}
                        max={1.0}
                        step={0.01}
                        value={commLatencyConfig.bandwidth_utilization}
                        onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, bandwidth_utilization: v ?? 0.95 }))}
                        className="w-full h-8 mt-1"
                      />
                    </div>
                    <div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Label className="text-xs text-gray-500 cursor-help">同步延迟 (µs)</Label>
                        </TooltipTrigger>
                        <TooltipContent>多卡同步操作的固定开销，如 Barrier、AllReduce 初始化延迟</TooltipContent>
                      </Tooltip>
                      <NumberInput
                        min={0}
                        max={10}
                        step={0.1}
                        value={commLatencyConfig.sync_latency_us}
                        onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, sync_latency_us: v ?? 0 }))}
                        className="w-full h-8 mt-1"
                      />
                    </div>
                  </div>

                  {/* 互联相关 */}
                  <div className="border-t border-dashed my-3 pt-2">
                    <span className="text-xs text-gray-500">互联相关</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Label className="text-xs text-gray-500 cursor-help">switch_delay (µs)</Label>
                        </TooltipTrigger>
                        <TooltipContent>网络交换机的数据包转发延迟 (典型值: 0.5-2 µs)</TooltipContent>
                      </Tooltip>
                      <NumberInput
                        min={0}
                        max={10}
                        step={0.05}
                        value={commLatencyConfig.switch_delay_us}
                        onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, switch_delay_us: v ?? 1.0 }))}
                        className="w-full h-8 mt-1"
                      />
                    </div>
                    <div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Label className="text-xs text-gray-500 cursor-help">cable_delay (µs)</Label>
                        </TooltipTrigger>
                        <TooltipContent>网络线缆的光/电信号传输延迟，约 5 ns/米 (典型值: 0.01-0.05 µs)</TooltipContent>
                      </Tooltip>
                      <NumberInput
                        min={0}
                        max={1}
                        step={0.005}
                        value={commLatencyConfig.cable_delay_us}
                        onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, cable_delay_us: v ?? 0.025 }))}
                        className="w-full h-8 mt-1"
                      />
                    </div>
                  </div>

                  {/* 芯片延迟参数 */}
                  <div className="border-t border-dashed my-3 pt-2">
                    <span className="text-xs text-gray-500">芯片延迟参数</span>
                  </div>
                  <div className="grid grid-cols-3 gap-4 mb-3">
                    <div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Label className="text-xs text-gray-500 cursor-help">chip_to_chip (µs)</Label>
                        </TooltipTrigger>
                        <TooltipContent>芯片间物理互联延迟 (NVLink/SophgoLink)</TooltipContent>
                      </Tooltip>
                      <NumberInput
                        min={0}
                        max={1}
                        step={0.01}
                        value={commLatencyConfig.chip_to_chip_us}
                        onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, chip_to_chip_us: v ?? 0.2 }))}
                        className="w-full h-8 mt-1"
                      />
                    </div>
                    <div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Label className="text-xs text-gray-500 cursor-help">memory_read (µs)</Label>
                        </TooltipTrigger>
                        <TooltipContent>显存读延迟 (DDR/HBM)</TooltipContent>
                      </Tooltip>
                      <NumberInput
                        min={0}
                        max={1}
                        step={0.01}
                        value={commLatencyConfig.memory_read_latency_us}
                        onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, memory_read_latency_us: v ?? 0.15 }))}
                        className="w-full h-8 mt-1"
                      />
                    </div>
                    <div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Label className="text-xs text-gray-500 cursor-help">memory_write (µs)</Label>
                        </TooltipTrigger>
                        <TooltipContent>显存写延迟 (DDR/HBM)</TooltipContent>
                      </Tooltip>
                      <NumberInput
                        min={0}
                        max={1}
                        step={0.01}
                        value={commLatencyConfig.memory_write_latency_us}
                        onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, memory_write_latency_us: v ?? 0.01 }))}
                        className="w-full h-8 mt-1"
                      />
                    </div>
                    <div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Label className="text-xs text-gray-500 cursor-help">noc_latency (µs)</Label>
                        </TooltipTrigger>
                        <TooltipContent>片上网络延迟 (NoC)</TooltipContent>
                      </Tooltip>
                      <NumberInput
                        min={0}
                        max={1}
                        step={0.01}
                        value={commLatencyConfig.noc_latency_us}
                        onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, noc_latency_us: v ?? 0.05 }))}
                        className="w-full h-8 mt-1"
                      />
                    </div>
                    <div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Label className="text-xs text-gray-500 cursor-help">die_to_die (µs)</Label>
                        </TooltipTrigger>
                        <TooltipContent>Die-to-Die 延迟 (多Die芯片)</TooltipContent>
                      </Tooltip>
                      <NumberInput
                        min={0}
                        max={1}
                        step={0.01}
                        value={commLatencyConfig.die_to_die_latency_us}
                        onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, die_to_die_latency_us: v ?? 0.04 }))}
                        className="w-full h-8 mt-1"
                      />
                    </div>
                  </div>

                  {/* 计算结果：通信启动开销 */}
                  <div className="border-t border-dashed my-3 pt-2">
                    <span className="text-xs text-gray-500">通信启动开销 (start_lat)</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="p-2 bg-gray-100 rounded border border-gray-300 cursor-help">
                          <span className="text-xs text-gray-500">AllReduce start_lat</span>
                          <div className="text-sm font-medium text-blue-500">
                            {(2 * commLatencyConfig.chip_to_chip_us + commLatencyConfig.memory_read_latency_us + commLatencyConfig.memory_write_latency_us + commLatencyConfig.noc_latency_us + 2 * commLatencyConfig.die_to_die_latency_us).toFixed(2)} µs
                          </div>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <div className="text-xs">
                          <div className="font-medium mb-1">AllReduce start_lat 计算公式:</div>
                          <div className="font-mono">2×chip_to_chip + memory_read + memory_write + noc + 2×die_to_die</div>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="p-2 bg-gray-100 rounded border border-gray-300 cursor-help">
                          <span className="text-xs text-gray-500">Dispatch/Combine start_lat</span>
                          <div className="text-sm font-medium text-purple-500">
                            {(2 * commLatencyConfig.chip_to_chip_us + commLatencyConfig.memory_read_latency_us + commLatencyConfig.memory_write_latency_us + commLatencyConfig.noc_latency_us + 2 * commLatencyConfig.die_to_die_latency_us + 2 * commLatencyConfig.switch_delay_us + 2 * commLatencyConfig.cable_delay_us).toFixed(2)} µs
                          </div>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <div className="text-xs">
                          <div className="font-medium mb-1">Dispatch/Combine start_lat 计算公式:</div>
                          <div className="font-mono">2×chip_to_chip + memory_read + memory_write + noc + 2×die_to_die + 2×switch + 2×cable</div>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  </div>
                </CollapsibleContent>
              </Collapsible>

              {/* 保存、另存为、重置按钮 */}
              <div className="flex gap-2 mt-3">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleSaveConfig}
                  disabled={saveLoading || !selectedTopologyConfig}
                >
                  <Save className="h-3.5 w-3.5 mr-1" />
                  保存
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSaveAsModalOpen(true)}
                >
                  <Copy className="h-3.5 w-3.5 mr-1" />
                  另存为
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleResetDelayConfig}
                >
                  <RotateCcw className="h-3.5 w-3.5 mr-1" />
                  重置
                </Button>
              </div>
            </div>
            </BaseCard>
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
        <BaseCard title="分析任务" accentColor="#fa8c16" collapsible defaultExpanded style={{ marginTop: 16 }}>
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
    </TooltipProvider>
  )
}

export { AnalysisResultDisplay }
export default DeploymentAnalysisPanel
