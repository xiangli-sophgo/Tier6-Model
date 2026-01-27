/**
 * LLM 部署分析面板
 *
 * 提供模型配置、推理配置、硬件配置、并行策略配置和分析结果展示
 */

import React, { useState, useCallback, useRef } from 'react'
import { useWorkbench } from '../../../contexts/WorkbenchContext'
import {
  Typography,
  Button,
  Select,
  message,
  InputNumber,
  Tooltip,
  Row,
  Col,
  Modal,
  Input,
  Space,
  Collapse,
  Divider,
  Switch,
} from 'antd'
import {
  PlayCircleOutlined,
  SearchOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  SaveOutlined,
  CopyOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
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
import { ExecutorConfigPanel } from './ExecutorConfigPanel'
import { useTaskWebSocket, TaskUpdate } from '../../../hooks/useTaskWebSocket'

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

  // 分区搜索开关（默认关闭，提升速度）
  const [enablePartitionSearch, setEnablePartitionSearch] = useState<boolean>(false)

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
      message.success(`已加载拓扑配置: ${config.name}`)
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
      // 如果没有拓扑配置，使用默认值
      console.warn('未找到拓扑配置，使用默认硬件配置')
      const defaultConfig: HardwareConfig = {
        chip: {
          chip_type: 'SG2260',
          flops_dtype: 'BF16',
          compute_tflops_fp16: 800,
          compute_tops_int8: 1600,
          num_cores: 256,
          memory_gb: 64,
          memory_bandwidth_gbps: 1200,
          memory_bandwidth_utilization: 0.85,
        },
        node: {
          chips_per_node: 8,
          intra_node_bandwidth_gbps: 450,
          intra_node_latency_us: 0.35,
        },
        cluster: {
          num_nodes: 1,
          inter_node_bandwidth_gbps: 200,
          inter_node_latency_us: 2,
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
    const maxTP = Math.min(128, modelConfig.num_attention_heads, hardwareConfig.node.chips_per_node)

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
      message.warning('请先选择一个配置文件，或使用「另存为」创建新配置')
      return
    }
    const existingConfig = topologyConfigs.find(c => c.name === selectedTopologyConfig)
    if (!existingConfig) {
      message.error('配置文件不存在')
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
      message.success(`已保存配置: ${selectedTopologyConfig}`)
    } catch (error) {
      console.error('保存配置失败:', error)
      message.error('保存配置失败')
    } finally {
      setSaveLoading(false)
    }
  }, [selectedTopologyConfig, topologyConfigs, commLatencyConfig, refreshTopologyConfigs])

  // 另存为新配置
  const handleSaveAsConfig = useCallback(async () => {
    if (!newConfigName.trim()) {
      message.warning('请输入配置名称')
      return
    }
    // 检查名称是否已存在
    if (topologyConfigs.some(c => c.name === newConfigName.trim())) {
      message.error('配置名称已存在，请使用其他名称')
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
        board_configs: baseConfig?.board_configs || {
          u1: { count: 0, chips: { npu: 0, cpu: 0 } },
          u2: { count: 0, chips: { npu: 0, cpu: 0 } },
          u4: { count: 0, chips: { npu: 0, cpu: 0 } },
        },
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
      message.success(`已创建新配置: ${newConfigName.trim()}`)
    } catch (error) {
      console.error('另存为配置失败:', error)
      message.error('另存为配置失败')
    } finally {
      setSaveLoading(false)
    }
  }, [newConfigName, newConfigDesc, topologyConfigs, selectedTopologyConfig, localPodCount, localRacksPerPod, localRackConfig, commLatencyConfig, refreshTopologyConfigs])

  // 重置延迟设置为默认值
  const handleResetDelayConfig = useCallback(() => {
    setCommLatencyConfig({ ...DEFAULT_COMM_LATENCY_CONFIG })
    message.success('已重置延迟设置为默认值')
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
    console.log('[DEBUG Panel] handleTaskUpdate called:', update)
    console.log('[DEBUG Panel] Current tasks:', analysisTasksRef.current.map(t => ({ id: t.id, status: t.status })))

    // 直接检查任务是否存在于本地任务列表中（使用 ref 访问最新值）
    const taskExists = analysisTasksRef.current.some(t => t.id === update.task_id)
    console.log('[DEBUG Panel] Task exists:', taskExists, 'for task_id:', update.task_id)

    if (!taskExists) {
      // 可能是其他页面的任务或已删除的任务，忽略
      console.log('[DEBUG Panel] Task not found, ignoring')
      return
    }

    const localTaskId = update.task_id
    console.log('[DEBUG Panel] Updating task:', localTaskId, 'with progress:', update.progress)

    if (update.status === 'running') {
      // 更新进度
      updateTask(localTaskId, {
        progress: { current: Math.round(update.progress), total: 100 },
      })
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
            throughput: topPlan.throughput as number,
            mfu: topPlan.mfu as number,
            mbu: topPlan.mbu as number,
          })
        } else {
          updateTask(localTaskId, {
            status: 'failed',
            endTime: Date.now(),
            error: '未找到可行方案',
          })
        }
      } catch (error) {
        console.error('获取任务结果失败:', error)
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
    onConnect: () => console.log('[DeploymentAnalysis] WebSocket connected'),
    onDisconnect: () => console.log('[DeploymentAnalysis] WebSocket disconnected'),
  })

  // 查看任务结果（跳转到结果管理页面）
  const viewTaskResult = useCallback((task: AnalysisTask) => {
    // 跳转到结果管理页面
    ui.setViewMode('results')
    message.info(`已跳转到结果管理，请查找实验: ${task.experimentName || task.benchmarkName || task.modelName}`)
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

  // 计算最大可用芯片数
  const maxChips = hardwareConfig ? hardwareConfig.node.chips_per_node * hardwareConfig.cluster.num_nodes : 0

  // 运行分析（提交到后端执行）
  const handleRunAnalysis = useCallback(async () => {
    if (!hardwareConfig) return
    if (!topology) {
      message.error('拓扑配置未加载，请先配置拓扑')
      return
    }

    const strategy = parallelismMode === 'manual' ? manualStrategy : { dp: 1, tp: 1, pp: 1, ep: 1, sp: 1, moe_tp: 1 }

    // 生成 Benchmark 配置名称
    const benchmarkName = generateBenchmarkName(modelConfig, inferenceConfig)
    // 使用用户输入的实验名称，如果为空则使用 Benchmark 名称
    const finalExperimentName = experimentName.trim() || benchmarkName

    // 生成临时任务 ID（提交后会更新为后端返回的 ID）
    const tempTaskId = `temp-${Date.now()}`

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
      chips: maxChips,
    }
    addTask(newTask)

    // 捕获当前配置
    const currentModelConfig = { ...modelConfig }
    const currentInferenceConfig = { ...inferenceConfig }
    const currentHardwareConfig = { ...hardwareConfig }
    const currentParallelismMode = parallelismMode
    const currentManualStrategy = { ...manualStrategy }

    try {
      // 构建完整的拓扑配置（物理拓扑 + 通信延迟配置）
      const fullTopology = {
        ...topology,
        comm_latency_config: { ...commLatencyConfig },
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
      })

      const backendTaskId = response.task_id

      // 更新任务 ID 为后端返回的真实 ID
      setAnalysisTasks(prev => prev.map(t =>
        t.id === tempTaskId ? { ...t, id: backendTaskId } : t
      ))

      message.success('任务已提交')
    } catch (error) {
      console.error('提交任务失败:', error)
      const msg = error instanceof Error ? error.message : '未知错误'
      // 提交失败，更新任务状态为失败
      updateTask(tempTaskId, {
        status: 'failed',
        endTime: Date.now(),
        error: msg,
      })
      message.error(`提交任务失败: ${msg}`)
    }
  }, [experimentName, taskMaxWorkers, modelConfig, inferenceConfig, hardwareConfig, parallelismMode, manualStrategy, maxChips, topology, addTask, updateTask, commLatencyConfig, setAnalysisTasks])

  // 如果硬件配置未加载，显示提示（不再是加载中）
  if (!hardwareConfig) {
    return (
      <div style={{ padding: 40, textAlign: 'center' }}>
        <Text type="secondary" style={{ display: 'block', marginBottom: 16, fontSize: 14 }}>
          未找到拓扑配置
        </Text>
        <Text type="secondary" style={{ fontSize: 13 }}>
          请先在「拓扑设置」页面配置芯片和网络拓扑
        </Text>
      </div>
    )
  }

  return (
    <div style={{ padding: 0 }}>
      {/* 上方：Benchmark 设置和部署设置（左右两列） */}
      <Row gutter={32} style={{ marginBottom: 16 }}>
        {/* 左列：Benchmark 设置 + 并行策略 */}
        <Col span={12}>
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
            <div style={{ marginTop: 16, marginBottom: 8, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <span style={{ fontSize: 13, color: '#666', marginRight: 8 }}>启用 Tile 搜索</span>
                <Tooltip title="开启时使用最优tile搜索以获得最高精度，关闭时使用固定tile大小以显著提升评估速度">
                  <span style={{ color: '#8c8c8c', cursor: 'help' }}>ⓘ</span>
                </Tooltip>
              </div>
              <Switch
                checked={enableTileSearch}
                onChange={setEnableTileSearch}
                checkedChildren="开"
                unCheckedChildren="关"
              />
            </div>

            {/* 分区搜索开关 */}
            <div style={{ marginTop: 12, marginBottom: 8, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <span style={{ fontSize: 13, color: '#666', marginRight: 8 }}>启用分区搜索</span>
                <Tooltip title="开启时搜索最优分区策略（极慢，单个GEMM需100+秒），关闭时使用固定分区（推荐，速度提升100倍）">
                  <span style={{ color: '#8c8c8c', cursor: 'help' }}>ⓘ</span>
                </Tooltip>
              </div>
              <Switch
                checked={enablePartitionSearch}
                onChange={setEnablePartitionSearch}
                checkedChildren="开"
                unCheckedChildren="关"
              />
            </div>

            {/* 最大模拟 token 数 */}
            <div style={{ marginTop: 12, marginBottom: 8, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <span style={{ fontSize: 13, color: '#666', marginRight: 8 }}>最大模拟 Token 数</span>
                <Tooltip title="Decode 阶段模拟的 token 数量，值越小评估越快但精度略降。推荐：快速评估用 1-2，精确评估用 4-8">
                  <span style={{ color: '#8c8c8c', cursor: 'help' }}>ⓘ</span>
                </Tooltip>
              </div>
              <InputNumber
                min={1}
                max={16}
                value={maxSimulatedTokens}
                onChange={(value) => setMaxSimulatedTokens(value || 4)}
                style={{ width: 80 }}
              />
            </div>

            {/* 实验名称和任务并发数 */}
            <Row gutter={16} style={{ marginTop: 16 }}>
              <Col span={14}>
                <div style={{ marginBottom: 6, fontSize: 13, color: '#666' }}>实验名称</div>
                <Input
                  placeholder="留空则使用 Benchmark 名称"
                  value={experimentName}
                  onChange={(e) => setExperimentName(e.target.value)}
                  allowClear
                />
              </Col>
              <Col span={10}>
                <div style={{ marginBottom: 6, fontSize: 13, color: '#666' }}>
                  任务并发数
                  <Tooltip title="本次评估使用的 worker 数量（1-16）">
                    <span style={{ marginLeft: 4, color: '#8c8c8c', cursor: 'help' }}>ⓘ</span>
                  </Tooltip>
                </div>
                <InputNumber
                  min={1}
                  max={16}
                  value={taskMaxWorkers}
                  onChange={(value) => setTaskMaxWorkers(value || 4)}
                  style={{ width: '100%' }}
                />
              </Col>
            </Row>

            {/* 运行按钮 */}
            <Button
              type="primary"
              icon={parallelismMode === 'auto' ? <SearchOutlined /> : <PlayCircleOutlined />}
              onClick={handleRunAnalysis}
              block
              size="large"
              style={{
                marginTop: 16,
                height: 44,
                borderRadius: 8,
                background: colors.primary,
                boxShadow: '0 2px 8px rgba(94, 106, 210, 0.3)',
              }}
            >
              {parallelismMode === 'auto' ? '开始方案评估' : '运行分析'}
            </Button>
          </BaseCard>
        </Col>

        {/* 右列：拓扑设置 */}
        <Col span={12}>
          <BaseCard title="拓扑设置" accentColor="#722ed1" collapsible defaultExpanded>
          <div style={{ marginBottom: 16 }}>
            {/* 拓扑配置文件选择 */}
            <div style={{ ...configRowStyle, marginBottom: 10 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                <span style={{ color: '#ff4d4f' }}>*</span> 拓扑配置文件
              </Text>
              <Select
                size="small"
                value={selectedTopologyConfig}
                onChange={handleSelectTopologyConfig}
                placeholder="使用当前拓扑"
                allowClear
                style={{ width: 180 }}
                options={topologyConfigs.map(c => ({
                  value: c.name,
                  label: c.name,
                }))}
              />
            </div>

            {chipGroups.length === 0 ? (
              <div style={{ padding: 12, background: colors.warningLight, borderRadius: 8, border: '1px solid #ffd591' }}>
                <Text type="warning">
                  <WarningOutlined style={{ marginRight: 6 }} />
                  请先在「互联拓扑」中配置芯片类型，或选择已保存的配置文件
                </Text>
              </div>
            ) : (
              <>
                {chipGroups.length > 1 && (
                  <div style={{ ...configRowStyle, marginBottom: 8 }}>
                    <Text style={{ fontSize: 12 }}>分析芯片类型</Text>
                    <Select
                      size="small"
                      value={selectedChipType}
                      onChange={setSelectedChipType}
                      style={{ width: 140 }}
                      options={chipGroups.map(g => ({
                        value: g.presetId || g.chipType,
                        label: `${g.chipType} (${g.totalCount * localPodCount * localRacksPerPod}个)`,
                      }))}
                    />
                  </div>
                )}

                {/* 拓扑结构概览 */}
                <div style={{ padding: 10, background: colors.successLight, borderRadius: 8, fontSize: 12, border: '1px solid #b7eb8f', marginBottom: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                    <Text><CheckCircleOutlined style={{ color: colors.success, marginRight: 4 }} />拓扑概览</Text>
                    <Text>共 <b>{hardwareConfig.node.chips_per_node * hardwareConfig.cluster.num_nodes}</b> 个芯片</Text>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px', color: colors.textSecondary }}>
                    <span>Pod: {localPodCount} 个</span>
                    <span>Rack: {localRacksPerPod * localPodCount} 个</span>
                    <span>Board: {localRackConfig ? localRackConfig.boards.reduce((sum, b) => sum + b.count, 0) * localRacksPerPod * localPodCount : 0} 个</span>
                    <span>Chip: {hardwareConfig.node.chips_per_node * hardwareConfig.cluster.num_nodes} 个</span>
                  </div>
                </div>

                {/* 芯片硬件参数 */}
                <Collapse
                  size="small"
                  style={{ marginBottom: 12, background: 'transparent' }}
                  defaultActiveKey={[]}
                  expandIconPosition="start"
                  className="delay-settings-collapse"
                  items={[
                    {
                      key: 'chip',
                      label: `芯片硬件参数: ${hardwareConfig.chip.chip_type}`,
                      children: (
                        <div>
                          {/* 算力 */}
                          <Row gutter={[16, 8]}>
                            <Col span={8}>
                              <Tooltip title="FP8 精度算力 (通常是 BF16/FP16 的 2 倍)">
                                <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>FP8 (TFLOPS)</Text></div>
                              </Tooltip>
                              <InputNumber
                                size="small"
                                min={0}
                                value={hardwareConfig.chip.compute_tflops_fp8 ?? (hardwareConfig.chip.compute_tflops_fp16 * 2)}
                                onChange={(v) => setHardwareConfig(prev => prev ? {
                                  ...prev,
                                  chip: { ...prev.chip, compute_tflops_fp8: v ?? 0, compute_tflops_fp16: (v ?? 0) / 2 }
                                } : prev)}
                                style={{ width: '100%' }}
                              />
                            </Col>
                            <Col span={8}>
                              <Tooltip title={`${hardwareConfig.chip.flops_dtype || 'BF16'} 精度算力`}>
                                <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>{hardwareConfig.chip.flops_dtype || 'BF16'} (TFLOPS)</Text></div>
                              </Tooltip>
                              <InputNumber
                                size="small"
                                min={0}
                                value={hardwareConfig.chip.compute_tflops_fp16}
                                onChange={(v) => setHardwareConfig(prev => prev ? {
                                  ...prev,
                                  chip: { ...prev.chip, compute_tflops_fp16: v ?? 0, compute_tflops_fp8: (v ?? 0) * 2 }
                                } : prev)}
                                style={{ width: '100%' }}
                              />
                            </Col>
                          </Row>

                          {/* Memory */}
                          <Divider orientation="left" orientationMargin={0} style={{ margin: '12px 0 8px', fontSize: 12 }}>Memory</Divider>
                          <Row gutter={[16, 8]}>
                            <Col span={8}>
                              <Tooltip title="内存容量">
                                <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>容量 (GB)</Text></div>
                              </Tooltip>
                              <InputNumber
                                size="small"
                                min={0}
                                value={hardwareConfig.chip.memory_gb}
                                onChange={(v) => setHardwareConfig(prev => prev ? {
                                  ...prev,
                                  chip: { ...prev.chip, memory_gb: v ?? 0 }
                                } : prev)}
                                style={{ width: '100%' }}
                              />
                            </Col>
                            <Col span={8}>
                              <Tooltip title="内存总带宽 (理论峰值)">
                                <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>带宽 (TB/s)</Text></div>
                              </Tooltip>
                              <InputNumber
                                size="small"
                                min={0}
                                step={0.1}
                                value={Number((hardwareConfig.chip.memory_bandwidth_gbps / 1000).toFixed(1))}
                                onChange={(v) => setHardwareConfig(prev => prev ? {
                                  ...prev,
                                  chip: { ...prev.chip, memory_bandwidth_gbps: (v ?? 0) * 1000 }
                                } : prev)}
                                style={{ width: '100%' }}
                              />
                            </Col>
                            <Col span={8}>
                              <Tooltip title="LMEM/SRAM 片上缓存容量">
                                <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>LMEM (MB)</Text></div>
                              </Tooltip>
                              <InputNumber
                                size="small"
                                min={0}
                                value={hardwareConfig.chip.lmem_mb ?? 2}
                                onChange={(v) => setHardwareConfig(prev => prev ? {
                                  ...prev,
                                  chip: { ...prev.chip, lmem_mb: v ?? 0 }
                                } : prev)}
                                style={{ width: '100%' }}
                              />
                            </Col>
                          </Row>

                          {/* C2C BW / 互联带宽 */}
                          <Divider orientation="left" orientationMargin={0} style={{ margin: '12px 0 8px', fontSize: 12 }}>C2C BW / 互联带宽</Divider>
                          <Row gutter={[16, 8]}>
                            <Col span={8}>
                              <Tooltip title="芯片间互联单向带宽">
                                <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>单向 (GB/s)</Text></div>
                              </Tooltip>
                              <InputNumber
                                size="small"
                                min={0}
                                precision={1}
                                value={hardwareConfig.chip.c2c_bandwidth_gbps ?? hardwareConfig.node.intra_node_bandwidth_gbps}
                                onChange={(v) => setHardwareConfig(prev => prev ? {
                                  ...prev,
                                  chip: { ...prev.chip, c2c_bandwidth_gbps: v ?? 0 }
                                } : prev)}
                                style={{ width: '100%' }}
                              />
                            </Col>
                            <Col span={8}>
                              <Tooltip title="芯片间互联双向带宽">
                                <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>双向 (GB/s)</Text></div>
                              </Tooltip>
                              <InputNumber
                                size="small"
                                min={0}
                                precision={1}
                                value={hardwareConfig.chip.c2c_bandwidth_bidirectional_gbps ?? 996}
                                onChange={(v) => setHardwareConfig(prev => prev ? {
                                  ...prev,
                                  chip: { ...prev.chip, c2c_bandwidth_bidirectional_gbps: v ?? 0 }
                                } : prev)}
                                style={{ width: '100%' }}
                              />
                            </Col>
                          </Row>
                        </div>
                      ),
                    },
                  ]}
                />
              </>
            )}
          </div>

          {/* 延迟配置 - 使用 Collapse 折叠面板 */}
          <div style={{ marginBottom: 12 }}>
            <Collapse
              size="small"
              style={{ marginBottom: 12, background: 'transparent' }}
              defaultActiveKey={[]}
              expandIconPosition="start"
              className="delay-settings-collapse"
              items={[
                {
                  key: 'delay',
                  label: '互联通信参数',
                  children: (
                    <div>
                      {/* 协议参数 */}
                      <Row gutter={[16, 8]}>
                        <Col span={8}>
                          <Tooltip title="Tensor Parallelism Round Trip Time: 张量并行通信的往返延迟">
                            <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>TP RTT (µs)</Text></div>
                          </Tooltip>
                          <InputNumber
                            size="small"
                            min={0}
                            max={10}
                            step={0.05}
                            value={commLatencyConfig.rtt_tp_us}
                            onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, rtt_tp_us: v ?? 0.35 }))}
                            style={{ width: '100%' }}
                          />
                        </Col>
                        <Col span={8}>
                          <Tooltip title="Expert Parallelism Round Trip Time: 专家并行通信的往返延迟">
                            <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>EP RTT (µs)</Text></div>
                          </Tooltip>
                          <InputNumber
                            size="small"
                            min={0}
                            max={10}
                            step={0.05}
                            value={commLatencyConfig.rtt_ep_us}
                            onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, rtt_ep_us: v ?? 0.85 }))}
                            style={{ width: '100%' }}
                          />
                        </Col>
                        <Col span={8}>
                          <Tooltip title="链路带宽利用率: 实际可用带宽与理论峰值带宽的比例 (典型值: 0.85-0.95)">
                            <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>链路带宽利用率</Text></div>
                          </Tooltip>
                          <InputNumber
                            size="small"
                            min={0.5}
                            max={1.0}
                            step={0.01}
                            value={commLatencyConfig.bandwidth_utilization}
                            onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, bandwidth_utilization: v ?? 0.95 }))}
                            style={{ width: '100%' }}
                          />
                        </Col>
                        <Col span={8}>
                          <Tooltip title="多卡同步操作的固定开销，如 Barrier、AllReduce 初始化延迟">
                            <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>同步延迟 (µs)</Text></div>
                          </Tooltip>
                          <InputNumber
                            size="small"
                            min={0}
                            max={10}
                            step={0.1}
                            value={commLatencyConfig.sync_latency_us}
                            onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, sync_latency_us: v ?? 0 }))}
                            style={{ width: '100%' }}
                          />
                        </Col>
                      </Row>

                      {/* 互联相关 */}
                      <Divider orientation="left" orientationMargin={0} style={{ margin: '12px 0 8px', fontSize: 12 }}>互联相关</Divider>
                      <Row gutter={[16, 8]}>
                        <Col span={8}>
                          <Tooltip title="网络交换机的数据包转发延迟 (典型值: 0.5-2 µs)">
                            <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>switch_delay (µs)</Text></div>
                          </Tooltip>
                          <InputNumber
                            size="small"
                            min={0}
                            max={10}
                            step={0.05}
                            value={commLatencyConfig.switch_delay_us}
                            onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, switch_delay_us: v ?? 1.0 }))}
                            style={{ width: '100%' }}
                          />
                        </Col>
                        <Col span={8}>
                          <Tooltip title="网络线缆的光/电信号传输延迟，约 5 ns/米 (典型值: 0.01-0.05 µs)">
                            <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>cable_delay (µs)</Text></div>
                          </Tooltip>
                          <InputNumber
                            size="small"
                            min={0}
                            max={1}
                            step={0.005}
                            value={commLatencyConfig.cable_delay_us}
                            onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, cable_delay_us: v ?? 0.025 }))}
                            style={{ width: '100%' }}
                          />
                        </Col>
                      </Row>

                      {/* 芯片延迟参数 */}
                      <Divider orientation="left" orientationMargin={0} style={{ margin: '12px 0 8px', fontSize: 12 }}>芯片延迟参数</Divider>
                      <Row gutter={[16, 8]}>
                        <Col span={8}>
                          <Tooltip title="芯片间物理互联延迟 (NVLink/SophgoLink)">
                            <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>chip_to_chip (µs)</Text></div>
                          </Tooltip>
                          <InputNumber
                            size="small"
                            min={0}
                            max={1}
                            step={0.01}
                            value={commLatencyConfig.chip_to_chip_us}
                            onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, chip_to_chip_us: v ?? 0.2 }))}
                            style={{ width: '100%' }}
                          />
                        </Col>
                        <Col span={8}>
                          <Tooltip title="显存读延迟 (DDR/HBM)">
                            <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>memory_read (µs)</Text></div>
                          </Tooltip>
                          <InputNumber
                            size="small"
                            min={0}
                            max={1}
                            step={0.01}
                            value={commLatencyConfig.memory_read_latency_us}
                            onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, memory_read_latency_us: v ?? 0.15 }))}
                            style={{ width: '100%' }}
                          />
                        </Col>
                        <Col span={8}>
                          <Tooltip title="显存写延迟 (DDR/HBM)">
                            <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>memory_write (µs)</Text></div>
                          </Tooltip>
                          <InputNumber
                            size="small"
                            min={0}
                            max={1}
                            step={0.01}
                            value={commLatencyConfig.memory_write_latency_us}
                            onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, memory_write_latency_us: v ?? 0.01 }))}
                            style={{ width: '100%' }}
                          />
                        </Col>
                        <Col span={8}>
                          <Tooltip title="片上网络延迟 (NoC)">
                            <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>noc_latency (µs)</Text></div>
                          </Tooltip>
                          <InputNumber
                            size="small"
                            min={0}
                            max={1}
                            step={0.01}
                            value={commLatencyConfig.noc_latency_us}
                            onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, noc_latency_us: v ?? 0.05 }))}
                            style={{ width: '100%' }}
                          />
                        </Col>
                        <Col span={8}>
                          <Tooltip title="Die-to-Die 延迟 (多Die芯片)">
                            <div style={{ marginBottom: 4 }}><Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>die_to_die (µs)</Text></div>
                          </Tooltip>
                          <InputNumber
                            size="small"
                            min={0}
                            max={1}
                            step={0.01}
                            value={commLatencyConfig.die_to_die_latency_us}
                            onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, die_to_die_latency_us: v ?? 0.04 }))}
                            style={{ width: '100%' }}
                          />
                        </Col>
                      </Row>

                      {/* 计算结果：通信启动开销 */}
                      <Divider orientation="left" orientationMargin={0} style={{ margin: '12px 0 8px', fontSize: 12 }}>通信启动开销 (start_lat)</Divider>
                      <Row gutter={[16, 8]}>
                        <Col span={12}>
                          <Tooltip
                            title={
                              <div style={{ fontSize: 12 }}>
                                <div style={{ fontWeight: 500, marginBottom: 4 }}>AllReduce start_lat 计算公式:</div>
                                <div style={{ fontFamily: 'monospace' }}>2×chip_to_chip + memory_read + memory_write + noc + 2×die_to_die</div>
                                <div style={{ marginTop: 8, borderTop: '1px solid rgba(255,255,255,0.2)', paddingTop: 8 }}>
                                  <div>= 2×{commLatencyConfig.chip_to_chip_us} + {commLatencyConfig.memory_read_latency_us} + {commLatencyConfig.memory_write_latency_us} + {commLatencyConfig.noc_latency_us} + 2×{commLatencyConfig.die_to_die_latency_us}</div>
                                  <div style={{ fontWeight: 500, marginTop: 4 }}>= {(2 * commLatencyConfig.chip_to_chip_us + commLatencyConfig.memory_read_latency_us + commLatencyConfig.memory_write_latency_us + commLatencyConfig.noc_latency_us + 2 * commLatencyConfig.die_to_die_latency_us).toFixed(2)} µs</div>
                                </div>
                              </div>
                            }
                            placement="top"
                          >
                            <div style={{
                              padding: '8px 12px',
                              background: '#f5f5f5',
                              borderRadius: 4,
                              cursor: 'help',
                              border: '1px solid #d9d9d9'
                            }}>
                              <Text type="secondary" style={{ fontSize: 12 }}>AllReduce start_lat</Text>
                              <div style={{ fontSize: 14, fontWeight: 500, color: '#1890ff' }}>
                                {(2 * commLatencyConfig.chip_to_chip_us + commLatencyConfig.memory_read_latency_us + commLatencyConfig.memory_write_latency_us + commLatencyConfig.noc_latency_us + 2 * commLatencyConfig.die_to_die_latency_us).toFixed(2)} µs
                              </div>
                            </div>
                          </Tooltip>
                        </Col>
                        <Col span={12}>
                          <Tooltip
                            title={
                              <div style={{ fontSize: 12 }}>
                                <div style={{ fontWeight: 500, marginBottom: 4 }}>Dispatch/Combine start_lat 计算公式:</div>
                                <div style={{ fontFamily: 'monospace' }}>2×chip_to_chip + memory_read + memory_write + noc + 2×die_to_die + 2×switch + 2×cable</div>
                                <div style={{ marginTop: 8, borderTop: '1px solid rgba(255,255,255,0.2)', paddingTop: 8 }}>
                                  <div>= 2×{commLatencyConfig.chip_to_chip_us} + {commLatencyConfig.memory_read_latency_us} + {commLatencyConfig.memory_write_latency_us} + {commLatencyConfig.noc_latency_us} + 2×{commLatencyConfig.die_to_die_latency_us} + 2×{commLatencyConfig.switch_delay_us} + 2×{commLatencyConfig.cable_delay_us}</div>
                                  <div style={{ fontWeight: 500, marginTop: 4 }}>= {(2 * commLatencyConfig.chip_to_chip_us + commLatencyConfig.memory_read_latency_us + commLatencyConfig.memory_write_latency_us + commLatencyConfig.noc_latency_us + 2 * commLatencyConfig.die_to_die_latency_us + 2 * commLatencyConfig.switch_delay_us + 2 * commLatencyConfig.cable_delay_us).toFixed(2)} µs</div>
                                </div>
                              </div>
                            }
                            placement="top"
                          >
                            <div style={{
                              padding: '8px 12px',
                              background: '#f5f5f5',
                              borderRadius: 4,
                              cursor: 'help',
                              border: '1px solid #d9d9d9'
                            }}>
                              <Text type="secondary" style={{ fontSize: 12 }}>Dispatch/Combine start_lat</Text>
                              <div style={{ fontSize: 14, fontWeight: 500, color: '#722ed1' }}>
                                {(2 * commLatencyConfig.chip_to_chip_us + commLatencyConfig.memory_read_latency_us + commLatencyConfig.memory_write_latency_us + commLatencyConfig.noc_latency_us + 2 * commLatencyConfig.die_to_die_latency_us + 2 * commLatencyConfig.switch_delay_us + 2 * commLatencyConfig.cable_delay_us).toFixed(2)} µs
                              </div>
                            </div>
                          </Tooltip>
                        </Col>
                      </Row>
                    </div>
                  ),
                },
              ]}
            />
            <style>{`
              .delay-settings-collapse .ant-collapse-header {
                background: #f0f0f0 !important;
                border-radius: 4px !important;
                font-weight: 500;
              }
              .delay-settings-collapse .ant-collapse-content {
                background: #fff;
              }
            `}</style>

            {/* 保存、另存为、重置按钮 */}
            <Space>
              <Button
                size="small"
                icon={<SaveOutlined />}
                onClick={handleSaveConfig}
                loading={saveLoading}
                disabled={!selectedTopologyConfig}
              >
                保存
              </Button>
              <Button
                size="small"
                icon={<CopyOutlined />}
                onClick={() => setSaveAsModalOpen(true)}
              >
                另存为
              </Button>
              <Button
                size="small"
                icon={<ReloadOutlined />}
                onClick={handleResetDelayConfig}
              >
                重置
              </Button>
            </Space>
          </div>
          </BaseCard>
        </Col>
      </Row>

      {/* 另存为弹窗 */}
      <Modal
        title="另存为新配置"
        open={saveAsModalOpen}
        onOk={handleSaveAsConfig}
        onCancel={() => {
          setSaveAsModalOpen(false)
          setNewConfigName('')
          setNewConfigDesc('')
        }}
        confirmLoading={saveLoading}
        okText="保存"
        cancelText="取消"
      >
        <div style={{ marginBottom: 16 }}>
          <Text style={{ display: 'block', marginBottom: 8 }}>配置名称 <span style={{ color: '#ff4d4f' }}>*</span></Text>
          <Input
            value={newConfigName}
            onChange={(e) => setNewConfigName(e.target.value)}
            placeholder="请输入配置名称"
          />
        </div>
        <div>
          <Text style={{ display: 'block', marginBottom: 8 }}>描述 (可选)</Text>
          <Input.TextArea
            value={newConfigDesc}
            onChange={(e) => setNewConfigDesc(e.target.value)}
            placeholder="请输入配置描述"
            rows={3}
          />
        </div>
      </Modal>

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
  )
}

export { AnalysisResultDisplay }
export default DeploymentAnalysisPanel
