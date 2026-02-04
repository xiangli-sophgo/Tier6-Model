/**
 * LLM 部署分析面板
 *
 * 提供模型配置、推理配置、硬件配置、并行策略配置和分析结果展示
 */

import React, { useState, useCallback, useRef, useMemo } from 'react'
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
  HardwareParams,
  DEFAULT_HARDWARE_PARAMS,
  DEFAULT_CHIP_HARDWARE,
} from '../shared'
import { AnalysisTaskList } from './AnalysisTaskList'
import { listConfigs, getConfig, saveConfig, SavedConfig } from '../../../api/topology'
import {
  BenchmarkConfigSelector,
  colors,
} from './ConfigSelectors'
import {
  generateBenchmarkName,
  generateTopologyName,
} from '../../../utils/configNameGenerator'
import { BaseCard } from '@/components/common/BaseCard'
import { ParallelismConfigPanel } from './ParallelismConfigPanel'
import { AnalysisResultDisplay } from './AnalysisResultDisplay'
import { useTaskWebSocket, TaskUpdate } from '../../../hooks/useTaskWebSocket'
import { TopologyInfoCard } from './TopologyInfoCard'
import { SweepConfigPanel } from './ParameterSweep/SweepConfigPanel'
import { extractSweepableParameters } from './ParameterSweep/parameterExtractors'
import {
  generateCombinationsWithBinding,
  applyParameterCombination
} from './ParameterSweep/sweepHelpers'
import type { SweepParam } from './ParameterSweep/sweepTypes'
// createBenchmark 不再需要，Sweep 模式直接传递完整配置而不创建临时文件


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

  // 本地硬件参数（多芯片独立配置 - 可编辑）
  const [localHardwareParams, setLocalHardwareParams] = useState<HardwareParams | null>(null)

  // 实验名称（用户自定义，留空则使用 Benchmark 名称）
  const [experimentName, setExperimentName] = useState<string>('')

  // 实验描述（用户自定义，留空则自动生成）
  const [experimentDescription, setExperimentDescription] = useState<string>('')

  // 任务并发数（本次评估使用的 worker 数量）
  const [taskMaxWorkers, setTaskMaxWorkers] = useState<number>(4)

  // Tile 搜索开关（默认开启）
  const [enableTileSearch, setEnableTileSearch] = useState<boolean>(true)

  // 分区搜索开关（默认开启）
  const [enablePartitionSearch, setEnablePartitionSearch] = useState<boolean>(true)

  // 最大模拟 token 数（默认 4）
  const [maxSimulatedTokens, setMaxSimulatedTokens] = useState<number>(4)

  // 原始配置快照（用于修改追踪）
  const [originalBenchmarkConfig, setOriginalBenchmarkConfig] = useState<{
    model: LLMModelConfig | null
    inference: InferenceConfig | null
  }>({ model: null, inference: null })

  const [originalTopologyConfig, setOriginalTopologyConfig] = useState<{
    hardwareParams: HardwareParams | null
    commLatency: CommLatencyConfig | null
  }>({ hardwareParams: null, commLatency: null })

  // 加载拓扑配置列表，并自动选择第一个配置
  React.useEffect(() => {
    const loadTopologyConfigs = async () => {
      try {
        const configs = await listConfigs()
        setTopologyConfigs(configs)

        // 自动加载第一个配置
        if (configs.length > 0 && !selectedTopologyConfig) {
          const firstConfigName = configs[0].name
          setSelectedTopologyConfig(firstConfigName)

          // 获取完整配置并应用
          try {
            const fullConfig = await getConfig(firstConfigName)
            if (fullConfig) {
              if (fullConfig.rack_config) {
                setLocalRackConfig(fullConfig.rack_config as RackConfig)
              }
              setLocalPodCount(fullConfig.pod_count || 1)
              setLocalRacksPerPod(fullConfig.racks_per_pod || 1)
              if (fullConfig.comm_latency_config) {
                setCommLatencyConfig(fullConfig.comm_latency_config)
              }
              // 恢复硬件参数
              if (fullConfig.hardware_params) {
                const hw = fullConfig.hardware_params as any
                if (hw.chips) {
                  setLocalHardwareParams({
                    chips: { ...DEFAULT_HARDWARE_PARAMS.chips, ...hw.chips },
                    interconnect: {
                      c2c: { ...DEFAULT_HARDWARE_PARAMS.interconnect.c2c, ...hw.interconnect?.c2c },
                      b2b: { ...DEFAULT_HARDWARE_PARAMS.interconnect.b2b, ...hw.interconnect?.b2b },
                      r2r: { ...DEFAULT_HARDWARE_PARAMS.interconnect.r2r, ...hw.interconnect?.r2r },
                      p2p: { ...DEFAULT_HARDWARE_PARAMS.interconnect.p2p, ...hw.interconnect?.p2p },
                    },
                  })
                } else if (hw.chip) {
                  const chipName = hw.chip.name || 'SG2262'
                  setLocalHardwareParams({
                    chips: { [chipName]: { ...DEFAULT_CHIP_HARDWARE, ...hw.chip } },
                    interconnect: {
                      c2c: { ...DEFAULT_HARDWARE_PARAMS.interconnect.c2c, ...hw.interconnect?.c2c },
                      b2b: { ...DEFAULT_HARDWARE_PARAMS.interconnect.b2b, ...hw.interconnect?.b2b },
                      r2r: { ...DEFAULT_HARDWARE_PARAMS.interconnect.r2r, ...hw.interconnect?.r2r },
                      p2p: { ...DEFAULT_HARDWARE_PARAMS.interconnect.p2p, ...hw.interconnect?.p2p },
                    },
                  })
                }
              }
            }
          } catch (error) {
            console.error('加载默认配置失败:', error)
          }
        }
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
  const handleSelectTopologyConfig = useCallback(async (configName: string | undefined) => {
    setSelectedTopologyConfig(configName)
    if (!configName) {
      // 清除选择，使用 props 传入的配置
      setLocalRackConfig(rackConfig)
      setLocalPodCount(podCount)
      setLocalRacksPerPod(racksPerPod)
      // 重置延迟设置为默认值
      setCommLatencyConfig({ ...DEFAULT_COMM_LATENCY_CONFIG })
      // 重置硬件参数
      setLocalHardwareParams(null)
      return
    }

    // 从后端获取完整配置（列表API只返回摘要信息）
    try {
      const config = await getConfig(configName)
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
        // 恢复硬件参数（支持新格式 chips 和旧格式 chip）
        if (config.hardware_params) {
          const hw = config.hardware_params as any
          if (hw.chips) {
            // 新格式
            setLocalHardwareParams({
              chips: { ...DEFAULT_HARDWARE_PARAMS.chips, ...hw.chips },
              interconnect: {
                c2c: { ...DEFAULT_HARDWARE_PARAMS.interconnect.c2c, ...hw.interconnect?.c2c },
                b2b: { ...DEFAULT_HARDWARE_PARAMS.interconnect.b2b, ...hw.interconnect?.b2b },
                r2r: { ...DEFAULT_HARDWARE_PARAMS.interconnect.r2r, ...hw.interconnect?.r2r },
                p2p: { ...DEFAULT_HARDWARE_PARAMS.interconnect.p2p, ...hw.interconnect?.p2p },
              },
            })
          } else if (hw.chip) {
            // 旧格式兼容
            const chipName = hw.chip.name || 'SG2262'
            setLocalHardwareParams({
              chips: { [chipName]: { ...DEFAULT_CHIP_HARDWARE, ...hw.chip } },
              interconnect: {
                c2c: { ...DEFAULT_HARDWARE_PARAMS.interconnect.c2c, ...hw.interconnect?.c2c },
                b2b: { ...DEFAULT_HARDWARE_PARAMS.interconnect.b2b, ...hw.interconnect?.b2b },
                r2r: { ...DEFAULT_HARDWARE_PARAMS.interconnect.r2r, ...hw.interconnect?.r2r },
                p2p: { ...DEFAULT_HARDWARE_PARAMS.interconnect.p2p, ...hw.interconnect?.p2p },
              },
            })
          }
        } else {
          setLocalHardwareParams(null)
        }

        // 保存原始拓扑配置快照（用于修改追踪）
        const hwParams = config.hardware_params as any
        const originalHwParams: HardwareParams | null = hwParams ? {
          chips: hwParams.chips ? { ...hwParams.chips } : (hwParams.chip ? { [hwParams.chip.name || 'SG2262']: { ...hwParams.chip } } : {}),
          interconnect: {
            c2c: { ...DEFAULT_HARDWARE_PARAMS.interconnect.c2c, ...hwParams.interconnect?.c2c },
            b2b: { ...DEFAULT_HARDWARE_PARAMS.interconnect.b2b, ...hwParams.interconnect?.b2b },
            r2r: { ...DEFAULT_HARDWARE_PARAMS.interconnect.r2r, ...hwParams.interconnect?.r2r },
            p2p: { ...DEFAULT_HARDWARE_PARAMS.interconnect.p2p, ...hwParams.interconnect?.p2p },
          },
        } : null

        setOriginalTopologyConfig({
          hardwareParams: originalHwParams,
          commLatency: config.comm_latency_config ? { ...config.comm_latency_config } : null,
        })

        toast.success(`已加载拓扑配置: ${config.name}`)
      }
    } catch (error) {
      console.error('获取拓扑配置失败:', error)
      toast.error('加载配置失败')
    }
  }, [rackConfig, podCount, racksPerPod])

  // 从拓扑配置中提取硬件配置（不依赖后端）
  React.useEffect(() => {
    // 从拓扑配置提取硬件参数
    if (localRackConfig && localRackConfig.boards.length > 0 && topology?.connections) {
      const groups = extractChipGroupsFromConfig(localRackConfig.boards, localHardwareParams || undefined)
      if (groups.length > 0) {
        const firstChipType = groups[0].presetId || groups[0].chipType
        const config = generateHardwareConfigFromPanelConfig(
          localPodCount,
          localRacksPerPod,
          localRackConfig.boards.map(b => ({ chips: b.chips, count: b.count })),
          topology.connections,
          firstChipType,
          localHardwareParams || undefined
        )
        if (config) {
          setHardwareConfig(config)
        }
      }
    } else if (!hardwareConfig) {
      // 如果没有拓扑配置，使用默认值（SG2260E 参数）
      console.warn('未找到拓扑配置，使用默认硬件配置')
      const defaultConfig: HardwareConfig = {
        hardware_params: { chips: {}, interconnect: { c2c: { bandwidth_gbps: 0, latency_us: 0 }, b2b: { bandwidth_gbps: 0, latency_us: 0 }, r2r: { bandwidth_gbps: 0, latency_us: 0 }, p2p: { bandwidth_gbps: 0, latency_us: 0 } } },
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
  }, [localRackConfig, topology, localPodCount, localRacksPerPod, localHardwareParams])

  // 生成 rack 配置的稳定 key（用于比较是否变化，避免 JSON.stringify 的性能开销）
  const rackConfigKey = useMemo(() => {
    if (!localRackConfig) return ''
    // 只比较关键字段：boards 数量、每个 board 的芯片配置
    return localRackConfig.boards.map(b =>
      `${b.count}:${b.chips.map(c => `${c.name}x${c.count}`).join(',')}`
    ).join('|')
  }, [localRackConfig])

  // 当拓扑配置变化时，提取芯片组信息并更新硬件配置
  React.useEffect(() => {
    if (!localRackConfig || localRackConfig.boards.length === 0) {
      setChipGroups([])
      return
    }

    // 如果有保存的硬件参数，优先使用保存的芯片配置
    let groups: ChipGroupInfo[]
    if (localHardwareParams?.chips && Object.keys(localHardwareParams.chips).length > 0) {
      // 从保存的 hardware_params.chips 构建 chipGroups
      groups = Object.entries(localHardwareParams.chips).map(([chipName, chipConfig]) => ({
        chipType: chipName,
        presetId: chipConfig.name === chipName ? undefined : chipConfig.name,
        chipConfig: chipConfig,
        totalCount: 0, // 这里不需要准确的数量，只用于显示配置
        boardCount: 0,
        chipsPerBoard: 0,
      }))
    } else {
      // 否则从 rack_config.boards 提取
      groups = extractChipGroupsFromConfig(localRackConfig.boards, localHardwareParams || undefined)
    }
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
        currentSelectedType,
        localHardwareParams || undefined
      )
      if (config) {
        setHardwareConfig(config)
      }
    }
  }, [rackConfigKey, localPodCount, localRacksPerPod, topology?.connections, localHardwareParams])

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
        selectedChipType,
        localHardwareParams || undefined
      )
      if (config) {
        setHardwareConfig(config)
      }
    }
  }, [selectedChipType, localRackConfig, chipGroups, localPodCount, localRacksPerPod, topology?.connections, localHardwareParams])

  // 并行策略状态
  const [parallelismMode, setParallelismMode] = useState<'manual' | 'auto' | 'sweep'>('manual')
  const [manualStrategy, setManualStrategy] = useState<ParallelismStrategy>({
    dp: 1, tp: 1, pp: 1, ep: 1, sp: 1, moe_tp: 1,
  })

  // 参数遍历状态
  const [sweepParams, setSweepParams] = useState<SweepParam[]>([])

  // 计算最大可用芯片数（从拓扑配置中提取实际芯片总数）
  const maxChips = React.useMemo(() => {
    if (!topology) return 0
    const summary = extractHardwareSummary(topology)
    return summary.totalChips
  }, [topology])

  // 当模型配置或硬件配置变化时，更新手动策略为满足约束的默认值
  React.useEffect(() => {
    if (!hardwareConfig) return

    const isMoE = modelConfig.model_type === 'moe' && modelConfig.moe_config
    const maxTP = Math.min(128, modelConfig.num_attention_heads, maxChips)

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
  }, [modelConfig, hardwareConfig, maxChips])

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

  // 获取选中的拓扑配置对象（用于参数提取）
  const selectedTopologyConfigObj = React.useMemo(() => {
    return selectedTopologyConfig
      ? topologyConfigs.find(c => c.name === selectedTopologyConfig)
      : null
  }, [selectedTopologyConfig, topologyConfigs])

  // 可遍历参数列表（动态计算 - 现在包含拓扑配置参数）
  const sweepableParams = React.useMemo(
    () => extractSweepableParameters(
      modelConfig,
      inferenceConfig,
      localHardwareParams,
      manualStrategy,
      selectedTopologyConfigObj
    ),
    [modelConfig, inferenceConfig, localHardwareParams, manualStrategy, selectedTopologyConfigObj]
  )

  // 总组合数（参数遍历）
  const totalCombinations = React.useMemo(() => {
    if (sweepParams.length === 0) return 0
    let total = 1
    for (const param of sweepParams) {
      total *= param.values.length
    }
    return total
  }, [sweepParams])

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
        // 保存硬件参数（如果有本地修改）
        hardware_params: localHardwareParams ? {
          chips: localHardwareParams.chips,
          interconnect: localHardwareParams.interconnect,
        } as any : undefined,
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
  }, [newConfigName, newConfigDesc, topologyConfigs, selectedTopologyConfig, localPodCount, localRacksPerPod, localRackConfig, commLatencyConfig, localHardwareParams, refreshTopologyConfigs])

  // 保存拓扑配置（更新现有配置）
  const handleSaveTopologyConfig = useCallback(async () => {
    if (!selectedTopologyConfig) {
      toast.warning('请先选择一个配置文件')
      return
    }
    try {
      const updatedConfig: SavedConfig = {
        name: selectedTopologyConfig,
        description: topologyConfigs.find(c => c.name === selectedTopologyConfig)?.description,
        pod_count: localPodCount,
        racks_per_pod: localRacksPerPod,
        rack_config: localRackConfig ? {
          total_u: localRackConfig.total_u,
          boards: localRackConfig.boards,
        } : undefined,
        comm_latency_config: { ...commLatencyConfig },
        hardware_params: localHardwareParams ? {
          chips: localHardwareParams.chips,
          interconnect: localHardwareParams.interconnect,
        } as any : undefined,
      }
      await saveConfig(updatedConfig)
      await refreshTopologyConfigs()
      toast.success(`已保存配置: ${selectedTopologyConfig}`)
    } catch (error) {
      console.error('保存配置失败:', error)
      toast.error('保存配置失败')
    }
  }, [selectedTopologyConfig, topologyConfigs, localPodCount, localRacksPerPod, localRackConfig, commLatencyConfig, localHardwareParams, refreshTopologyConfigs])

  // 拓扑配置另存为
  const handleSaveAsTopologyConfig = useCallback(async (name: string, description?: string) => {
    try {
      const newConfig: SavedConfig = {
        name,
        description,
        pod_count: localPodCount,
        racks_per_pod: localRacksPerPod,
        rack_config: localRackConfig ? {
          total_u: localRackConfig.total_u,
          boards: localRackConfig.boards,
        } : undefined,
        comm_latency_config: { ...commLatencyConfig },
        hardware_params: localHardwareParams ? {
          chips: localHardwareParams.chips,
          interconnect: localHardwareParams.interconnect,
        } as any : undefined,
      }
      await saveConfig(newConfig)
      await refreshTopologyConfigs()
      setSelectedTopologyConfig(name)
      toast.success(`已创建新配置: ${name}`)
    } catch (error) {
      console.error('另存为配置失败:', error)
      toast.error('另存为配置失败')
      throw error
    }
  }, [localPodCount, localRacksPerPod, localRackConfig, commLatencyConfig, localHardwareParams, refreshTopologyConfigs])

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

    // 基于当前配置内容生成名称
    const benchmarkName = generateBenchmarkName(modelConfig, inferenceConfig)
    const topologyName = generateTopologyName({
      pod_count: localPodCount,
      racks_per_pod: localRacksPerPod,
      rack_config: localRackConfig,
    })

    // 使用用户输入的实验名称，如果为空则使用生成的 Benchmark 名称
    const finalExperimentName = experimentName.trim() || benchmarkName

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
      benchmarkName: benchmarkName,
      parallelism: strategy,
      mode: parallelismMode,
      chips: actualChips,
    }
    addTask(newTask)

    try {
      // 提交任务：传递完整配置内容，不再依赖后端从文件加载
      const response = await submitEvaluation({
        experiment_name: finalExperimentName,
        description: experimentDescription.trim() || benchmarkName,
        experiment_description: experimentDescription.trim() || undefined,

        // 配置来源标记
        benchmark_name: benchmarkName,
        topology_config_name: topologyName,

        // 完整配置内容
        benchmark_config: {
          model: modelConfig as unknown as Record<string, unknown>,
          inference: inferenceConfig as unknown as Record<string, unknown>,
        },
        topology_config: {
          name: topologyName,
          pod_count: localPodCount,
          racks_per_pod: localRacksPerPod,
          rack_config: localRackConfig,
          hardware_params: {
            chips: localHardwareParams?.chips || {},
            interconnect: localHardwareParams?.interconnect || {},
          },
          comm_latency_config: commLatencyConfig,
        },

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
    experimentName, taskMaxWorkers, modelConfig, inferenceConfig, parallelismMode, manualStrategy,
    maxChips, addTask, updateTask, setAnalysisTasks, enableTileSearch,
    enablePartitionSearch, maxSimulatedTokens, selectedTopologyConfig, selectedBenchmark,
    localPodCount, localRacksPerPod, localRackConfig, localHardwareParams, commLatencyConfig
  ])

  // 运行参数遍历（批量提交任务）
  const handleRunSweep = useCallback(async () => {
    // 验证：必须选择配置文件
    if (!selectedTopologyConfig) {
      toast.error('请先选择拓扑配置文件')
      return
    }
    if (!selectedBenchmark) {
      toast.error('请先选择 Benchmark 配置')
      return
    }
    if (sweepParams.length === 0) {
      toast.error('请至少添加一个遍历参数')
      return
    }

    // 生成参数组合（支持绑定）
    const combinations = generateCombinationsWithBinding(sweepParams)
    if (combinations.length === 0) {
      toast.error('没有生成任何参数组合')
      return
    }
    if (combinations.length > 500) {
      toast.error(`组合数过多 (${combinations.length})，请减少参数或值的数量（最大500）`)
      return
    }

    // 基础拓扑名称（不随 sweep 参数变化）
    const baseTopologyName = generateTopologyName({
      pod_count: localPodCount,
      racks_per_pod: localRacksPerPod,
      rack_config: localRackConfig,
    })
    const baseExperimentName = experimentName.trim() || generateBenchmarkName(modelConfig, inferenceConfig)

    // 生成实验级别的描述：优先使用用户输入，否则自动生成
    const autoGeneratedDesc = (() => {
      const paramDescs = sweepParams.map(param => {
        const values = param.values.map(v => String(v)).join(', ')
        return `${param.key}=[${values}]`
      })
      return `参数扫描: ${paramDescs.join(', ')} (${combinations.length}组)`
    })()
    const finalExperimentDesc = experimentDescription.trim() || autoGeneratedDesc

    toast.info(`开始批量提交 ${combinations.length} 个任务...`)

    // 批量提交任务
    for (const [idx, combo] of combinations.entries()) {
      try {
        // 1. 应用参数覆盖，生成新的配置对象
        const overriddenConfig = applyParameterCombination(
          {
            model: modelConfig,
            inference: inferenceConfig,
            hardware: localHardwareParams || { chips: {}, interconnect: { c2c: { bandwidth_gbps: 0, latency_us: 0 }, b2b: { bandwidth_gbps: 0, latency_us: 0 }, r2r: { bandwidth_gbps: 0, latency_us: 0 }, p2p: { bandwidth_gbps: 0, latency_us: 0 } } },
            parallelism: manualStrategy,
          },
          combo
        )

        // 2. 基于变体配置生成名称（不再创建临时文件）
        const variantBenchmarkName = generateBenchmarkName(
          overriddenConfig.model as unknown as LLMModelConfig,
          overriddenConfig.inference as unknown as InferenceConfig
        )

        // 3. 提交任务（传递完整配置，不创建临时文件）
        // 所有 sweep 任务使用相同的实验名称，便于参数分析
        const comboDesc = Object.entries(combo)
          .map(([key, value]) => `${key}=${value}`)
          .join(', ')
        const sweepExperimentName = `${baseExperimentName}_sweep`

        // 先创建本地任务记录
        const tempTaskId = `temp-${Date.now()}-${idx}`
        const newTask: AnalysisTask = {
          id: tempTaskId,
          status: 'running',
          startTime: Date.now(),
          experimentName: sweepExperimentName,
          modelName: (overriddenConfig.model as any).model_name || modelConfig.model_name,
          benchmarkName: variantBenchmarkName,
          parallelism: overriddenConfig.parallelism,
          mode: 'manual',
          chips: overriddenConfig.parallelism.dp * overriddenConfig.parallelism.tp,
        }
        addTask(newTask)

        const response = await submitEvaluation({
          experiment_name: sweepExperimentName,
          // 第一个任务使用实验描述，后续任务使用任务描述
          description: idx === 0 ? finalExperimentDesc : `[${idx + 1}/${combinations.length}] ${comboDesc}`,
          experiment_description: finalExperimentDesc,  // 传递实验描述给后端

          // 配置来源标记
          benchmark_name: variantBenchmarkName,
          topology_config_name: baseTopologyName,

          // 完整配置内容（变体配置）
          benchmark_config: {
            model: overriddenConfig.model,
            inference: overriddenConfig.inference,
          },
          topology_config: {
            name: baseTopologyName,
            pod_count: localPodCount,
            racks_per_pod: localRacksPerPod,
            rack_config: localRackConfig,
            hardware_params: {
              chips: overriddenConfig.hardware?.chips || localHardwareParams?.chips || {},
              interconnect: overriddenConfig.hardware?.interconnect || localHardwareParams?.interconnect || {},
            },
            comm_latency_config: commLatencyConfig,
          },

          search_mode: 'manual',
          manual_parallelism: overriddenConfig.parallelism as unknown as Record<string, unknown>,
          max_workers: taskMaxWorkers,
          enable_tile_search: enableTileSearch,
          enable_partition_search: enablePartitionSearch,
          max_simulated_tokens: maxSimulatedTokens,
        })

        // 更新任务 ID 为后端返回的真实 ID
        setAnalysisTasks(prev => prev.map(t =>
          t.id === tempTaskId ? { ...t, id: response.task_id } : t
        ))

        toast.success(`已提交任务 ${idx + 1}/${combinations.length}`)
      } catch (error) {
        console.error(`提交任务 ${idx + 1} 失败:`, error)
        const msg = error instanceof Error ? error.message : '未知错误'
        toast.error(`任务 ${idx + 1} 提交失败: ${msg}`)
      }
    }

    toast.success(`批量提交完成！共 ${combinations.length} 个任务`)
  }, [
    selectedTopologyConfig,
    selectedBenchmark,
    sweepParams,
    experimentName,
    modelConfig,
    inferenceConfig,
    localHardwareParams,
    manualStrategy,
    taskMaxWorkers,
    enableTileSearch,
    enablePartitionSearch,
    maxSimulatedTokens,
    addTask,
    setAnalysisTasks,
    localPodCount,
    localRacksPerPod,
    localRackConfig,
    commLatencyConfig,
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
            <BaseCard title="Benchmark 设置" collapsible gradient>
              <BenchmarkConfigSelector
                modelConfig={modelConfig}
                onModelChange={setModelConfig}
                inferenceConfig={inferenceConfig}
                onInferenceChange={setInferenceConfig}
                onBenchmarkSelect={setSelectedBenchmark}
              />
            </BaseCard>

            {/* 部署策略卡片 */}
            <BaseCard collapsible title="部署策略" gradient className="mt-4">
              <ParallelismConfigPanel
                mode={parallelismMode}
                onModeChange={setParallelismMode}
                manualStrategy={manualStrategy}
                onManualStrategyChange={setManualStrategy}
                maxChips={maxChips}
                modelConfig={modelConfig}
                hardwareConfig={hardwareConfig}
              />

              {/* 参数遍历面板（仅在 sweep 模式下显示） */}
              {parallelismMode === 'sweep' && (
                <>
                  <div className="mt-4">
                    <SweepConfigPanel
                      sweepableParams={sweepableParams}
                      sweepParams={sweepParams}
                      onSweepParamsChange={setSweepParams}
                      benchmarkName={selectedBenchmark}
                      topologyName={selectedTopologyConfig}
                    />
                  </div>
                  {/* 分割线 */}
                  <div className="my-6 border-t border-gray-200" />
                </>
              )}

              {/* Tile 搜索、分区搜索、最大模拟 Token 数 - 横向排列 */}
              <div className="grid grid-cols-3 gap-4 mt-4">
                {/* Tile 搜索开关 */}
                <div>
                  <div className="mb-1.5 text-[13px] text-gray-600 flex items-center">
                    启用 Tile 搜索
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 ml-1 text-gray-400 cursor-help" />
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
                <div>
                  <div className="mb-1.5 text-[13px] text-gray-600 flex items-center">
                    启用分区搜索
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 ml-1 text-gray-400 cursor-help" />
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

                {/* 最大模拟 Token 数 */}
                <div>
                  <div className="mb-1.5 text-[13px] text-gray-600 flex items-center">
                    最大模拟 Token 数
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="h-3.5 w-3.5 ml-1 text-gray-400 cursor-help" />
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
                    className="w-full"
                  />
                </div>
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

              {/* 实验描述 */}
              <div className="mt-4">
                <div className="mb-1.5 text-[13px] text-gray-600">实验描述</div>
                <Input
                  placeholder="留空则自动生成（参数扫描会显示扫描范围）"
                  value={experimentDescription}
                  onChange={(e) => setExperimentDescription(e.target.value)}
                />
              </div>

              {/* 运行按钮 */}
              <Button
                onClick={
                  parallelismMode === 'sweep'
                    ? handleRunSweep
                    : handleRunAnalysis
                }
                disabled={parallelismMode === 'sweep' && (sweepParams.length === 0 || totalCombinations > 500)}
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
                ) : parallelismMode === 'manual' ? (
                  <>
                    <PlayCircle className="h-4 w-4 mr-2" />
                    运行分析
                  </>
                ) : (
                  <>
                    <PlayCircle className="h-4 w-4 mr-2" />
                    批量评估 ({totalCombinations} 组)
                  </>
                )}
              </Button>
            </BaseCard>
          </div>

          {/* 右列：拓扑配置（可编辑） */}
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
              interconnectParams={localHardwareParams?.interconnect || interconnectParams}
              commLatencyConfig={commLatencyConfig}
              // 可编辑模式 props
              hardwareParams={localHardwareParams || undefined}
              onHardwareParamsChange={setLocalHardwareParams}
              onCommLatencyChange={setCommLatencyConfig}
              onSaveConfig={handleSaveTopologyConfig}
              onSaveAsConfig={handleSaveAsTopologyConfig}
              allConfigs={topologyConfigs}
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
        <BaseCard collapsible title="分析任务" gradient className="mt-4">
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
