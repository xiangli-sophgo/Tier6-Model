import React from 'react'
import {
  GlobalSwitchConfig,
  ManualConnectionConfig,
  ConnectionMode,
  HierarchyLevel,
  LayoutType,
  HierarchicalTopology,
} from '../../types'
import { TopologyTrafficResult, PlanAnalysisResult, HardwareConfig, LLMModelConfig, InferenceConfig, ParallelismStrategy } from '../../utils/llmDeployment/types'
import { InfeasibleResult } from '../../utils/llmDeployment'

// 历史记录项
export interface AnalysisHistoryItem {
  id: string
  timestamp: number
  modelName: string
  parallelism: ParallelismStrategy
  score: number
  ttft: number
  tpot: number
  tps: number  // 集群总吞吐 (tokens/s)
  mfu: number
  mbu: number
  cost: number | null
  chips: number
  result: PlanAnalysisResult
  topKPlans?: PlanAnalysisResult[]
  searchMode?: 'manual' | 'auto'
  modelConfig: LLMModelConfig
  inferenceConfig: InferenceConfig
  hardwareConfig: HardwareConfig
}

// 子任务进度（自动搜索模式）
export interface SubTaskProgress {
  candidateIndex: number                        // 候选方案索引
  parallelism: ParallelismStrategy              // 并行策略
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number                              // 0-100
  chips?: number                                // 使用的芯片数
}

// 分析任务（轻量级，用于任务列表）
export interface AnalysisTask {
  id: string                                    // 唯一ID (uuid，与后端 task_id 一致)
  status: 'running' | 'completed' | 'failed' | 'cancelled'
  startTime: number                             // 开始时间戳
  endTime?: number                              // 结束时间戳

  // 实验信息
  experimentName?: string                       // 实验名称（用于跳转到结果管理页面）

  // 配置摘要
  modelName: string
  benchmarkName?: string                        // Benchmark描述 (如: "B=64, Seq=1024/4096")
  parallelism: ParallelismStrategy
  mode: 'manual' | 'auto'
  chips: number

  // 结果摘要（完成时填充）
  score?: number
  ttft?: number                                 // ms
  tpot?: number                                 // ms/token
  tps?: number                                  // 集群总吞吐 (tokens/s)
  mfu?: number                                  // Model FLOPs Utilization (0-1)
  mbu?: number                                  // Memory Bandwidth Utilization (0-1)
  error?: string                                // 失败原因

  // 进度（自动搜索模式）
  progress?: { current: number; total: number }

  // 子任务进度（自动搜索模式的多个候选方案）
  subTasks?: SubTaskProgress[]

  // 关联的历史记录ID（完成后自动保存到历史记录）
  historyId?: string
}

// 视图模式：历史列表（第一层） 或 结果详情（第二层）
export type AnalysisViewMode = 'history' | 'detail'

// 部署分析数据（用于传递给右侧分析面板）
export interface DeploymentAnalysisData {
  result: PlanAnalysisResult | null
  topKPlans: PlanAnalysisResult[]
  /** 不可行方案列表 */
  infeasiblePlans?: InfeasibleResult[]
  hardware: HardwareConfig
  model: LLMModelConfig
  inference?: InferenceConfig
  loading: boolean
  errorMsg: string | null
  searchStats: { evaluated: number; feasible: number; timeMs: number } | null
  searchProgress?: {
    stage: 'idle' | 'generating' | 'evaluating' | 'completed' | 'cancelled'
    totalCandidates: number
    currentEvaluating: number
    evaluated: number
  }
  /** 取消评估的回调 */
  onCancelEvaluation?: () => void
  onSelectPlan: (plan: PlanAnalysisResult) => void
  onMapToTopology?: () => void
  onClearTraffic?: () => void
  canMapToTopology?: boolean
  onSwitchToAnalysis?: () => void
  // 视图模式：历史列表 或 详情
  viewMode: AnalysisViewMode
  onViewModeChange: (mode: AnalysisViewMode) => void
  // 历史记录相关
  history: AnalysisHistoryItem[]
  onLoadFromHistory: (item: AnalysisHistoryItem) => void
  onDeleteHistory: (id: string) => void
  onClearHistory: () => void
}

// ============================================
// 自定义图标
// ============================================

// 自定义芯片图标 - 带引脚的芯片，中心白色
export const ChipIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg viewBox="0 0 100 100" width="1em" height="1em" fill="currentColor" className={className}>
    {/* 芯片主体 */}
    <rect x="20" y="20" width="60" height="60" rx="4" fill="currentColor"/>
    {/* 中心白色区域 */}
    <rect x="30" y="30" width="40" height="40" rx="2" fill="white"/>
    {/* 上方引脚 */}
    <rect x="28" y="8" width="8" height="12" fill="currentColor"/>
    <rect x="46" y="8" width="8" height="12" fill="currentColor"/>
    <rect x="64" y="8" width="8" height="12" fill="currentColor"/>
    {/* 下方引脚 */}
    <rect x="28" y="80" width="8" height="12" fill="currentColor"/>
    <rect x="46" y="80" width="8" height="12" fill="currentColor"/>
    <rect x="64" y="80" width="8" height="12" fill="currentColor"/>
    {/* 左侧引脚 */}
    <rect x="8" y="28" width="12" height="8" fill="currentColor"/>
    <rect x="8" y="46" width="12" height="8" fill="currentColor"/>
    <rect x="8" y="64" width="12" height="8" fill="currentColor"/>
    {/* 右侧引脚 */}
    <rect x="80" y="28" width="12" height="8" fill="currentColor"/>
    <rect x="80" y="46" width="12" height="8" fill="currentColor"/>
    <rect x="80" y="64" width="12" height="8" fill="currentColor"/>
  </svg>
)

// 自定义PCB板卡图标 - 带芯片和电路线（线框风格）
export const BoardIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg viewBox="0 0 100 80" width="1em" height="1em" fill="currentColor" className={className}>
    {/* PCB主体边框 */}
    <rect x="5" y="5" width="90" height="70" rx="3" fill="none" stroke="currentColor" strokeWidth="3"/>
    {/* 芯片1 - 左上 */}
    <rect x="12" y="12" width="18" height="18" rx="2" fill="currentColor"/>
    {/* 芯片2 - 右上 */}
    <rect x="70" y="12" width="18" height="18" rx="2" fill="currentColor"/>
    {/* 芯片3 - 中下 */}
    <rect x="38" y="45" width="24" height="20" rx="2" fill="currentColor"/>
    {/* 电路线 */}
    <path d="M30 21 L70 21" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M12 40 L38 40 L38 50" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M62 55 L88 55 L88 30" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M21 30 L21 45 L38 45" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M79 30 L79 40 L62 40 L62 45" stroke="currentColor" strokeWidth="2" fill="none"/>
    {/* 连接点 */}
    <circle cx="10" cy="40" r="3" fill="currentColor"/>
    <circle cx="90" cy="40" r="3" fill="currentColor"/>
    <circle cx="50" cy="10" r="3" fill="currentColor"/>
    <circle cx="50" cy="70" r="3" fill="currentColor"/>
  </svg>
)

// ============================================
// 类型定义
// ============================================

export interface ChipCounts {
  npu: number
  cpu: number
}

// 已废弃：BoardTypeConfig 和 BoardConfigs（保留用于类型兼容）
export interface BoardTypeConfig {
  count: number
  chips: ChipCounts
}

// 灵活板卡配置
export interface FlexBoardChipConfig {
  name: string                      // Chip名称/型号
  count: number                     // 数量
  preset_id?: string                // 预设ID (如 'h100-sxm')，为空表示自定义
}

export interface FlexBoardConfig {
  id: string            // 唯一ID
  name: string          // 板卡名称
  u_height: number      // U高度 (1-10)
  count: number         // 板卡数量
  chips: FlexBoardChipConfig[]  // Chip配置列表
}

export interface RackConfig {
  total_u: number              // Rack总U数，默认42
  boards: FlexBoardConfig[]    // 板卡配置列表
}

// 互联参数配置
export interface InterconnectParams {
  bandwidth_gbps: number       // 带宽 (GB/s)
  latency_us: number           // 延迟 (us)
}

// 芯片硬件参数（完整配置，与 types.ts 中的 ChipHardwareConfig 对齐）
export interface ChipHardwareParams {
  name: string                 // 芯片名称/型号
  num_cores: number            // 核心数
  compute_tflops_fp8: number   // FP8 算力 (TFLOPS)
  compute_tflops_bf16: number  // BF16 算力 (TFLOPS)
  memory_capacity_gb: number   // 显存容量 (GB)
  memory_bandwidth_gbps: number // 显存带宽 (GB/s)
  memory_bandwidth_utilization: number // 带宽利用率 (0-1)
  lmem_capacity_mb: number     // 片上缓存容量 (MB)
  lmem_bandwidth_gbps: number  // 片上缓存带宽 (GB/s)
  cost_per_hour?: number       // 成本 ($/hour)
  // 微架构参数
  cube_m?: number              // 矩阵单元 M 维度
  cube_k?: number              // 矩阵单元 K 维度
  cube_n?: number              // 矩阵单元 N 维度
  sram_size_kb?: number        // 每核 SRAM 大小 (KB)
  sram_utilization?: number    // SRAM 可用比例 (0-1)
  lane_num?: number            // SIMD lane 数量
  align_bytes?: number         // 内存对齐字节数
  compute_dma_overlap_rate?: number // 计算-搬运重叠率 (0-1)
}

// 完整硬件配置（用于保存）
export interface HardwareParams {
  chip: ChipHardwareParams
  interconnect: {
    c2c: InterconnectParams    // Chip间（板内）
    b2b: InterconnectParams    // Board间（机架内）
    r2r: InterconnectParams    // Rack间（Pod内）
    p2p: InterconnectParams    // Pod间
  }
}

// Switch 3D显示配置
export interface SwitchDisplayConfig {
  position: 'top' | 'middle' | 'bottom'
  uHeight: number
}

export interface ConfigPanelProps {
  topology: HierarchicalTopology | null
  onGenerate: (config: {
    pod_count: number
    racks_per_pod: number
    rack_config?: RackConfig
    switch_config?: GlobalSwitchConfig
    manual_connections?: ManualConnectionConfig
    interconnect_config?: {
      c2c?: InterconnectParams  // Chip-to-Chip
      b2b?: InterconnectParams  // Board-to-Board
      r2r?: InterconnectParams  // Rack-to-Rack
      p2p?: InterconnectParams  // Pod-to-Pod
    }
  }) => void
  loading: boolean
  currentLevel?: 'datacenter' | 'pod' | 'rack' | 'board'
  // 芯片选择相关
  selectedChipId?: string  // 选中的芯片 ID（格式：boardId-chipIndex）
  onChipTabActivate?: () => void  // 切换到 Chip Tab 的回调
  // 编辑连接相关
  manualConnectionConfig?: ManualConnectionConfig
  onManualConnectionConfigChange?: (config: ManualConnectionConfig) => void
  connectionMode?: ConnectionMode
  onConnectionModeChange?: (mode: ConnectionMode) => void
  selectedNodes?: Set<string>
  onSelectedNodesChange?: (nodes: Set<string>) => void
  targetNodes?: Set<string>
  onTargetNodesChange?: (nodes: Set<string>) => void
  onBatchConnect?: (level: HierarchyLevel) => void
  onDeleteManualConnection?: (connectionId: string) => void
  currentViewConnections?: { source: string; target: string; type?: string; bandwidth?: number; latency?: number }[]  // 当前视图的连接
  onDeleteConnection?: (source: string, target: string) => void  // 删除连接
  onUpdateConnectionParams?: (source: string, target: string, bandwidth?: number, latency?: number) => void  // 更新连接参数
  // 布局相关
  layoutType?: LayoutType
  onLayoutTypeChange?: (type: LayoutType) => void
  viewMode?: '3d' | 'topology' | 'knowledge'
  // Switch 3D显示配置
  switchDisplayConfig?: SwitchDisplayConfig
  onSwitchDisplayConfigChange?: (config: SwitchDisplayConfig) => void
  // 外部控制聚焦的层级（点击容器时切换）
  focusedLevel?: 'datacenter' | 'pod' | 'rack' | 'board' | null
  // 流量热力图
  trafficResult?: TopologyTrafficResult | null
  onTrafficResultChange?: (result: TopologyTrafficResult | null) => void
  // 部署分析结果（用于右侧图表）
  onAnalysisDataChange?: (data: DeploymentAnalysisData | null) => void
  // 历史记录 (由 WorkbenchContext 统一管理)
  analysisHistory?: AnalysisHistoryItem[]
  onAddToHistory?: (item: Omit<AnalysisHistoryItem, 'id' | 'timestamp'>) => void
  onDeleteHistory?: (id: string) => void
  onClearHistory?: () => void
}

// ============================================
// 常量
// ============================================

// localStorage缓存key
export const CONFIG_CACHE_KEY = 'tier6_topology_config_cache'
export const ANALYSIS_TASKS_KEY = 'tier6_analysis_tasks'

// 默认Rack配置
export const DEFAULT_RACK_CONFIG: RackConfig = {
  total_u: 42,
  boards: [
    { id: 'board_1', name: 'Board', u_height: 2, count: 8, chips: [{ name: 'SG2262', count: 8, preset_id: 'sg2262' }] },
  ],
}

// 默认Switch配置
export const DEFAULT_SWITCH_CONFIG: GlobalSwitchConfig = {
  switch_types: [
    { id: 'switch_48', name: 'Switch', port_count: 48 },
  ],
  inter_pod: { enabled: false, layers: [], downlink_redundancy: 1, connect_to_upper_level: true, keep_direct_topology: false },
  inter_rack: { enabled: false, layers: [], downlink_redundancy: 1, connect_to_upper_level: true, keep_direct_topology: false },
  inter_board: { enabled: false, layers: [], downlink_redundancy: 1, connect_to_upper_level: true, switch_position: 'top', switch_u_height: 1, keep_direct_topology: false },
  inter_chip: { enabled: false, layers: [], downlink_redundancy: 1, connect_to_upper_level: true, keep_direct_topology: false },
}

// 默认芯片硬件参数
export const DEFAULT_CHIP_HARDWARE: ChipHardwareParams = {
  name: 'SG2262',
  num_cores: 64,
  compute_tflops_fp8: 256,
  compute_tflops_bf16: 128,
  memory_capacity_gb: 32,
  memory_bandwidth_gbps: 819,
  memory_bandwidth_utilization: 0.85,
  lmem_capacity_mb: 128,
  lmem_bandwidth_gbps: 12000,
  cube_m: 16,
  cube_k: 32,
  cube_n: 8,
  sram_size_kb: 2048,
  sram_utilization: 0.45,
  lane_num: 16,
  align_bytes: 32,
  compute_dma_overlap_rate: 0.8,
}

// 默认互联参数
export const DEFAULT_INTERCONNECT: HardwareParams['interconnect'] = {
  c2c: { bandwidth_gbps: 900, latency_us: 1.0 },
  b2b: { bandwidth_gbps: 450, latency_us: 0.35 },
  r2r: { bandwidth_gbps: 200, latency_us: 2.0 },
  p2p: { bandwidth_gbps: 100, latency_us: 5.0 },
}

// 默认完整硬件配置
export const DEFAULT_HARDWARE_PARAMS: HardwareParams = {
  chip: DEFAULT_CHIP_HARDWARE,
  interconnect: DEFAULT_INTERCONNECT,
}

// ============================================
// 工具函数
// ============================================

// 从localStorage加载缓存配置
export const loadCachedConfig = () => {
  try {
    const cached = localStorage.getItem(CONFIG_CACHE_KEY)
    if (cached) {
      return JSON.parse(cached)
    }
  } catch (error) {
    console.error('加载缓存配置失败:', error)
  }
  return null
}

// 保存配置到localStorage
export const saveCachedConfig = (config: {
  podCount: number
  racksPerPod: number
  rackConfig?: RackConfig
  switchConfig?: GlobalSwitchConfig
  manualConnectionConfig?: ManualConnectionConfig
  hardwareParams?: HardwareParams
}) => {
  try {
    localStorage.setItem(CONFIG_CACHE_KEY, JSON.stringify(config))
  } catch (error) {
    console.error('缓存配置失败:', error)
  }
}

// 从localStorage加载分析任务列表
export const loadAnalysisTasks = (): AnalysisTask[] => {
  try {
    const cached = localStorage.getItem(ANALYSIS_TASKS_KEY)
    if (cached) {
      return JSON.parse(cached)
    }
  } catch (error) {
    console.error('加载分析任务失败:', error)
  }
  return []
}

// 保存分析任务列表到localStorage
export const saveAnalysisTasks = (tasks: AnalysisTask[]) => {
  try {
    localStorage.setItem(ANALYSIS_TASKS_KEY, JSON.stringify(tasks))
  } catch (error) {
    console.error('保存分析任务失败:', error)
  }
}

// ============================================
// 统一的样式定义
// ============================================

/**
 * 配置行样式：用于表单字段的通用布局
 */
export const configRowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: 10,
}

// 生成唯一ID
export const generateTaskId = (): string => {
  return `task_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`
}
