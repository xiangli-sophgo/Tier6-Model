import Icon from '@ant-design/icons'
import {
  GlobalSwitchConfig,
  ManualConnectionConfig,
  ConnectionMode,
  HierarchyLevel,
  LayoutType,
  HierarchicalTopology,
} from '../../types'
import { TopologyTrafficResult, PlanAnalysisResult, HardwareConfig, LLMModelConfig, InferenceConfig, ParallelismStrategy } from '../../utils/llmDeployment/types'

// 历史记录项
export interface AnalysisHistoryItem {
  id: string
  timestamp: number
  modelName: string
  parallelism: ParallelismStrategy
  score: number
  ttft: number
  tpot: number
  throughput: number
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

// 视图模式：历史列表（第一层） 或 结果详情（第二层）
export type AnalysisViewMode = 'history' | 'detail'

// 部署分析数据（用于传递给右侧分析面板）
export interface DeploymentAnalysisData {
  result: PlanAnalysisResult | null
  topKPlans: PlanAnalysisResult[]
  hardware: HardwareConfig
  model: LLMModelConfig
  inference?: InferenceConfig
  loading: boolean
  errorMsg: string | null
  searchStats: { evaluated: number; feasible: number; timeMs: number } | null
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
const ChipSvg = () => (
  <svg viewBox="0 0 100 100" width="1em" height="1em" fill="currentColor">
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
export const ChipIcon = () => <Icon component={ChipSvg} />

// 自定义PCB板卡图标 - 带芯片和电路线（线框风格）
const BoardSvg = () => (
  <svg viewBox="0 0 100 80" width="1em" height="1em" fill="currentColor">
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
export const BoardIcon = () => <Icon component={BoardSvg} />

// ============================================
// 类型定义
// ============================================

export interface ChipCounts {
  npu: number
  cpu: number
}

export interface BoardTypeConfig {
  count: number
  chips: ChipCounts
}

export interface BoardConfigs {
  u1: BoardTypeConfig
  u2: BoardTypeConfig
  u4: BoardTypeConfig
}

// 新的灵活板卡配置
export interface FlexBoardChipConfig {
  name: string                      // Chip名称/型号
  count: number                     // 数量
  preset_id?: string                // 预设ID (如 'h100-sxm')，为空表示自定义
  // 自定义参数（preset_id为空时使用）
  compute_tflops_fp16?: number      // FP16算力 (TFLOPs)
  memory_gb?: number                // 显存容量 (GB)
  memory_bandwidth_gbps?: number    // 显存带宽 (GB/s)
  memory_bandwidth_utilization?: number  // 带宽利用率 (0-1)
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
    board_configs: BoardConfigs
    rack_config?: RackConfig
    switch_config?: GlobalSwitchConfig
    manual_connections?: ManualConnectionConfig
  }) => void
  loading: boolean
  currentLevel?: 'datacenter' | 'pod' | 'rack' | 'board'
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
  viewMode?: '3d' | 'topology' | 'analysis' | 'knowledge'
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

// 默认配置
export const DEFAULT_BOARD_CONFIGS: BoardConfigs = {
  u1: { count: 0, chips: { npu: 2, cpu: 0 } },
  u2: { count: 8, chips: { npu: 8, cpu: 0 } },
  u4: { count: 0, chips: { npu: 16, cpu: 2 } },
}

// 默认Rack配置
export const DEFAULT_RACK_CONFIG: RackConfig = {
  total_u: 42,
  boards: [
    { id: 'board_1', name: 'Board', u_height: 2, count: 8, chips: [{ name: 'H100-SXM', count: 8, preset_id: 'h100-sxm' }] },
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
  boardConfigs: BoardConfigs
  rackConfig?: RackConfig
  switchConfig?: GlobalSwitchConfig
  manualConnectionConfig?: ManualConnectionConfig
}) => {
  try {
    localStorage.setItem(CONFIG_CACHE_KEY, JSON.stringify(config))
  } catch (error) {
    console.error('缓存配置失败:', error)
  }
}
