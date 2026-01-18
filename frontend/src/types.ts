// ============================================
// 层级类型定义
// ============================================

// 视图层级
export type ViewLevel = 'pod' | 'rack' | 'board' | 'chip';

// Chip类型
export type ChipType = 'chip';

// ============================================
// 层级配置接口
// ============================================

// Chip配置
export interface ChipConfig {
  id: string;
  type: ChipType;
  position: [number, number];  // [行, 列]
  label?: string;
}

// Board配置
export interface BoardConfig {
  id: string;
  u_position: number;     // 起始U位 (1-42)
  u_height: number;       // 占用U数
  label: string;
  chips: ChipConfig[];
}

// Rack配置
export interface RackConfig {
  id: string;
  position: [number, number];  // 在Pod中的网格位置
  label: string;
  total_u: number;        // 总U数，默认42
  boards: BoardConfig[];
}

// Pod配置
export interface PodConfig {
  id: string;
  label: string;
  grid_size: [number, number];  // Rack排列网格 [行, 列]
  racks: RackConfig[];
}

// 连接配置
export interface ConnectionConfig {
  source: string;
  target: string;
  type: 'intra' | 'inter' | 'switch' | 'manual';
  bandwidth?: number;
  latency?: number;  // 延迟 (ns)
  connection_role?: 'uplink' | 'downlink' | 'inter';  // Switch连接角色
  is_manual?: boolean;  // 是否为手动添加的连接
}

// ============================================
// 手动连接配置接口
// ============================================

// 层级类型
export type HierarchyLevel = 'datacenter' | 'pod' | 'rack' | 'board';

// 手动连接
export interface ManualConnection {
  id: string;
  source: string;
  target: string;
  hierarchy_level: HierarchyLevel;
  bandwidth?: number;
  latency?: number;  // 延迟 (ns)
  description?: string;
  created_at?: string;
}

// 层级默认连接参数
export interface LevelConnectionDefaults {
  bandwidth?: number;  // 默认带宽 (GB/s)
  latency?: number;    // 默认延迟 (us)
}

// 手动连接配置
export interface ManualConnectionConfig {
  enabled: boolean;
  mode: 'append' | 'replace';
  connections: ManualConnection[];
  // 各层级默认连接参数
  level_defaults?: {
    datacenter?: LevelConnectionDefaults;
    pod?: LevelConnectionDefaults;
    rack?: LevelConnectionDefaults;
    board?: LevelConnectionDefaults;
  };
}

// 连接模式: view=查看, select_source=选择源节点, select_target=选择目标节点
export type ConnectionMode = 'view' | 'select' | 'connect' | 'select_source' | 'select_target';

// 布局类型（基础布局，不包含手动模式）
export type LayoutType = 'auto' | 'circle' | 'grid' | 'force';

// ============================================
// Switch配置接口
// ============================================

// Switch类型预定义
export interface SwitchTypeConfig {
  id: string;           // 类型标识，如 "leaf_72", "spine_512"
  name: string;         // 显示名称，如 "72端口Leaf交换机"
  port_count: number;   // 总端口数
}

// 单层Switch配置
export interface SwitchLayerConfig {
  layer_name: string;       // 层名称，如 "leaf", "spine"
  switch_type_id: string;   // 使用的Switch类型ID
  count: number;            // 该层Switch数量
  inter_connect: boolean;   // 同层Switch是否互联
  // Rack层级Switch的物理位置配置
  u_start_position?: number;  // 起始U位 (1-42)，多台Switch从此位置向上排列
  u_height?: number;          // 每台Switch占用U数，默认1
}

// 直连拓扑类型
export type DirectTopologyType = 'none' | 'full_mesh' | 'full_mesh_2d' | 'ring' | 'torus_2d' | 'torus_3d';

// Switch与下层节点的连接模式
export type SwitchConnectionMode = 'full_mesh' | 'custom';

// 自定义Switch连接
export interface SwitchCustomConnection {
  device_id: string;        // 设备ID
  switch_indices: number[]; // 连接到的Switch索引列表
}

// Switch位置类型
export type SwitchPosition = 'top' | 'middle' | 'bottom';

// 层级Switch配置（支持多层Switch，如Leaf-Spine）
export interface HierarchyLevelSwitchConfig {
  enabled: boolean;                     // 是否启用该层级的Switch
  layers: SwitchLayerConfig[];          // Switch层列表（从下到上）
  downlink_redundancy: number;          // 每节点连接数（自定义模式使用）
  connect_to_upper_level: boolean;      // 是否连接到上层的Switch
  direct_topology?: DirectTopologyType; // 无Switch时的直连拓扑类型
  keep_direct_topology?: boolean;       // 启用Switch时是否同时保留节点直连
  connection_mode?: SwitchConnectionMode;           // Switch与下层节点的连接模式
  custom_connections?: SwitchCustomConnection[];    // 自定义连接配置
  // Rack层级Switch的3D显示配置
  switch_position?: SwitchPosition;     // Switch位置: top/middle/bottom
  switch_u_height?: number;             // Switch U高度 (1-4U)
}

// 全局Switch配置
export interface GlobalSwitchConfig {
  switch_types: SwitchTypeConfig[];                    // 预定义的Switch类型
  inter_pod: HierarchyLevelSwitchConfig;               // Pod间交换机
  inter_rack: HierarchyLevelSwitchConfig;              // Rack间交换机
  inter_board: HierarchyLevelSwitchConfig;             // Board间交换机
  inter_chip: HierarchyLevelSwitchConfig;              // Chip间交换机
}

// Switch实例
export interface SwitchInstance {
  id: string;                                           // 唯一标识
  type_id: string;                                      // Switch类型ID
  layer: string;                                        // 所在层，如 "leaf", "spine"
  hierarchy_level: 'inter_pod' | 'inter_rack' | 'inter_board' | 'inter_chip';  // Switch层级
  parent_id?: string;                                   // 父节点ID (rack层级时为rack_id)
  label: string;                                        // 显示标签
  uplink_ports_used: number;                            // 上行端口使用数
  downlink_ports_used: number;                          // 下行端口使用数
  inter_ports_used: number;                             // 同层互联端口使用数
  u_height?: number;                                    // 占用U数（用于3D显示）
  u_position?: number;                                  // U位置（用于3D显示，与Board统一计算）
}

// 完整拓扑数据
export interface HierarchicalTopology {
  pods: PodConfig[];
  connections: ConnectionConfig[];
  switches: SwitchInstance[];                           // Switch实例列表
  switch_config?: GlobalSwitchConfig;                   // Switch配置
  manual_connections?: ManualConnectionConfig;          // 手动连接配置
}

// ============================================
// 视图状态
// ============================================

export interface ViewState {
  level: ViewLevel;
  path: string[];           // 当前路径 ['pod_0', 'rack_1', 'board_2']
  selectedNode?: string;
}

// 面包屑项
export interface BreadcrumbItem {
  level: ViewLevel;
  id: string;
  label: string;
}

// ============================================
// 常量定义
// ============================================

// 层级显示名称
export const LEVEL_NAMES: Record<ViewLevel, string> = {
  pod: 'Pod (机柜组)',
  rack: 'Rack (机柜)',
  board: 'Board (板卡)',
  chip: 'Chip (芯片)',
};

// Chip类型显示名称
export const CHIP_TYPE_NAMES: Record<ChipType, string> = {
  chip: 'Chip',
};

// 层级颜色
export const LEVEL_COLORS: Record<ViewLevel, string> = {
  pod: '#fa8c16',      // 橙色
  rack: '#1890ff',     // 蓝色
  board: '#52c41a',    // 绿色
  chip: '#722ed1',     // 紫色
};

// Chip类型颜色
export const CHIP_TYPE_COLORS: Record<ChipType, string> = {
  chip: '#d97706',     // 琥珀色
};

// Switch层级颜色
export const SWITCH_LAYER_COLORS: Record<string, string> = {
  leaf: '#13c2c2',     // 青色
  spine: '#faad14',    // 金色
  core: '#f5222d',     // 红色
};

// Switch层级显示名称
export const SWITCH_LAYER_NAMES: Record<string, string> = {
  leaf: 'Leaf交换机',
  spine: 'Spine交换机',
  core: '核心交换机',
};

// ============================================
// 物理尺寸常量 (3D世界单位)
// ============================================

// Rack尺寸
export const RACK_DIMENSIONS = {
  width: 0.6,           // 19英寸标准宽度
  depth: 1.0,           // 深度
  uHeight: 0.0445,      // 单U高度 (1U ≈ 4.45cm)
  totalU: 42,           // 标准42U
  get fullHeight() { return this.totalU * this.uHeight; },
};

// Board尺寸
export const BOARD_DIMENSIONS = {
  width: 0.5,
  depth: 0.8,
  height: 0.04,
};

// Chip尺寸 [width(x), height(y-厚度), depth(z)]
export const CHIP_DIMENSIONS: Record<ChipType, [number, number, number]> = {
  chip: [0.07, 0.02, 0.07],
};

// 相机预设位置
export const CAMERA_PRESETS: Record<ViewLevel, [number, number, number]> = {
  pod: [5, 4, 5],
  rack: [2, 2, 3],
  board: [1, 0.8, 1],
  chip: [0.3, 0.25, 0.35],
};

// 相机距离限制
export const CAMERA_DISTANCE: Record<ViewLevel, { min: number; max: number }> = {
  pod: { min: 2, max: 30 },
  rack: { min: 0.5, max: 8 },
  board: { min: 0.3, max: 3 },
  chip: { min: 0.1, max: 1 },
};

// ============================================
// LOD (Level of Detail) 配置
// ============================================

// LOD 级别
export type LODLevel = 'high' | 'medium' | 'low';

// LOD 距离阈值（单位：3D世界单位）
export const LOD_THRESHOLDS = {
  high: 2,      // 距离 < 2: 高细节（完整引脚、电路纹理、文字）
  medium: 5,    // 距离 2-5: 中细节（简化引脚）
  low: Infinity // 距离 > 5: 低细节（仅 Box）
};

// 引脚渲染配置
export const PIN_CONFIG = {
  pinsPerSide: 6,           // 每边引脚数
  pinWidth: 0.006,          // 引脚宽度
  pinDepth: 0.004,          // 引脚深度
  pinHeightRatio: 0.3,      // 引脚高度相对于芯片厚度的比例
  pinColor: '#a0a0a0',      // 引脚颜色
  pinMetalness: 0.4,
  pinRoughness: 0.6,
};

// 电路纹理配置
export const CIRCUIT_TRACE_CONFIG = {
  horizontalCount: 3,       // 水平纹理数量
  verticalCount: 3,         // 垂直纹理数量
  traceHeight: 0.001,       // 纹理高度
  traceWidth: 0.002,        // 纹理宽度
  traceColor: '#333',       // 纹理颜色
};

// ============================================
// 键盘快捷键配置
// ============================================

export const KEYBOARD_SHORTCUTS = {
  back: ['Escape', 'Backspace'],      // 返回上一级
  resetView: ['KeyR'],                 // 重置相机视角
};

// ============================================
// 多层级视图配置
// ============================================

// 相邻层级组合
export type AdjacentLevelPair = 'datacenter_pod' | 'pod_rack' | 'rack_board' | 'board_chip';

// 多层级视图选项
export interface MultiLevelViewOptions {
  enabled: boolean;
  levelPair: AdjacentLevelPair;
  expandedContainers: Set<string>;  // 展开的容器节点ID
}

// 层级组合显示名称
export const LEVEL_PAIR_NAMES: Record<AdjacentLevelPair, string> = {
  datacenter_pod: 'Datacenter + Pod',
  pod_rack: 'Pod + Rack',
  rack_board: 'Rack + Board',
  board_chip: 'Board + Chip',
};

