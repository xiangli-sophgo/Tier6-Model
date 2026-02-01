/**
 * 设计令牌 (Design Tokens)
 *
 * 统一管理所有颜色、间距、字体等设计元素
 * 避免硬编码，便于主题切换和维护
 */

// ============================================
// 基础色板 (Color Palette)
// ============================================

/**
 * 基础颜色定义
 * 来源于 Ant Design 色板
 */
export const PALETTE = {
  // 蓝色系
  blue: {
    main: '#1890ff',
    light: '#e6f7ff',
    dark: '#096dd9',
    shadow: 'rgba(24, 144, 255, 0.15)',
  },
  // 绿色系
  green: {
    main: '#52c41a',
    light: '#f6ffed',
    dark: '#389e0d',
  },
  // 橙色系
  orange: {
    main: '#fa8c16',
    light: '#fff7e6',
    dark: '#d46b08',
  },
  // 紫色系
  purple: {
    main: '#722ed1',
    light: '#f9f0ff',
    dark: '#531dab',
  },
  // 金色系
  gold: {
    main: '#faad14',
    light: '#fffbe6',
    dark: '#d48806',
  },
  // 红色系
  red: {
    main: '#f5222d',
    light: '#fff2f0',
    dark: '#cf1322',
  },
  // 品红色系
  magenta: {
    main: '#eb2f96',
    light: '#fff0f6',
    dark: '#c41d7f',
  },
  // 粉色系
  pink: {
    main: '#f759ab',
    light: '#fff0f6',
    dark: '#c41d7f',
  },
  // 青色系
  cyan: {
    main: '#13c2c2',
    light: '#e6fffb',
    dark: '#08979c',
  },
  // 深蓝色系
  deepBlue: {
    main: '#2f54eb',
    light: '#f0f5ff',
    dark: '#1d39c4',
  },
  // 琥珀色系
  amber: {
    main: '#d97706',
    light: '#fffbeb',
    dark: '#b45309',
  },
  // 中性色
  gray: {
    50: '#fafafa',
    100: '#f5f5f5',
    200: '#f0f0f0',
    300: '#e8e8e8',
    400: '#d9d9d9',
    500: '#bfbfbf',
    600: '#8c8c8c',
    700: '#666666',
    800: '#434343',
    900: '#1a1a1a',
  },
} as const

// ============================================
// 品牌色 (Brand Colors)
// ============================================

/**
 * 主品牌色
 */
export const BRAND = {
  primary: {
    main: '#5E6AD2',
    light: '#E8EAFC',
    dark: '#4A56B0',
  },
} as const

// ============================================
// 语义色 (Semantic Colors)
// ============================================

/**
 * 语义化颜色：用于传达信息和状态
 */
export const SEMANTIC = {
  // 交互色（链接、按钮等）
  interactive: {
    main: PALETTE.blue.main,
    light: PALETTE.blue.light,
    shadow: PALETTE.blue.shadow,
  },
  // 成功状态
  success: {
    main: PALETTE.green.main,
    light: PALETTE.green.light,
    dark: PALETTE.green.dark,
  },
  // 警告状态
  warning: {
    main: PALETTE.gold.main,
    light: PALETTE.gold.light,
    dark: PALETTE.gold.dark,
  },
  // 错误状态
  error: {
    main: '#ff4d4f',
    light: PALETTE.red.light,
    dark: PALETTE.red.dark,
  },
  // 信息提示
  info: {
    main: PALETTE.blue.main,
    light: PALETTE.blue.light,
    dark: PALETTE.blue.dark,
  },
} as const

// ============================================
// 层级颜色 (Hierarchy Colors)
// ============================================

/**
 * 拓扑层级颜色
 * 用于3D可视化和拓扑图中区分不同层级
 */
export const HIERARCHY = {
  pod: PALETTE.orange.main,    // 橙色 - Pod层级
  rack: PALETTE.blue.main,      // 蓝色 - Rack层级
  board: PALETTE.green.main,    // 绿色 - Board层级
  chip: PALETTE.purple.main,    // 紫色 - Chip层级
} as const

/**
 * Chip类型颜色
 */
export const CHIP_TYPE = {
  chip: PALETTE.amber.main,     // 琥珀色 - 通用Chip
} as const

/**
 * Switch层级颜色
 */
export const SWITCH_LAYER = {
  leaf: PALETTE.cyan.main,      // 青色 - Leaf交换机
  spine: PALETTE.gold.main,     // 金色 - Spine交换机
  core: PALETTE.red.main,       // 红色 - Core交换机
} as const

// ============================================
// 任务类型颜色 (Task Type Colors)
// ============================================

/**
 * LLM推理任务时间分解颜色
 * 用于甘特图、性能分析图表等
 */
export const TASK_TYPE = {
  // 计算任务
  compute: PALETTE.green.main,      // 绿色 - 计算密集型
  // 访存任务
  memory: PALETTE.blue.main,        // 蓝色 - 内存访问
  // 通信任务
  communication: {
    tp: PALETTE.purple.main,        // 紫色 - 张量并行通信
    pp: PALETTE.magenta.main,       // 品红 - 流水线并行通信
    ep: PALETTE.pink.main,          // 粉色 - 专家并行通信
    sp: PALETTE.deepBlue.main,      // 深蓝 - 序列并行通信
  },
} as const

// ============================================
// UI 基础颜色 (UI Base Colors)
// ============================================

/**
 * 文本颜色
 */
export const TEXT = {
  primary: PALETTE.gray[900],       // 主要文本 - 深黑
  secondary: PALETTE.gray[700],     // 次要文本 - 灰色
  muted: PALETTE.gray[600],         // 弱化文本 - 浅灰
  placeholder: PALETTE.gray[500],   // 占位符
  disabled: PALETTE.gray[400],      // 禁用状态
} as const

/**
 * 边框颜色
 */
export const BORDER = {
  default: PALETTE.gray[300],       // 默认边框
  light: PALETTE.gray[200],         // 浅色边框
  dark: PALETTE.gray[400],          // 深色边框
} as const

/**
 * 背景颜色
 */
export const BACKGROUND = {
  primary: '#ffffff',               // 主背景 - 白色
  secondary: PALETTE.gray[50],      // 次级背景 - 极浅灰
  tertiary: PALETTE.gray[100],      // 三级背景 - 浅灰
  card: '#ffffff',                  // 卡片背景
  hover: PALETTE.gray[50],          // 悬停背景
  active: PALETTE.gray[100],        // 激活背景
} as const

// ============================================
// 导出统一的颜色对象
// ============================================

/**
 * 统一的颜色系统
 * 推荐在组件中使用此对象，避免直接使用硬编码颜色值
 */
export const COLORS = {
  // 基础色板
  palette: PALETTE,

  // 品牌色
  brand: BRAND,

  // 语义色
  semantic: SEMANTIC,

  // 业务领域色
  hierarchy: HIERARCHY,
  chipType: CHIP_TYPE,
  switchLayer: SWITCH_LAYER,
  taskType: TASK_TYPE,

  // UI基础色
  text: TEXT,
  border: BORDER,
  background: BACKGROUND,
} as const

// ============================================
// 兼容性导出（便于迁移）
// ============================================

/**
 * @deprecated 请使用 COLORS.hierarchy
 * 为了兼容旧代码，保留此导出
 */
export const LEVEL_COLORS: Record<string, string> = {
  pod: HIERARCHY.pod,
  rack: HIERARCHY.rack,
  board: HIERARCHY.board,
  chip: HIERARCHY.chip,
}

/**
 * @deprecated 请使用 COLORS.chipType
 */
export const CHIP_TYPE_COLORS: Record<string, string> = {
  chip: CHIP_TYPE.chip,
}

/**
 * @deprecated 请使用 COLORS.switchLayer
 */
export const SWITCH_LAYER_COLORS: Record<string, string> = {
  leaf: SWITCH_LAYER.leaf,
  spine: SWITCH_LAYER.spine,
  core: SWITCH_LAYER.core,
}

/**
 * @deprecated 请使用 COLORS.taskType
 */
export const TIME_BREAKDOWN_COLORS = {
  compute: TASK_TYPE.compute,
  memory: TASK_TYPE.memory,
  tp: TASK_TYPE.communication.tp,
  pp: TASK_TYPE.communication.pp,
  ep: TASK_TYPE.communication.ep,
  sp: TASK_TYPE.communication.sp,
} as const

/**
 * @deprecated 请使用 COLORS.taskType
 */
export const TIME_BREAKDOWN_LABELS = {
  compute: '计算',
  memory: '访存',
  tp: 'TP通信',
  pp: 'PP通信',
  ep: 'EP通信',
  sp: 'SP通信',
} as const

/**
 * ConfigSelectors 样式颜色对象（兼容）
 * @deprecated 请逐步迁移到 COLORS 对象
 */
export const colors = {
  primary: BRAND.primary.main,
  primaryLight: BRAND.primary.light,
  interactive: SEMANTIC.interactive.main,
  interactiveLight: SEMANTIC.interactive.light,
  interactiveShadow: SEMANTIC.interactive.shadow,
  success: SEMANTIC.success.main,
  successLight: SEMANTIC.success.light,
  warning: SEMANTIC.warning.main,
  warningLight: SEMANTIC.warning.light,
  error: SEMANTIC.error.main,
  errorLight: SEMANTIC.error.light,
  border: BORDER.default,
  borderLight: BORDER.light,
  background: BACKGROUND.secondary,
  backgroundDark: BACKGROUND.tertiary,
  cardBg: BACKGROUND.card,
  text: TEXT.primary,
  textSecondary: TEXT.secondary,
} as const

// ============================================
// 类型导出
// ============================================

export type ColorToken = typeof COLORS
export type PaletteColor = keyof typeof PALETTE
export type SemanticColor = keyof typeof SEMANTIC
export type HierarchyLevel = keyof typeof HIERARCHY
