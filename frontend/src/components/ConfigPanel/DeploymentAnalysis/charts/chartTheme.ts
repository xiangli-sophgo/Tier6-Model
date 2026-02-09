/**
 * 图表统一主题配置
 * 定义颜色方案、卡片样式、渐变色等
 */

/** 主色调 */
export const CHART_COLORS = {
  /** 主色 - 淡蓝色 */
  primary: '#60A5FA',
  /** 成功色 - 绿色 */
  success: '#52c41a',
  /** 警告色 - 橙色 */
  warning: '#faad14',
  /** 危险色 - 红色 */
  danger: '#ff4d4f',
  /** 信息色 - 蓝色 */
  info: '#1890ff',
  /** 蓝色 */
  purple: '#3B82F6',
  /** 品红色 */
  magenta: '#eb2f96',
  /** 青色 */
  cyan: '#13c2c2',
} as const

/** 图表配色方案（用于多系列数据） */
export const CHART_SERIES_COLORS = [
  '#60A5FA', // 淡蓝色
  '#52c41a', // 绿色
  '#faad14', // 橙色
  '#3B82F6', // 蓝色
  '#eb2f96', // 品红
  '#13c2c2', // 青色
  '#1890ff', // 信息蓝
  '#ff7a45', // 珊瑚橙
] as const

/** 渐变色配置 */
export const CHART_GRADIENTS = {
  /** 蓝色渐变（主色调） */
  blue: {
    start: '#60A5FA',
    end: '#93C5FD',
  },
  /** 绿色渐变（成功） */
  green: {
    start: '#52c41a',
    end: '#95de64',
  },
  /** 橙色渐变（警告） */
  orange: {
    start: '#fa8c16',
    end: '#ffc53d',
  },
  /** 红色渐变（危险） */
  red: {
    start: '#ff4d4f',
    end: '#ff7875',
  },
  /** 蓝色渐变 */
  purple: {
    start: '#3B82F6',
    end: '#60A5FA',
  },
} as const

/** 卡片样式 */
export const CHART_CARD_STYLE: React.CSSProperties = {
  background: '#fff',
  borderRadius: 12,
  padding: 16,
  border: '1px solid #E5E5E5',
  boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
}

/** 卡片标题样式 */
export const CHART_TITLE_STYLE: React.CSSProperties = {
  fontSize: 14,
  fontWeight: 600,
  color: '#1a1a1a',
  marginBottom: 12,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
}

/** 评分状态颜色 */
export const SCORE_STATUS_COLORS = {
  /** 优秀 (>=80) */
  excellent: '#52c41a',
  /** 良好 (>=60) */
  good: '#faad14',
  /** 较差 (<60) */
  poor: '#ff4d4f',
} as const

/** 根据分数获取状态颜色 */
export function getScoreColor(score: number): string {
  if (score >= 80) return SCORE_STATUS_COLORS.excellent
  if (score >= 60) return SCORE_STATUS_COLORS.good
  return SCORE_STATUS_COLORS.poor
}

/** 根据分数获取状态标签 */
export function getScoreLabel(score: number): string {
  if (score >= 90) return '优秀'
  if (score >= 80) return '良好'
  if (score >= 60) return '一般'
  if (score >= 40) return '较差'
  return '差'
}

/** 甘特图任务类型颜色 - 简化为 5 大类 */
export const GANTT_CATEGORY_COLORS = {
  /** 计算类 - 绿色系 */
  compute: {
    primary: '#52c41a',
    light: '#95de64',
    dark: '#237804',
  },
  /** 内存访问类 - 橙色系 */
  memory: {
    primary: '#fa8c16',
    light: '#ffc53d',
    dark: '#d48806',
  },
  /** TP 通信 - 蓝色 */
  tp: {
    primary: '#1890ff',
    light: '#69c0ff',
    dark: '#096dd9',
  },
  /** PP 通信 - 蓝色 */
  pp: {
    primary: '#3B82F6',
    light: '#60A5FA',
    dark: '#2563EB',
  },
  /** EP/MoE 通信 - 品红色 */
  ep: {
    primary: '#eb2f96',
    light: '#ff85c0',
    dark: '#c41d7f',
  },
  /** 其他（空闲、气泡等） */
  other: {
    primary: '#d9d9d9',
    light: '#f0f0f0',
    dark: '#8c8c8c',
  },
} as const

/** 将任务类型映射到大类 */
export function getTaskCategory(taskType: string): keyof typeof GANTT_CATEGORY_COLORS {
  // 计算类
  if ([
    'compute', 'embedding', 'layernorm',
    'attention_qkv', 'attention_score', 'attention_softmax', 'attention_output',
    'ffn_gate', 'ffn_up', 'ffn_down', 'lm_head',
    'rmsnorm_q_lora', 'rmsnorm_kv_lora', 'mm_q_lora_a', 'mm_q_lora_b', 'mm_kv_lora_a',
    'attn_fc', 'bmm_qk', 'bmm_sv',
  ].includes(taskType)) {
    return 'compute'
  }

  // 内存访问类
  if ([
    'pcie_h2d', 'pcie_d2h', 'hbm_write', 'hbm_read',
    'weight_load', 'kv_cache_read', 'kv_cache_write',
  ].includes(taskType)) {
    return 'memory'
  }

  // TP 通信
  if (['tp_comm', 'sp_allgather', 'sp_reduce_scatter'].includes(taskType)) {
    return 'tp'
  }

  // PP 通信
  if (['pp_comm', 'dp_gradient_sync'].includes(taskType)) {
    return 'pp'
  }

  // EP/MoE 通信
  if ([
    'ep_comm', 'ep_dispatch', 'ep_combine',
    'moe_gate', 'moe_expert', 'moe_shared_expert',
  ].includes(taskType)) {
    return 'ep'
  }

  // 其他
  return 'other'
}

/** 雷达图六维评分维度定义 */
export const RADAR_DIMENSIONS = [
  { key: 'latency', name: '延迟', tip: 'TTFT 越低越好' },
  { key: 'throughput', name: '吞吐', tip: 'TPS/Chip 越高越好' },
  { key: 'efficiency', name: '效率', tip: '综合 MFU + MBU' },
  { key: 'balance', name: '均衡', tip: '负载均匀度' },
  { key: 'memory', name: '显存', tip: '60-80% 最优' },
  { key: 'communication', name: '通信', tip: '通信开销越小越好' },
] as const

/** 瓶颈类型配色 */
export const BOTTLENECK_COLORS = {
  memory: '#1890ff',    // 带宽受限 - 蓝色
  compute: '#52c41a',   // 算力受限 - 绿色
  communication: '#faad14', // 通信受限 - 橙色
  balanced: '#3B82F6',  // 均衡 - 蓝色
} as const

/** 显存分解配色 */
export const MEMORY_COMPONENT_COLORS = {
  model: '#3B82F6',      // 模型参数 - 蓝色（最深）
  kv: '#60A5FA',         // KV Cache - 淡蓝色（中等）
  activation: '#93C5FD', // 激活值 - 浅蓝色（最浅）
  overhead: '#BFDBFE',   // 其他开销 - 极浅蓝色
} as const

/** 空状态提示文字 */
export const EMPTY_STATE_TEXT = {
  noData: '暂无数据',
  noResult: '运行分析以查看结果',
  loading: '加载中...',
  error: '加载失败',
} as const

/** 通用图表配置 */
export const ECHARTS_COMMON_CONFIG = {
  /** 工具提示通用配置 - 浅色风格 */
  tooltip: {
    backgroundColor: 'rgba(255, 255, 255, 0.98)',
    borderColor: '#e5e5e5',
    borderWidth: 1,
    textStyle: {
      color: '#333',
      fontSize: 12,
    } as const,
    padding: [10, 14],
    extraCssText: 'box-shadow: 0 4px 12px rgba(0,0,0,0.1);',
  },
  /** 网格通用配置 */
  grid: {
    left: 50,
    right: 20,
    top: 40,
    bottom: 40,
    containLabel: true,
  },
  /** 坐标轴通用配置 */
  axis: {
    axisLine: {
      lineStyle: { color: '#d9d9d9' },
    },
    axisTick: {
      lineStyle: { color: '#d9d9d9' },
    },
    axisLabel: {
      color: '#666',
      fontSize: 11,
    },
    splitLine: {
      lineStyle: { color: '#f0f0f0', type: 'dashed' as const },
    },
  },
}
