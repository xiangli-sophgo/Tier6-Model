/**
 * 统一的格式化工具库
 *
 * 整合了所有数字、字节、时间、百分比等格式化函数
 * 解决了 JavaScript 浮点数精度问题
 */

// ============================================
// 基础数字格式化
// ============================================

/**
 * 格式化数字，自动处理浮点数精度
 * @param value 数值
 * @param decimals 保留小数位数，默认2位
 * @param removeTrailingZeros 是否移除末尾的0，默认true
 */
export function formatNumber(
  value: number | undefined | null,
  decimals: number = 2,
  removeTrailingZeros: boolean = true
): string {
  if (value === undefined || value === null || isNaN(value)) {
    return '0'
  }

  // 使用 toFixed 处理精度
  let result = value.toFixed(decimals)

  // 移除末尾的0和小数点
  if (removeTrailingZeros) {
    result = result.replace(/\.?0+$/, '')
  }

  return result
}

/**
 * 格式化带单位的数字（如延迟、带宽等）
 */
export function formatMetric(
  value: number | undefined | null,
  unit: string = '',
  decimals: number = 2
): string {
  const formatted = formatNumber(value, decimals)
  return unit ? `${formatted} ${unit}` : formatted
}

/**
 * 格式化大数字（添加千分位）
 */
export function formatLargeNumber(
  value: number | undefined | null,
  decimals: number = 0
): string {
  if (value === undefined || value === null || isNaN(value)) {
    return '0'
  }
  return value.toLocaleString('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: decimals,
  })
}

// ============================================
// 百分比格式化
// ============================================

/**
 * 格式化百分比
 * @param value 比例值 (0-1)
 * @param decimals 小数位数，默认1位
 * @param removeTrailingZeros 是否移除末尾的0，默认true
 */
export function formatPercent(
  value: number | undefined | null,
  decimals: number = 1,
  removeTrailingZeros: boolean = true
): string {
  if (value === undefined || value === null || isNaN(value)) {
    return '0%'
  }

  const percentValue = value * 100
  const formatted = removeTrailingZeros
    ? formatNumber(percentValue, decimals, true)
    : percentValue.toFixed(decimals)

  return `${formatted}%`
}

/**
 * 格式化百分比数值（已经是百分比的数值）
 * @param value 百分比数值 (0-100)
 * @param decimals 小数位数，默认1位
 * @param removeTrailingZeros 是否移除末尾的0，默认true
 */
export function formatPercentValue(
  value: number | undefined | null,
  decimals: number = 1,
  removeTrailingZeros: boolean = true
): string {
  if (value === undefined || value === null || isNaN(value)) {
    return '0%'
  }

  const formatted = removeTrailingZeros
    ? formatNumber(value, decimals, true)
    : value.toFixed(decimals)

  return `${formatted}%`
}

// ============================================
// 字节数格式化
// ============================================

/**
 * 格式化字节数
 * @param bytes 字节数
 * @param decimals 小数位数，默认2位
 * @returns 格式化后的字符串 (B, KB, MB, GB, TB, PB)
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 B'
  if (bytes < 0) return '-' + formatBytes(-bytes, decimals)

  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
  const k = 1024
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  const value = bytes / Math.pow(k, i)

  return `${value.toFixed(i > 0 ? decimals : 0)} ${units[i]}`
}

// ============================================
// 时间格式化
// ============================================

/**
 * 格式化时间 (微秒)
 * @param us 微秒数
 * @param decimals 小数位数，默认2位
 * @returns 格式化后的字符串 (ns, µs, ms, s)
 */
export function formatTime(us: number, decimals: number = 2): string {
  if (us < 0) return '-' + formatTime(-us, decimals)
  if (us < 1) return `${(us * 1000).toFixed(0)} ns`
  if (us < 1000) return `${us.toFixed(decimals)} µs`
  if (us < 1000000) return `${(us / 1000).toFixed(decimals)} ms`
  return `${(us / 1000000).toFixed(decimals)} s`
}

/**
 * 格式化时间 (毫秒)
 * @param ms 毫秒数
 * @param decimals 小数位数，默认2位
 * @returns 格式化后的字符串 (ns, µs, ms, s)
 */
export function formatTimeMs(ms: number, decimals: number = 2): string {
  if (ms < 0) return '-' + formatTimeMs(-ms, decimals)
  if (ms < 0.001) return `${(ms * 1000000).toFixed(0)} ns`
  if (ms < 1) return `${(ms * 1000).toFixed(decimals)} µs`
  if (ms < 1000) return `${ms.toFixed(decimals)} ms`
  return `${(ms / 1000).toFixed(decimals)} s`
}

// ============================================
// 计算性能格式化
// ============================================

/**
 * 格式化 FLOPs (浮点运算次数)
 * @param flops FLOPs 数
 * @param decimals 小数位数，默认2位
 * @returns 格式化后的字符串 (FLOPs, KFLOPs, MFLOPs, GFLOPs, TFLOPs, PFLOPs, EFLOPs)
 */
export function formatFlops(flops: number, decimals: number = 2): string {
  if (flops === 0) return '0 FLOPs'
  if (flops < 0) return '-' + formatFlops(-flops, decimals)

  const units = ['', 'K', 'M', 'G', 'T', 'P', 'E']
  const k = 1000
  const i = Math.floor(Math.log(flops) / Math.log(k))
  const value = flops / Math.pow(k, i)

  return `${value.toFixed(decimals)} ${units[i]}FLOPs`
}

/**
 * 格式化带宽 (bytes/s)
 * @param bytesPerSecond 字节每秒
 * @param decimals 小数位数，默认2位
 * @returns 格式化后的字符串 (B/s, KB/s, MB/s, GB/s, TB/s)
 */
export function formatBandwidth(bytesPerSecond: number, decimals: number = 2): string {
  if (bytesPerSecond === 0) return '0 B/s'
  if (bytesPerSecond < 0) return '-' + formatBandwidth(-bytesPerSecond, decimals)

  const units = ['B/s', 'KB/s', 'MB/s', 'GB/s', 'TB/s', 'PB/s']
  const k = 1024
  const i = Math.floor(Math.log(bytesPerSecond) / Math.log(k))
  const value = bytesPerSecond / Math.pow(k, i)

  return `${value.toFixed(decimals)} ${units[i]}`
}

// ============================================
// 特殊格式化
// ============================================

/**
 * 格式化 GEMM 矩阵乘法形状
 * @param shape [M, N, K] 数组
 * @returns 格式化后的字符串 "M=xxx, N=xxx, K=xxx"
 */
export function formatGemmShape(shape: [number, number, number]): string {
  const [m, n, k] = shape
  return `M=${m}, N=${n}, K=${k}`
}

/**
 * 格式化延迟 (自动选择单位)
 * @param latency 延迟值
 * @param unit 输入单位 ('us' | 'ms' | 's')，默认 'us'
 * @param decimals 小数位数，默认2位
 */
export function formatLatency(
  latency: number,
  unit: 'us' | 'ms' | 's' = 'us',
  decimals: number = 2
): string {
  // 统一转换为微秒
  let us = latency
  if (unit === 'ms') us = latency * 1000
  if (unit === 's') us = latency * 1000000

  return formatTime(us, decimals)
}

// ============================================
// 日期格式化
// ============================================

/**
 * 格式化日期字符串为中文本地化格式
 * @param dateStr 日期字符串
 * @returns 格式化后的日期字符串
 */
export function formatDate(dateStr: string | undefined | null): string {
  if (!dateStr) return '-'
  try { return new Date(dateStr).toLocaleString('zh-CN') }
  catch { return '-' }
}

// ============================================
// 参数量格式化
// ============================================

/**
 * 格式化参数量显示
 * @param params 参数量
 * @returns 格式化字符串 (如 "1.5B", "235M")
 */
export function formatParams(params: number): string {
  if (params >= 1e12) {
    return `${(params / 1e12).toFixed(1)}T`
  } else if (params >= 1e9) {
    return `${(params / 1e9).toFixed(1)}B`
  } else if (params >= 1e6) {
    return `${(params / 1e6).toFixed(1)}M`
  } else if (params >= 1e3) {
    return `${(params / 1e3).toFixed(1)}K`
  }
  return String(params)
}

// ============================================
// 导出常量
// ============================================

/**
 * 默认的格式化配置
 */
export const DEFAULT_FORMAT_CONFIG = {
  numberDecimals: 2,
  percentDecimals: 2,
  timeDecimals: 2,
  bytesDecimals: 2,
  flopsDecimals: 2,
  removeTrailingZeros: true,
} as const

// ============================================
// 指标统一精度配置
// ============================================

/** 每种指标的标准小数位数 */
export const METRIC_DECIMALS: Record<string, number> = {
  tps: 2,
  tps_per_chip: 2,
  tps_per_batch: 2,
  tpot: 4,
  ttft: 4,
  end_to_end_latency: 2,
  mfu: 2,     // 百分比的小数位
  mbu: 2,
  score: 2,
  chips: 0,
  dram_occupy: 2,
  flops: 2,
  cost_total: 2,
  cost_server: 2,
  cost_interconnect: 2,
  cost_per_chip: 2,
  cost_dfop: 4,
}

/**
 * 根据指标类型格式化数值
 * 统一各页面的显示精度
 *
 * @param metric 指标名称（如 'tps', 'mfu', 'dram_occupy'）
 * @param value 原始数值（MFU/MBU 为 0-1 比例值，内存为字节，flops 为 FLOPs）
 * @returns 格式化后的字符串
 */
export function formatMetricValue(metric: string, value: number | undefined | null): string {
  if (value === undefined || value === null || isNaN(value)) return '-'

  const decimals = METRIC_DECIMALS[metric] ?? 2

  // 百分比指标：MFU/MBU（0-1 → 百分比）
  if (metric === 'mfu' || metric === 'mbu') {
    return `${formatNumber(value * 100, decimals)}%`
  }

  // 内存占用：字节 → GB
  if (metric === 'dram_occupy') {
    const gb = value / (1024 ** 3)
    return formatNumber(gb, decimals)
  }

  // 计算量：FLOPs → TFLOPs
  if (metric === 'flops') {
    const tflops = value / 1e12
    return formatNumber(tflops, decimals)
  }

  // 整数指标
  if (decimals === 0) {
    return Number.isInteger(value) ? String(value) : String(Math.round(value))
  }

  // 其他数值
  return Number.isInteger(value) ? String(value) : formatNumber(value, decimals)
}

/**
 * 获取指标的标准小数位数
 * 用于图表等需要数值精度但不需要完整格式化的场景
 *
 * 注意：对于百分比指标 (mfu/mbu)，返回的是百分比值的小数位数
 *       （即数据已乘以100后的精度）
 *
 * @param metric 指标名称
 * @returns 小数位数
 */
export function getMetricDecimals(metric: string): number {
  return METRIC_DECIMALS[metric] ?? 2
}
