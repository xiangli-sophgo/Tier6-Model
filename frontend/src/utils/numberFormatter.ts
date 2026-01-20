/**
 * 数字格式化工具函数
 * 解决 JavaScript 浮点数精度问题
 */

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
 * 格式化百分比
 */
export function formatPercent(
  value: number | undefined | null,
  decimals: number = 1
): string {
  if (value === undefined || value === null || isNaN(value)) {
    return '0%'
  }
  return `${formatNumber(value * 100, decimals)}%`
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
