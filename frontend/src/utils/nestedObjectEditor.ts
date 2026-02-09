/**
 * 嵌套对象编辑工具函数
 *
 * 提供通用的嵌套对象访问、修改、克隆和检测功能
 * 用于 ChipPresetEditor 和 TopologyInfoCard 等组件
 */

/**
 * 获取嵌套属性值
 * @param obj 对象
 * @param path 点分隔的路径，如 "memory.gmem.capacity_gb"
 * @returns 属性值，如果路径不存在返回 undefined
 */
export function getNested(obj: unknown, path: string): unknown {
  let cur: unknown = obj
  for (const k of path.split('.')) {
    if (cur == null || typeof cur !== 'object') return undefined
    cur = (cur as Record<string, unknown>)[k]
  }
  return cur
}

/**
 * 设置嵌套属性值
 * @param obj 对象（会被修改）
 * @param path 点分隔的路径
 * @param val 要设置的值
 */
export function setNested(obj: Record<string, unknown>, path: string, val: unknown): void {
  const keys = path.split('.')
  let cur: Record<string, unknown> = obj
  for (let i = 0; i < keys.length - 1; i++) {
    if (cur[keys[i]] == null || typeof cur[keys[i]] !== 'object') {
      cur[keys[i]] = {}
    }
    cur = cur[keys[i]] as Record<string, unknown>
  }
  cur[keys[keys.length - 1]] = val
}

/**
 * 深度克隆对象
 * @param obj 要克隆的对象
 * @returns 克隆后的对象
 */
export function deepClone<T>(obj: T): T {
  return structuredClone(obj)
}

/**
 * 判断是否为普通对象（非数组）
 * @param v 要检测的值
 * @returns 是否为普通对象
 */
export function isPlainObject(v: unknown): v is Record<string, unknown> {
  return v != null && typeof v === 'object' && !Array.isArray(v)
}

/**
 * 检测两个值是否相同（通过 JSON 序列化比较）
 * @param a 值 A
 * @param b 值 B
 * @returns 是否相同
 */
export function isValueEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b)
}

/**
 * 检测值是否被修改
 * @param original 原始值
 * @param current 当前值
 * @returns 是否被修改
 */
export function isValueModified(original: unknown, current: unknown): boolean {
  return !isValueEqual(original, current)
}

/**
 * 获取嵌套路径的修改状态
 * @param originalObj 原始对象
 * @param currentObj 当前对象
 * @param path 点分隔的路径
 * @returns 是否被修改
 */
export function isPathModified(
  originalObj: unknown,
  currentObj: unknown,
  path: string
): boolean {
  if (!originalObj || !currentObj) return false
  const originalValue = getNested(originalObj, path)
  const currentValue = getNested(currentObj, path)
  return isValueModified(originalValue, currentValue)
}

/**
 * 提取错误信息
 * @param err 错误对象
 * @returns 错误消息字符串
 */
export function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}
