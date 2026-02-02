/**
 * 参数遍历功能 - 核心算法
 * 从 CrossRing 移植（包含参数绑定功能）
 */

import type { SweepParam } from './sweepTypes'

// ============================================
// 绑定组管理
// ============================================

/**
 * 获取下一个可用的绑定组ID
 */
export function getNextBindGroupId(existingGroups: string[]): string {
  const availableGroups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
  for (const group of availableGroups) {
    if (!existingGroups.includes(group)) {
      return group
    }
  }
  return 'A' // 如果全部用完，返回A
}

/**
 * 获取所有已使用的绑定组ID
 */
export function getExistingBindGroups(sweepParams: SweepParam[]): string[] {
  const groups = new Set<string>()
  for (const param of sweepParams) {
    if (param.bindGroupId) {
      groups.add(param.bindGroupId)
    }
  }
  return Array.from(groups).sort()
}

/**
 * 验证绑定组的一致性
 * 同一绑定组的参数必须有相同数量的值
 */
export function validateBindings(sweepParams: SweepParam[]): string[] {
  const errors: string[] = []
  const bindGroupsMap = new Map<string, SweepParam[]>()

  // 按绑定组分组
  for (const param of sweepParams) {
    if (param.bindGroupId) {
      if (!bindGroupsMap.has(param.bindGroupId)) {
        bindGroupsMap.set(param.bindGroupId, [])
      }
      bindGroupsMap.get(param.bindGroupId)!.push(param)
    }
  }

  // 验证每个绑定组
  for (const [groupId, params] of bindGroupsMap.entries()) {
    if (params.length < 2) {
      errors.push(`绑定组 ${groupId} 只有1个参数，至少需要2个参数才能绑定`)
      continue
    }

    // 检查值数量是否一致
    const firstValueCount = params[0].values.length
    for (const param of params) {
      if (param.values.length !== firstValueCount) {
        errors.push(
          `绑定组 ${groupId} 参数值数量不一致：` +
          `${params[0].label}(${firstValueCount}个) vs ${param.label}(${param.values.length}个)`
        )
      }
    }
  }

  return errors
}

/**
 * 计算参数值列表（线性遍历）
 */
export function calculateSweepValues(start: number, end: number, step: number): number[] {
  if (step <= 0 || start > end) return [start]

  const values: number[] = []
  // 使用浮点数容差避免精度问题
  for (let v = start; v <= end + step * 0.001; v += step) {
    values.push(Math.round(v * 1000) / 1000)
  }

  return values
}

/**
 * 生成笛卡尔积组合（无绑定版本）
 */
export function generateCombinations(sweepParams: SweepParam[]): Record<string, number>[] {
  if (sweepParams.length === 0) return []

  const combinations: Record<string, number>[] = []

  function generate(index: number, current: Record<string, number>) {
    if (index >= sweepParams.length) {
      combinations.push({ ...current })
      return
    }

    const param = sweepParams[index]
    for (const value of param.values) {
      generate(index + 1, { ...current, [param.key]: value })
    }
  }

  generate(0, {})
  return combinations
}

/**
 * 计算总组合数（无绑定版本）
 */
export function calculateTotalCombinations(sweepParams: SweepParam[]): number {
  if (sweepParams.length === 0) return 0

  let total = 1
  for (const param of sweepParams) {
    total *= param.values.length
  }

  return total
}

/**
 * 验证参数配置（包含绑定验证）
 */
export function validateSweepParams(sweepParams: SweepParam[]): string[] {
  const errors: string[] = []

  for (const param of sweepParams) {
    // 验证起始/结束/步长
    if (param.start > param.end) {
      errors.push(`参数 "${param.label}" 的起始值不能大于结束值`)
    }
    if (param.step <= 0) {
      errors.push(`参数 "${param.label}" 的步长必须大于0`)
    }
    if (param.values.length === 0) {
      errors.push(`参数 "${param.label}" 没有生成任何值`)
    }
    if (param.values.length > 100) {
      errors.push(`参数 "${param.label}" 生成了过多值（${param.values.length}个），建议减小范围或增大步长`)
    }
  }

  // 验证绑定组
  const bindingErrors = validateBindings(sweepParams)
  errors.push(...bindingErrors)

  // 验证总组合数（使用带绑定的计算）
  const total = calculateTotalCombinationsWithBinding(sweepParams)
  if (total > 500) {
    errors.push(`总组合数过多（${total}个），建议减少参数或值的数量（最大500）`)
  }

  return errors
}

// ============================================
// 带绑定的组合生成
// ============================================

/**
 * 计算总组合数（支持绑定）
 */
export function calculateTotalCombinationsWithBinding(sweepParams: SweepParam[]): number {
  if (sweepParams.length === 0) return 0

  // 按绑定组分组
  const bindGroupsMap = new Map<string, SweepParam[]>()
  const unboundParams: SweepParam[] = []

  for (const param of sweepParams) {
    if (param.bindGroupId) {
      if (!bindGroupsMap.has(param.bindGroupId)) {
        bindGroupsMap.set(param.bindGroupId, [])
      }
      bindGroupsMap.get(param.bindGroupId)!.push(param)
    } else {
      unboundParams.push(param)
    }
  }

  let total = 1

  // 绑定组：每组算一次（取第一个参数的值数量）
  for (const params of bindGroupsMap.values()) {
    if (params.length > 0) {
      total *= params[0].values.length
    }
  }

  // 未绑定参数：笛卡尔积
  for (const param of unboundParams) {
    total *= param.values.length
  }

  return total
}

/**
 * 生成参数组合（支持绑定）
 */
export function generateCombinationsWithBinding(sweepParams: SweepParam[]): Record<string, number>[] {
  if (sweepParams.length === 0) return []

  // 按绑定组分组
  const bindGroupsMap = new Map<string, SweepParam[]>()
  const unboundParams: SweepParam[] = []

  for (const param of sweepParams) {
    if (param.bindGroupId) {
      if (!bindGroupsMap.has(param.bindGroupId)) {
        bindGroupsMap.set(param.bindGroupId, [])
      }
      bindGroupsMap.get(param.bindGroupId)!.push(param)
    } else {
      unboundParams.push(param)
    }
  }

  // 构建参数列表（绑定组作为整体）
  const paramGroups: Array<{ type: 'group'; params: SweepParam[] } | { type: 'single'; param: SweepParam }> = []

  for (const params of bindGroupsMap.values()) {
    paramGroups.push({ type: 'group', params })
  }

  for (const param of unboundParams) {
    paramGroups.push({ type: 'single', param })
  }

  // 递归生成组合
  const combinations: Record<string, number>[] = []

  function generate(index: number, current: Record<string, number>) {
    if (index >= paramGroups.length) {
      combinations.push({ ...current })
      return
    }

    const group = paramGroups[index]

    if (group.type === 'group') {
      // 绑定组：同时遍历（索引同步）
      const valueCount = group.params[0].values.length
      for (let i = 0; i < valueCount; i++) {
        const updated = { ...current }
        for (const param of group.params) {
          updated[param.key] = param.values[i]
        }
        generate(index + 1, updated)
      }
    } else {
      // 单个参数：独立遍历
      for (const value of group.param.values) {
        generate(index + 1, { ...current, [group.param.key]: value })
      }
    }
  }

  generate(0, {})
  return combinations
}

/**
 * 应用参数组合到配置对象
 * @param baseConfig 基础配置对象 { model, inference, hardware, parallelism }
 * @param combination 参数组合 { "model.hidden_size": 4096, ... }
 * @returns 覆盖后的新配置对象
 */
export function applyParameterCombination(
  baseConfig: {
    model: any
    inference: any
    hardware: any
    parallelism: any
  },
  combination: Record<string, number>
): {
  model: any
  inference: any
  hardware: any
  parallelism: any
} {
  // 深拷贝基础配置
  const newConfig = {
    model: JSON.parse(JSON.stringify(baseConfig.model)),
    inference: JSON.parse(JSON.stringify(baseConfig.inference)),
    hardware: JSON.parse(JSON.stringify(baseConfig.hardware)),
    parallelism: JSON.parse(JSON.stringify(baseConfig.parallelism)),
  }

  // 应用参数覆盖
  for (const [path, value] of Object.entries(combination)) {
    const [category, ...keys] = path.split('.')

    if (category === 'model') {
      setValueByPath(newConfig.model, keys.join('.'), value)
    } else if (category === 'inference') {
      setValueByPath(newConfig.inference, keys.join('.'), value)
    } else if (category === 'hardware') {
      setValueByPath(newConfig.hardware, keys.join('.'), value)
    } else if (category === 'parallelism') {
      setValueByPath(newConfig.parallelism, keys.join('.'), value)
    }
  }

  return newConfig
}

/**
 * 根据路径设置值（深度设置）
 */
function setValueByPath(obj: any, path: string, value: number): void {
  const keys = path.split('.')
  let current = obj

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i]
    if (current[key] === undefined || current[key] === null) {
      current[key] = {}
    }
    current = current[key]
  }

  current[keys[keys.length - 1]] = value
}
