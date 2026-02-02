/**
 * 参数遍历功能 - 类型定义
 */

// 参数遍历配置
export interface SweepParam {
  key: string           // 参数路径 (如 "model.hidden_size", "inference.batch_size")
  label: string         // 显示名称
  start: number         // 起始值
  end: number           // 结束值
  step: number          // 步长
  values: number[]      // 计算得到的值列表
  currentValue: number  // 当前配置中的值（用于显示参考）
  unit?: string         // 单位 (如 "GB/s", "TFLOPS")
  bindGroupId?: string  // 绑定组ID (A-H) - 用于参数分组同步遍历
}

// 绑定组颜色常量（参考 CrossRing）
export const BIND_GROUP_COLORS: Record<string, string> = {
  'A': '#e6f7ff',  // 浅蓝
  'B': '#f6ffed',  // 浅绿
  'C': '#fff7e6',  // 浅橙
  'D': '#f9f0ff',  // 浅紫
  'E': '#fff1f0',  // 浅红
  'F': '#e6fffb',  // 浅青
  'G': '#fcffe6',  // 浅黄绿
  'H': '#fff0f6',  // 浅粉
}

// 可遍历参数元数据
export interface SweepableParameter {
  key: string                      // 参数路径
  label: string                    // 显示名称
  currentValue: number             // 当前值
  defaultRange: {                  // 建议范围
    min: number
    max: number
    step: number
  }
  unit?: string                    // 单位
  category: 'model' | 'inference' | 'hardware' | 'parallelism' | 'topology'
}

// 保存的遍历配置
export interface SavedSweepConfig {
  name: string
  description?: string
  params: SweepParam[]
  timestamp: number
}
