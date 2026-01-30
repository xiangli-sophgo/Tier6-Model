import * as React from 'react'
import { NumberInput, NumberInputProps } from './number-input'
import { TooltipLabel } from './tooltip-label'
import { cn } from '@/lib/utils'

export interface FormInputFieldProps extends Omit<NumberInputProps, 'className'> {
  /** 标签文本 */
  label: string
  /** Tooltip 提示内容 */
  tooltip?: string
  /** 是否必填 */
  required?: boolean
  /** 容器额外样式 */
  containerClassName?: string
  /** 输入框额外样式（会与默认样式合并） */
  inputClassName?: string
}

/**
 * 表单输入字段组件
 * - 组合了 TooltipLabel + NumberInput
 * - 统一的布局和样式
 * - 减少重复代码
 *
 * @example
 * ```tsx
 * <FormInputField
 *   label="Pod 数量"
 *   tooltip="数据中心内的Pod数量"
 *   value={podCount}
 *   onChange={(v) => setPodCount(v || 1)}
 *   min={1}
 *   max={10}
 * />
 * ```
 */
export const FormInputField: React.FC<FormInputFieldProps> = ({
  label,
  tooltip,
  required,
  containerClassName,
  inputClassName,
  ...inputProps
}) => {
  return (
    <div className={cn(containerClassName)}>
      <div className="mb-1">
        <TooltipLabel label={label} tooltip={tooltip} required={required} />
      </div>
      <NumberInput
        {...inputProps}
        className={cn('w-full h-7 mt-0.5', inputClassName)}
      />
    </div>
  )
}
