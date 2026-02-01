import * as React from 'react'
import { Label } from './label'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './tooltip'
import { cn } from '@/lib/utils'

export interface TooltipLabelProps {
  /** 标签文本 */
  label: React.ReactNode
  /** Tooltip 提示内容 */
  tooltip?: React.ReactNode
  /** 是否必填（显示红色星号） */
  required?: boolean
  /** 标签额外样式 */
  className?: string
  /** 是否包裹 TooltipProvider（如果外层已有可设为 false） */
  withProvider?: boolean
}

/**
 * 带 Tooltip 的标签组件
 * - 自动处理 Tooltip 显示逻辑
 * - 支持必填标记
 * - 统一的样式
 *
 * @example
 * ```tsx
 * <TooltipLabel label="Pod 数量" tooltip="数据中心内的Pod数量" />
 * <TooltipLabel label="带宽" tooltip="互联带宽" required />
 * ```
 */
export const TooltipLabel: React.FC<TooltipLabelProps> = ({
  label,
  tooltip,
  required,
  className = 'text-xs text-gray-500 cursor-help',
  withProvider = true,
}) => {
  const labelElement = (
    <Label className={cn(className)}>
      {label}
      {required && <span className="text-red-500 ml-0.5">*</span>}
    </Label>
  )

  // 如果没有 tooltip，直接返回 Label
  if (!tooltip) {
    return <Label className={cn(className)}>{label}</Label>
  }

  const tooltipContent = (
    <Tooltip>
      <TooltipTrigger asChild>
        {labelElement}
      </TooltipTrigger>
      <TooltipContent>{tooltip}</TooltipContent>
    </Tooltip>
  )

  // 根据 withProvider 决定是否包裹 TooltipProvider
  if (withProvider) {
    return <TooltipProvider>{tooltipContent}</TooltipProvider>
  }

  return tooltipContent
}
