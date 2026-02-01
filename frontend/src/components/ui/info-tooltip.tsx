/**
 * InfoTooltip - 统一的信息提示组件
 *
 * 封装了 TooltipProvider + Tooltip + TooltipTrigger + TooltipContent 组合
 * 避免在每个使用处重复这个嵌套结构
 */

import React from 'react'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './tooltip'

// ============================================
// 基础 InfoTooltip 组件
// ============================================

export interface InfoTooltipProps {
  /** 提示内容 */
  content: React.ReactNode
  /** 触发元素 */
  children: React.ReactNode
  /** 最大宽度（默认 400px） */
  maxWidth?: string
  /** 对齐方式 */
  side?: 'top' | 'right' | 'bottom' | 'left'
  /** 是否禁用 */
  disabled?: boolean
  /** 是否延迟显示（默认 true） */
  delayDuration?: number
  /** 自定义类名 */
  className?: string
}

/**
 * 基础信息提示组件
 *
 * @example
 * ```tsx
 * <InfoTooltip content="这是一个提示">
 *   <span>鼠标悬停查看提示</span>
 * </InfoTooltip>
 * ```
 */
export const InfoTooltip: React.FC<InfoTooltipProps> = ({
  content,
  children,
  maxWidth = '400px',
  side = 'top',
  disabled = false,
  delayDuration = 200,
  className,
}) => {
  // 如果禁用，直接返回子元素
  if (disabled) {
    return <>{children}</>
  }

  return (
    <TooltipProvider delayDuration={delayDuration}>
      <Tooltip>
        <TooltipTrigger asChild>{children}</TooltipTrigger>
        <TooltipContent side={side} className={className} style={{ maxWidth }}>
          {content}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

// ============================================
// 带帮助图标的 Tooltip
// ============================================

export interface HelpTooltipProps extends Omit<InfoTooltipProps, 'children'> {
  /** 标签文本 */
  label: string
  /** 标签样式 */
  labelClassName?: string
}

/**
 * 带帮助提示的标签组件
 * 常用于表单字段标签
 *
 * @example
 * ```tsx
 * <HelpTooltip
 *   label="隐藏层"
 *   content="Hidden Size：模型隐藏层维度"
 * />
 * ```
 */
export const HelpTooltip: React.FC<HelpTooltipProps> = ({
  label,
  labelClassName = 'text-gray-500 text-xs cursor-help',
  content,
  ...tooltipProps
}) => {
  return (
    <InfoTooltip content={content} {...tooltipProps}>
      <span className={labelClassName}>{label}</span>
    </InfoTooltip>
  )
}

// ============================================
// 带数学公式渲染的 Tooltip
// ============================================

export interface FormulaTooltipProps extends InfoTooltipProps {
  /** 公式渲染函数（可选，用于渲染 LaTeX 等） */
  renderFormula?: (text: string) => React.ReactNode
}

/**
 * 支持数学公式的 Tooltip
 * 用于显示包含 LaTeX 公式的提示
 *
 * @example
 * ```tsx
 * <FormulaTooltip
 *   content="计算公式：$E = mc^2$"
 *   renderFormula={renderLatex}
 * >
 *   <Card>公式卡片</Card>
 * </FormulaTooltip>
 * ```
 */
export const FormulaTooltip: React.FC<FormulaTooltipProps> = ({
  content,
  renderFormula,
  ...tooltipProps
}) => {
  const renderedContent = renderFormula && typeof content === 'string'
    ? renderFormula(content)
    : content

  return (
    <InfoTooltip content={renderedContent} {...tooltipProps} />
  )
}

// ============================================
// 快捷工具函数
// ============================================

/**
 * 为元素包裹 Tooltip 的快捷函数
 *
 * @example
 * ```tsx
 * const element = withTooltip(
 *   <span>内容</span>,
 *   "这是提示"
 * )
 * ```
 */
export function withTooltip(
  element: React.ReactNode,
  tooltip: React.ReactNode,
  props?: Partial<InfoTooltipProps>
): React.ReactElement {
  return (
    <InfoTooltip content={tooltip} {...props}>
      {element}
    </InfoTooltip>
  )
}

/**
 * 条件性地包裹 Tooltip
 * 只有当 tooltip 存在时才包裹
 *
 * @example
 * ```tsx
 * const element = conditionalTooltip(
 *   <span>内容</span>,
 *   someCondition ? "提示" : undefined
 * )
 * ```
 */
export function conditionalTooltip(
  element: React.ReactNode,
  tooltip: React.ReactNode | undefined,
  props?: Partial<InfoTooltipProps>
): React.ReactNode {
  if (!tooltip) {
    return element
  }
  return withTooltip(element, tooltip, props)
}
