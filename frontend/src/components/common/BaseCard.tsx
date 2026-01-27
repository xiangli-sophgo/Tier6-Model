/**
 * BaseCard 基类组件
 * 统一 Card 样式，支持标题区域背景色区分和可折叠功能
 */

import React, { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface BaseCardProps {
  /** 标题 */
  title: React.ReactNode
  /** 副标题（可选） */
  subtitle?: string
  /** 标题图标（可选） */
  icon?: React.ReactNode
  /** 右侧额外内容（可选） */
  extra?: React.ReactNode
  /** 标题区域主题色（可选，用于左侧边条） */
  accentColor?: string
  /** 内容区域 */
  children: React.ReactNode
  /** 自定义容器类名 */
  className?: string
  /** 自定义容器样式（兼容旧API） */
  style?: React.CSSProperties
  /** 内容区域类名 */
  bodyClassName?: string
  /** 内容区域样式（兼容旧API） */
  styles?: {
    body?: React.CSSProperties
  }
  /** 是否可折叠 */
  collapsible?: boolean
  /** 默认是否展开（仅在 collapsible=true 时生效） */
  defaultExpanded?: boolean
  /** 受控展开状态 */
  expanded?: boolean
  /** 展开状态变化回调 */
  onExpandChange?: (expanded: boolean) => void
}

export const BaseCard: React.FC<BaseCardProps> = ({
  title,
  subtitle,
  icon,
  extra,
  accentColor,
  children,
  className,
  style,
  bodyClassName,
  styles,
  collapsible = false,
  defaultExpanded = true,
  expanded: controlledExpanded,
  onExpandChange,
}) => {
  const [internalExpanded, setInternalExpanded] = useState(defaultExpanded)

  // 支持受控和非受控模式
  const isExpanded = controlledExpanded !== undefined ? controlledExpanded : internalExpanded

  const handleToggle = () => {
    if (!collapsible) return
    const newExpanded = !isExpanded
    setInternalExpanded(newExpanded)
    onExpandChange?.(newExpanded)
  }

  return (
    <div className={cn('overflow-hidden rounded-lg border border-border bg-bg-elevated', className)} style={style}>
      {/* 标题区域 */}
      <div
        className={cn(
          'flex items-center justify-between gap-3 border-b border-border bg-[#FAFAFA] px-5 py-3.5',
          collapsible && 'cursor-pointer select-none',
          collapsible && !isExpanded && 'border-b-0'
        )}
        onClick={handleToggle}
      >
        <div className="flex flex-1 items-center gap-2.5 min-w-0">
          {/* 可折叠图标 */}
          {collapsible && (
            <span className="flex-shrink-0 text-xs text-[#9ca3af] transition-transform">
              {isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            </span>
          )}

          {/* 左侧彩色边条 */}
          {accentColor && (
            <div
              className="h-full w-[3px] flex-shrink-0 self-stretch rounded-sm"
              style={{ background: accentColor }}
            />
          )}

          {/* 图标 */}
          {icon && <span className="flex-shrink-0">{icon}</span>}

          {/* 标题和副标题 */}
          <div className="flex-1 min-w-0">
            <div className="text-base font-semibold leading-tight text-[#1f2937]">{title}</div>
            {subtitle && <div className="mt-0.5 text-[13px] leading-tight text-[#6b7280]">{subtitle}</div>}
          </div>
        </div>

        {/* 右侧额外内容 */}
        {extra && (
          <div onClick={e => e.stopPropagation()}>
            {extra}
          </div>
        )}
      </div>

      {/* 内容区域 */}
      {(!collapsible || isExpanded) && (
        <div className={cn('bg-white p-5', bodyClassName)} style={styles?.body}>
          {children}
        </div>
      )}
    </div>
  )
}

export default BaseCard
