/**
 * BaseCard 基类组件
 * 统一 Card 样式，支持标题区域背景色区分和可折叠功能
 */

import React, { useState } from 'react'
import { ChevronDown, Pencil } from 'lucide-react'
import * as CollapsiblePrimitive from '@radix-ui/react-collapsible'
import { cn } from '@/lib/utils'

export interface BaseCardProps {
  /** 标题 */
  title?: React.ReactNode
  /** 副标题（可选） */
  subtitle?: string
  /** 标题图标（可选） */
  icon?: React.ReactNode
  /** 右侧额外内容（可选） */
  extra?: React.ReactNode
  /** 标题区域主题色（可选，用于左侧边条）- 使用gradient时无效 */
  accentColor?: string
  /** 是否使用渐变背景（蓝色渐变，不显示竖线） */
  gradient?: boolean
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
  /** 内容区自定义样式（关键！用于覆盖默认padding等） */
  contentClassName?: string
  /** 无标题模式（用于完全自定义布局） */
  titleless?: boolean
  /** 毛玻璃效果 */
  glassmorphism?: boolean
  /** 折叠区计数显示 */
  collapsibleCount?: number
  /** 编辑按钮支持 */
  onEdit?: () => void
  /** 编辑按钮文字（默认"编辑"） */
  editLabel?: string
  /** 点击事件 */
  onClick?: () => void
}

export const BaseCard: React.FC<BaseCardProps> = ({
  title,
  subtitle,
  icon,
  extra,
  accentColor,
  gradient = false,
  children,
  className,
  style,
  bodyClassName,
  styles,
  collapsible = false,
  defaultExpanded = true,
  expanded: controlledExpanded,
  onExpandChange,
  contentClassName,
  titleless = false,
  glassmorphism = false,
  collapsibleCount,
  onEdit,
  editLabel = '编辑',
  onClick,
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

  // titleless模式：完全自定义布局
  if (titleless) {
    return (
      <div
        className={cn(
          'card rounded-lg border bg-white shadow-none hover:shadow-md transition-shadow duration-300',
          glassmorphism && 'bg-white/80 backdrop-blur-sm',
          onClick && 'cursor-pointer',
          className
        )}
        style={style}
        onClick={onClick}
      >
        {children}
      </div>
    )
  }

  // 折叠模式使用Radix UI Collapsible
  if (collapsible) {
    return (
      <CollapsiblePrimitive.Root
        open={isExpanded}
        onOpenChange={handleToggle}
        className={cn(
          'card overflow-hidden rounded-lg border bg-white shadow-none hover:shadow-md transition-all duration-300',
          gradient ? 'border-blue-200/50' : 'border-gray-200',
          glassmorphism && 'bg-white/80 backdrop-blur-sm',
          className
        )}
        style={style}
      >
        {/* 标题区域（可点击触发折叠） */}
        <CollapsiblePrimitive.Trigger asChild>
          <div
            className={cn(
              'flex items-center justify-between gap-2 border-b py-3 px-2.5 transition-colors cursor-pointer select-none w-full',
              gradient
                ? 'bg-gradient-to-r from-blue-50 to-white border-blue-100 hover:from-blue-100 hover:to-blue-50'
                : 'border-gray-100 hover:bg-gray-50',
              !isExpanded && 'border-b-0'
            )}
          >
            <div className="flex flex-1 items-center gap-2 min-w-0">
              {/* 可折叠图标 */}
              <ChevronDown
                className={cn(
                  'h-3.5 w-3.5 flex-shrink-0 text-gray-400 transition-transform duration-200',
                  !isExpanded && '-rotate-90'
                )}
              />

              {/* 左侧彩色边条（仅在非gradient模式下显示） */}
              {!gradient && accentColor && (
                <div
                  className="h-4 w-[3px] flex-shrink-0 rounded-sm"
                  style={{ background: accentColor }}
                />
              )}

              {/* 图标 */}
              {icon && <span className="flex-shrink-0">{icon}</span>}

              {/* 标题和副标题 */}
              <div className="flex flex-1 items-center gap-2 min-w-0">
                <div className="text-sm font-semibold leading-tight text-gray-700">
                  {title}
                  {collapsibleCount !== undefined && (
                    <span className="ml-1.5 text-sm text-gray-500 font-normal">
                      ({collapsibleCount})
                    </span>
                  )}
                </div>
                {subtitle && <div className="mt-0.5 text-xs leading-tight text-gray-500">{subtitle}</div>}
              </div>
            </div>

            {/* 右侧额外内容 */}
            <div className="flex items-center gap-2">
              {onEdit && (
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onEdit()
                  }}
                  className="flex items-center gap-1.5 px-2 py-1 text-xs text-blue-600 hover:bg-blue-50 rounded transition-colors"
                >
                  <Pencil className="h-3 w-3" />
                  {editLabel}
                </button>
              )}
              {extra && (
                <div onClick={(e) => e.stopPropagation()}>
                  {extra}
                </div>
              )}
            </div>
          </div>
        </CollapsiblePrimitive.Trigger>

        {/* 内容区域（带折叠动画） */}
        <CollapsiblePrimitive.Content className="overflow-hidden data-[state=open]:animate-collapsible-down data-[state=closed]:animate-collapsible-up">
          <div className={cn('p-2.5', bodyClassName, contentClassName)} style={styles?.body}>
            {children}
          </div>
        </CollapsiblePrimitive.Content>
      </CollapsiblePrimitive.Root>
    )
  }

  // 标准模式（非折叠）
  return (
    <div
      className={cn(
        'card overflow-hidden rounded-lg border bg-white shadow-none hover:shadow-md transition-all duration-300',
        gradient ? 'border-blue-200/50' : 'border-gray-200',
        glassmorphism && 'bg-white/80 backdrop-blur-sm',
        className
      )}
      style={style}
    >
      {/* 标题区域 */}
      <div
        className={cn(
          'flex items-center justify-between gap-2 border-b py-3 px-2.5',
          gradient
            ? 'bg-gradient-to-r from-blue-50 to-white border-blue-100'
            : 'border-gray-100'
        )}
      >
        <div className="flex flex-1 items-center gap-2 min-w-0">
          {/* 左侧彩色边条（仅在非gradient模式下显示） */}
          {!gradient && accentColor && (
            <div
              className="h-4 w-[3px] flex-shrink-0 rounded-sm"
              style={{ background: accentColor }}
            />
          )}

          {/* 图标 */}
          {icon && <span className="flex-shrink-0">{icon}</span>}

          {/* 标题和副标题 */}
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold leading-tight text-gray-700">{title}</div>
            {subtitle && <div className="mt-0.5 text-xs leading-tight text-gray-500">{subtitle}</div>}
          </div>
        </div>

        {/* 右侧额外内容 */}
        <div className="flex items-center gap-2">
          {onEdit && (
            <button
              onClick={onEdit}
              className="flex items-center gap-1.5 px-2 py-1 text-xs text-blue-600 hover:bg-blue-50 rounded transition-colors"
            >
              <Pencil className="h-3 w-3" />
              {editLabel}
            </button>
          )}
          {extra && <div>{extra}</div>}
        </div>
      </div>

      {/* 内容区域 */}
      <div className={cn('p-2.5', bodyClassName, contentClassName)} style={styles?.body}>
        {children}
      </div>
    </div>
  )
}

export default BaseCard
