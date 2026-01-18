/**
 * BaseCard 基类组件
 * 统一 Card 样式，支持标题区域背景色区分和可折叠功能
 */

import React, { useState } from 'react'
import { DownOutlined, RightOutlined } from '@ant-design/icons'

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
  /** 自定义容器样式 */
  style?: React.CSSProperties
  /** 内容区域样式 */
  bodyStyle?: React.CSSProperties
  /** 是否可折叠 */
  collapsible?: boolean
  /** 默认是否展开（仅在 collapsible=true 时生效） */
  defaultExpanded?: boolean
  /** 受控展开状态 */
  expanded?: boolean
  /** 展开状态变化回调 */
  onExpandChange?: (expanded: boolean) => void
}

// 样式常量
const STYLES = {
  card: {
    background: '#fff',
    borderRadius: 12,
    border: '1px solid #e5e7eb',
    overflow: 'hidden',
  } as React.CSSProperties,
  header: {
    background: '#FAFAFA',
    padding: '14px 20px',
    borderBottom: '1px solid #E5E5E5',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  } as React.CSSProperties,
  headerCollapsible: {
    cursor: 'pointer',
    userSelect: 'none',
  } as React.CSSProperties,
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    flex: 1,
    minWidth: 0,
  } as React.CSSProperties,
  titleWrapper: {
    flex: 1,
    minWidth: 0,
  } as React.CSSProperties,
  title: {
    fontSize: 16,
    fontWeight: 600,
    color: '#1f2937',
    margin: 0,
    lineHeight: 1.4,
  } as React.CSSProperties,
  subtitle: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
    lineHeight: 1.4,
  } as React.CSSProperties,
  body: {
    padding: 20,
    background: '#fff',
  } as React.CSSProperties,
  collapseIcon: {
    fontSize: 12,
    color: '#9ca3af',
    transition: 'transform 0.2s',
    flexShrink: 0,
  } as React.CSSProperties,
  accentBar: {
    width: 3,
    borderRadius: 2,
    alignSelf: 'stretch',
    flexShrink: 0,
  } as React.CSSProperties,
}

export const BaseCard: React.FC<BaseCardProps> = ({
  title,
  subtitle,
  icon,
  extra,
  accentColor,
  children,
  style,
  bodyStyle,
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

  const headerStyle: React.CSSProperties = {
    ...STYLES.header,
    ...(collapsible ? STYLES.headerCollapsible : {}),
    ...(collapsible && !isExpanded ? { borderBottom: 'none' } : {}),
  }

  return (
    <div style={{ ...STYLES.card, ...style }}>
      {/* 标题区域 */}
      <div
        style={headerStyle}
        onClick={handleToggle}
      >
        <div style={STYLES.headerLeft}>
          {/* 可折叠图标 */}
          {collapsible && (
            <span style={STYLES.collapseIcon}>
              {isExpanded ? <DownOutlined /> : <RightOutlined />}
            </span>
          )}

          {/* 左侧彩色边条 */}
          {accentColor && (
            <div style={{ ...STYLES.accentBar, background: accentColor }} />
          )}

          {/* 图标 */}
          {icon && <span style={{ flexShrink: 0 }}>{icon}</span>}

          {/* 标题和副标题 */}
          <div style={STYLES.titleWrapper}>
            <div style={STYLES.title}>{title}</div>
            {subtitle && <div style={STYLES.subtitle}>{subtitle}</div>}
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
        <div style={{ ...STYLES.body, ...bodyStyle }}>
          {children}
        </div>
      )}
    </div>
  )
}

export default BaseCard
