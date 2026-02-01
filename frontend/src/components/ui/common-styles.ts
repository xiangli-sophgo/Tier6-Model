/**
 * 通用样式常量
 *
 * 统一管理常用的样式对象，避免在多个组件中重复定义
 */

import React from 'react'
import { colors } from '../../utils/design-tokens'

// ============================================
// 卡片样式
// ============================================

/**
 * 基础卡片样式
 */
export const baseCardStyle: React.CSSProperties = {
  background: colors.cardBg,
  borderRadius: 10,
  padding: 16,
  marginBottom: 12,
  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.06)',
  border: `1px solid ${colors.borderLight}`,
}

/**
 * Section 卡片样式（用于配置面板等）
 */
export const sectionCardStyle: React.CSSProperties = {
  ...baseCardStyle,
}

/**
 * 详情卡片包装样式
 */
export const detailWrapperStyle: React.CSSProperties = {
  background: colors.background,
  borderRadius: 8,
  padding: 16,
}

/**
 * 指标卡片样式（用于显示关键指标）
 */
export const metricCardStyle: React.CSSProperties = {
  background: colors.cardBg,
  border: `1px solid ${colors.borderLight}`,
  borderRadius: 8,
  padding: 12,
  transition: 'all 0.2s ease',
}

/**
 * 可点击卡片样式（带hover效果）
 */
export const clickableCardStyle = (selected: boolean = false): React.CSSProperties => ({
  ...metricCardStyle,
  cursor: 'pointer',
  border: selected ? `1px solid ${colors.interactive}` : `1px solid ${colors.borderLight}`,
  boxShadow: selected ? `0 0 0 2px ${colors.interactiveLight}` : undefined,
})

// ============================================
// 标题样式
// ============================================

/**
 * Section 标题样式
 */
export const sectionTitleStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: colors.text,
  marginBottom: 12,
  display: 'flex',
  alignItems: 'center',
  gap: 6,
}

/**
 * 子标题样式
 */
export const subTitleStyle: React.CSSProperties = {
  fontSize: 12,
  fontWeight: 500,
  color: colors.textSecondary,
  marginBottom: 8,
}

// ============================================
// 表格样式
// ============================================

/**
 * 基础表格样式
 */
export const tableStyle: React.CSSProperties = {
  width: '100%',
  borderCollapse: 'collapse',
  fontSize: 12,
}

/**
 * 表头单元格样式
 */
export const thStyle: React.CSSProperties = {
  padding: 8,
  textAlign: 'left',
  borderBottom: `1px solid ${colors.border}`,
  fontWeight: 600,
  color: colors.text,
  background: colors.background,
}

/**
 * 表格单元格样式
 */
export const tdStyle: React.CSSProperties = {
  padding: 8,
  borderBottom: `1px solid ${colors.borderLight}`,
  color: colors.text,
}

// ============================================
// 布局样式
// ============================================

/**
 * 配置行样式（标签 + 输入控件）
 */
export const configRowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: 10,
}

/**
 * Flex 容器样式（常用的居中布局）
 */
export const flexCenterStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}

/**
 * Flex 间距样式（space-between）
 */
export const flexBetweenStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
}

// ============================================
// 按钮样式
// ============================================

/**
 * 主按钮样式
 */
export const primaryButtonStyle: React.CSSProperties = {
  background: colors.primary,
  color: '#fff',
  border: 'none',
  borderRadius: 6,
  padding: '8px 16px',
  cursor: 'pointer',
  fontSize: 13,
  fontWeight: 500,
  transition: 'all 0.2s ease',
}

/**
 * 次要按钮样式
 */
export const secondaryButtonStyle: React.CSSProperties = {
  background: colors.background,
  color: colors.text,
  border: `1px solid ${colors.border}`,
  borderRadius: 6,
  padding: '8px 16px',
  cursor: 'pointer',
  fontSize: 13,
  fontWeight: 500,
  transition: 'all 0.2s ease',
}

// ============================================
// 输入框样式
// ============================================

/**
 * 基础输入框样式
 */
export const inputStyle: React.CSSProperties = {
  border: `1px solid ${colors.border}`,
  borderRadius: 6,
  padding: '6px 12px',
  fontSize: 13,
  color: colors.text,
  background: colors.cardBg,
  transition: 'all 0.2s ease',
}

/**
 * 小型输入框样式
 */
export const inputSmallStyle: React.CSSProperties = {
  ...inputStyle,
  padding: '4px 8px',
  fontSize: 12,
}

// ============================================
// Badge/标签样式
// ============================================

/**
 * Badge 样式工厂函数
 */
export const createBadgeStyle = (
  backgroundColor: string,
  color: string
): React.CSSProperties => ({
  display: 'inline-block',
  padding: '2px 8px',
  borderRadius: 4,
  fontSize: 11,
  fontWeight: 500,
  backgroundColor,
  color,
  border: 'none',
})

/**
 * 预设的 Badge 样式
 */
export const badgeStyles = {
  success: createBadgeStyle(colors.successLight, colors.success),
  warning: createBadgeStyle(colors.warningLight, colors.warning),
  error: createBadgeStyle(colors.errorLight, colors.error),
  info: createBadgeStyle(colors.interactiveLight, colors.interactive),
  default: createBadgeStyle(colors.background, colors.textSecondary),
}

// ============================================
// 键盘快捷键样式
// ============================================

/**
 * Kbd 标签样式（用于显示键盘快捷键）
 */
export const kbdStyle: React.CSSProperties = {
  background: colors.backgroundDark,
  padding: '2px 6px',
  borderRadius: 4,
  fontFamily: 'monospace',
  fontSize: 11,
  color: colors.textSecondary,
  border: `1px solid ${colors.border}`,
}

// ============================================
// 分隔线样式
// ============================================

/**
 * 水平分隔线样式
 */
export const dividerStyle: React.CSSProperties = {
  height: 1,
  background: colors.borderLight,
  border: 'none',
  margin: '12px 0',
}

/**
 * 垂直分隔线样式
 */
export const verticalDividerStyle: React.CSSProperties = {
  width: 1,
  background: colors.borderLight,
  border: 'none',
  margin: '0 12px',
}

// ============================================
// 动画样式
// ============================================

/**
 * 淡入淡出过渡
 */
export const fadeTransition: React.CSSProperties = {
  transition: 'opacity 0.2s ease',
}

/**
 * 滑动过渡
 */
export const slideTransition: React.CSSProperties = {
  transition: 'transform 0.2s ease',
}

/**
 * 全局过渡
 */
export const allTransition: React.CSSProperties = {
  transition: 'all 0.2s ease',
}
