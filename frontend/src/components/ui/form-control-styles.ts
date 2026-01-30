/**
 * 表单控件统一样式基类
 * 用于 Input、Select、Textarea 等组件保持一致的视觉风格
 */

/**
 * 基础样式 - 所有表单控件共用
 * 包含：尺寸、圆角、边框、背景、内边距、字体、阴影、过渡效果
 */
export const formControlBaseStyles = [
  // 布局和尺寸
  "flex h-9 w-full items-center",
  // 边框和圆角
  "rounded-lg border border-blue-200",
  // 背景
  "bg-white",
  // 内边距
  "px-3 py-1",
  // 字体
  "text-sm",
  // 阴影
  "shadow-xs",
  // 过渡效果
  "transition-colors",
].join(" ")

/**
 * 交互样式 - Hover 效果
 */
export const formControlHoverStyles = "hover:border-blue-300"

/**
 * 交互样式 - Focus 效果
 */
export const formControlFocusStyles = [
  "focus-visible:outline-none",
  "focus-visible:ring-2",
  "focus-visible:ring-blue-500",
  "focus-visible:ring-offset-0",
].join(" ")

/**
 * 禁用状态样式
 */
export const formControlDisabledStyles = [
  "disabled:cursor-not-allowed",
  "disabled:opacity-50",
].join(" ")

/**
 * 占位符样式
 */
export const formControlPlaceholderStyles = "placeholder:text-text-muted"

/**
 * 完整的表单控件样式（组合所有样式）
 */
export const formControlStyles = [
  formControlBaseStyles,
  formControlHoverStyles,
  formControlFocusStyles,
  formControlDisabledStyles,
  formControlPlaceholderStyles,
].join(" ")

/**
 * Select 专用样式（额外的下拉框特有样式）
 */
export const selectTriggerStyles = [
  formControlStyles,
  // Select 特有：下拉图标和文本处理
  "justify-between whitespace-nowrap",
  "[&>span]:line-clamp-1",
  // Select 特有：ring offset
  "ring-offset-bg-elevated",
].join(" ")
