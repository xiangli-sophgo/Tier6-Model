/**
 * LaTeX 公式渲染组件
 * 使用 react-katex 渲染数学公式
 * 参考 Notion 的简洁设计风格
 */

import React from 'react'
import { Tooltip } from 'antd'
import 'katex/dist/katex.min.css'
import katex from 'katex'

// ============================================
// 基础公式组件
// ============================================

interface FormulaProps {
  /** LaTeX 公式字符串 */
  tex: string
  /** 是否为块级显示 */
  block?: boolean
}

/**
 * 公式渲染组件 - 使用 katex.renderToString 确保正确渲染
 */
export const Formula: React.FC<FormulaProps> = ({ tex, block = false }) => {
  const html = React.useMemo(() => {
    try {
      return katex.renderToString(tex, {
        displayMode: block,
        throwOnError: false,
        strict: false,
      })
    } catch (e) {
      console.error('KaTeX render error:', e)
      return tex
    }
  }, [tex, block])

  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

// ============================================
// 公式卡片组件 - 主公式展示
// ============================================

interface FormulaCardProps {
  /** 公式标题 */
  title: string
  /** LaTeX 公式 */
  tex: string
  /** 公式说明 */
  description?: string
  /** 计算结果 */
  result?: string | number
  /** 结果单位 */
  unit?: string
  /** 结果颜色 */
  resultColor?: string
}

// 小节标题样式
const sectionTitleStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: '#374151',
  marginBottom: 10,
}

export const FormulaCard: React.FC<FormulaCardProps> = ({
  title,
  tex,
  description,
  result,
  unit,
  resultColor = '#5e6ad2',
}) => {
  const html = React.useMemo(() => {
    try {
      return katex.renderToString(tex, {
        displayMode: true,
        throwOnError: false,
        strict: false,
      })
    } catch (e) {
      console.error('KaTeX render error:', e)
      return tex
    }
  }, [tex])

  return (
    <div style={{ marginBottom: 20 }}>
      {title && (
        <div style={sectionTitleStyle}>{title}</div>
      )}
      <div style={{
        background: '#fafbfc',
        borderRadius: 8,
        padding: '16px 20px',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 16,
        }}>
          <div
            style={{
              fontSize: 18,
              color: '#1f2937',
              lineHeight: 1.6,
              flex: 1,
            }}
            dangerouslySetInnerHTML={{ __html: html }}
          />
          {result !== undefined && (
            <div style={{
              textAlign: 'right',
              flexShrink: 0,
            }}>
              <span style={{
                fontSize: 24,
                fontWeight: 700,
                color: resultColor,
              }}>
                {typeof result === 'number' ? result.toLocaleString() : result}
              </span>
              {unit && (
                <span style={{
                  fontSize: 14,
                  color: '#6b7280',
                  marginLeft: 4,
                }}>
                  {unit}
                </span>
              )}
            </div>
          )}
        </div>
        {description && (
          <div style={{
            fontSize: 12,
            color: '#9ca3af',
            marginTop: 8,
            lineHeight: 1.5,
          }}>
            {description}
          </div>
        )}
      </div>
    </div>
  )
}

// ============================================
// 变量定义组件
// ============================================

interface VariableItem {
  /** 变量符号 (LaTeX) */
  symbol: string
  /** 变量名称 */
  name: string
  /** 变量说明 */
  description?: string
  /** 当前值 */
  value?: string | number
  /** 单位 */
  unit?: string
}

interface VariableListProps {
  /** 变量列表 */
  variables: VariableItem[]
  /** 标题 */
  title?: string
}

// 内联公式渲染辅助函数
const renderInlineMath = (tex: string): string => {
  try {
    return katex.renderToString(tex, {
      displayMode: false,
      throwOnError: false,
      strict: false,
    })
  } catch (e) {
    console.error('KaTeX render error:', e)
    return tex
  }
}

// 渲染包含公式的文本，公式用 $...$ 包裹
const renderTextWithMath = (text: string): React.ReactNode => {
  const parts = text.split(/(\$[^$]+\$)/g)
  return parts.map((part, i) => {
    if (part.startsWith('$') && part.endsWith('$')) {
      const tex = part.slice(1, -1)
      return (
        <span
          key={i}
          dangerouslySetInnerHTML={{ __html: renderInlineMath(tex) }}
        />
      )
    }
    return <span key={i}>{part}</span>
  })
}

export const VariableList: React.FC<VariableListProps> = ({
  variables,
  title = '变量说明',
}) => (
  <div style={{ marginBottom: 20 }}>
    <div style={sectionTitleStyle}>{title}</div>
    <div style={{
      display: 'flex',
      flexWrap: 'wrap',
      gap: 8,
    }}>
      {variables.map((v, i) => {
        const card = (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '4px 10px',
              background: '#f8fafc',
              borderRadius: 6,
              border: '1px solid #e5e7eb',
              cursor: v.description ? 'help' : 'default',
            }}
          >
            <span
              style={{
                color: '#5e6ad2',
                fontSize: 13,
              }}
              dangerouslySetInnerHTML={{ __html: renderInlineMath(v.symbol) }}
            />
            <span style={{ fontSize: 12, color: '#475569' }}>
              {v.name}
            </span>
          </div>
        )
        return v.description ? (
          <Tooltip
            key={i}
            title={renderTextWithMath(v.description)}
            overlayStyle={{ maxWidth: 400 }}
            overlayInnerStyle={{ whiteSpace: 'nowrap' }}
          >
            {card}
          </Tooltip>
        ) : (
          <React.Fragment key={i}>{card}</React.Fragment>
        )
      })}
    </div>
  </div>
)

// ============================================
// 计算步骤组件
// ============================================

interface CalculationStep {
  /** 步骤标签 */
  label: string
  /** 公式 (LaTeX) */
  formula?: string
  /** 计算结果 */
  value: string | number
  /** 单位 */
  unit?: string
  /** 说明 */
  note?: string
}

interface CalculationStepsProps {
  /** 计算步骤 */
  steps: CalculationStep[]
  /** 标题 */
  title?: string
}

export const CalculationSteps: React.FC<CalculationStepsProps> = ({
  steps,
  title = '计算过程',
}) => (
  <div style={{ marginBottom: 20 }}>
    <div style={sectionTitleStyle}>{title}</div>
    <div
      style={{
        background: '#fff',
        borderRadius: 8,
        border: '1px solid #e5e7eb',
        overflow: 'hidden',
      }}
    >
      {steps.map((step, i) => (
        <div
          key={i}
          style={{
            display: 'grid',
            gridTemplateColumns: '100px 1fr 110px',
            alignItems: 'center',
            padding: '10px 16px',
            borderBottom: i < steps.length - 1 ? '1px solid #f0f0f0' : 'none',
          }}
        >
          {/* 标签（支持LaTeX渲染） */}
          <span
            style={{ fontSize: 13, color: '#6b7280' }}
            dangerouslySetInnerHTML={{ __html: renderInlineMath(step.label) }}
          />

          {/* 公式（居中） */}
          <div style={{ textAlign: 'center' }}>
            {step.formula && (
              <span
                style={{ fontSize: 14, color: '#374151' }}
                dangerouslySetInnerHTML={{ __html: renderInlineMath(`\\displaystyle ${step.formula}`) }}
              />
            )}
          </div>

          {/* 结果（居中，包含单位） */}
          <div style={{ textAlign: 'center' }}>
            <span style={{ fontSize: 15, fontWeight: 600, color: '#1f2937' }}>
              {typeof step.value === 'number' ? step.value.toLocaleString() : step.value}
            </span>
            {step.unit && (
              <span style={{ fontSize: 11, color: '#9ca3af', marginLeft: 4 }}>
                {step.unit}
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  </div>
)

// ============================================
// 结果展示组件
// ============================================

interface ResultDisplayProps {
  /** 指标名称 */
  label: string
  /** 结果值 */
  value: string | number
  /** 单位 */
  unit?: string
  /** 主题颜色 */
  color?: string
}

export const ResultDisplay: React.FC<ResultDisplayProps> = ({
  label,
  value,
  unit,
  color = '#5e6ad2',
}) => (
  <div style={{
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '14px 18px',
    background: `${color}08`,
    borderRadius: 10,
    border: `1px solid ${color}20`,
  }}>
    <span style={{
      fontSize: 14,
      color: '#374151',
      fontWeight: 500,
    }}>
      {label}
    </span>
    <div>
      <span style={{
        fontSize: 24,
        fontWeight: 700,
        color: color,
      }}>
        {typeof value === 'number' ? value.toLocaleString() : value}
      </span>
      {unit && (
        <span style={{
          fontSize: 13,
          color: '#6b7280',
          marginLeft: 6,
        }}>
          {unit}
        </span>
      )}
    </div>
  </div>
)

export default Formula
