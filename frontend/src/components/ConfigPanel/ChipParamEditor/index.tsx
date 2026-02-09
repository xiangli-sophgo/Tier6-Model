/**
 * ChipParamEditor - 通用芯片参数编辑器组件
 *
 * 动态渲染芯片配置的所有字段，支持可编辑/只读模式切换
 * 不包含预设管理逻辑，专注于字段编辑
 *
 * 用于 TopologyInfoCard 和其他需要编辑单个芯片配置的场景
 */

import React, { useCallback } from 'react'
import { Input } from '@/components/ui/input'
import { NumberInput } from '@/components/ui/number-input'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { BaseCard } from '@/components/common/BaseCard'
import type { ChipPreset } from '@/types/math_model'
import {
  isPlainObject,
  isPathModified,
} from '@/utils/nestedObjectEditor'

// ==================== Props ====================

interface ChipParamEditorProps {
  /** 芯片名称（用于标题显示） */
  chipName: string
  /** 自定义标题（如果提供，将覆盖默认标题） */
  title?: React.ReactNode
  /** 当前芯片参数 */
  chipParams: ChipPreset
  /** 原始芯片参数（用于修改追踪） */
  originalParams?: ChipPreset | null
  /** 是否可编辑 */
  isEditable: boolean
  /** 参数变更回调 */
  onParamChange?: (path: string, value: unknown) => void
  /** 展开状态 */
  expanded?: boolean
  /** 展开状态变更回调 */
  onExpandChange?: () => void
}

// ==================== Section 标题映射 ====================

const SECTION_LABELS: Record<string, string> = {
  basic: '基础参数',
  cores: '核心配置',
  compute_units: '计算单元',
  memory: '内存配置',
  dma_engines: 'DMA 配置',
  interconnect: '片内互联',
}

// ==================== 格式化工具 ====================

const formatNumber = (value: number | undefined, decimals = 2): string => {
  if (value === undefined || value === null) return '-'
  if (Number.isInteger(value)) return String(value)
  return value.toFixed(decimals)
}

// ==================== Component ====================

export const ChipParamEditor: React.FC<ChipParamEditorProps> = ({
  chipName,
  title,
  chipParams,
  originalParams,
  isEditable,
  onParamChange,
  expanded = false,
  onExpandChange,
}) => {
  // 通用字段更新
  const updateField = useCallback((path: string, val: unknown) => {
    if (!onParamChange) return
    onParamChange(path, val)
  }, [onParamChange])

  // 修改检测
  const isFieldModified = useCallback((path: string): boolean => {
    if (!originalParams) return false
    return isPathModified(originalParams, chipParams, path)
  }, [originalParams, chipParams])

  // ==================== 渲染函数 ====================

  const modBadge = (path: string) =>
    isFieldModified(path)
      ? <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
      : null

  /** 渲染单个叶子字段 */
  const renderField = (key: string, val: unknown, path: string) => {
    const modified = isFieldModified(path)

    // 只读模式
    if (!isEditable) {
      let displayValue: React.ReactNode = '-'
      if (typeof val === 'boolean') {
        displayValue = val ? 'true' : 'false'
      } else if (typeof val === 'string') {
        displayValue = val || '-'
      } else if (typeof val === 'number') {
        displayValue = formatNumber(val)
      }

      return (
        <div key={path} className="flex flex-col gap-0.5 py-1.5 px-2.5 bg-gray-50/50 rounded border border-gray-100">
          <span className="text-[10px] text-gray-500">{key}</span>
          <span className="text-xs font-medium text-gray-800 break-all">{displayValue}</span>
        </div>
      )
    }

    // 可编辑模式
    return (
      <div key={path} className={`p-2 rounded ${modified ? 'bg-blue-50/50' : ''}`}>
        <div className="mb-1 flex items-center gap-1.5">
          <span className="text-[13px] text-gray-600 font-mono">{key}</span>
          {modBadge(path)}
        </div>
        {typeof val === 'boolean' ? (
          <Select value={val ? 'true' : 'false'} onValueChange={(v) => updateField(path, v === 'true')}>
            <SelectTrigger className="h-7 w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="true">true</SelectItem>
              <SelectItem value="false">false</SelectItem>
            </SelectContent>
          </Select>
        ) : typeof val === 'string' ? (
          <Input value={val} onChange={(e) => updateField(path, e.target.value)} className="h-7" />
        ) : typeof val === 'number' && val !== 0 && (Math.abs(val) < 0.001 || Math.abs(val) >= 1e7) ? (
          <Input
            value={val.toExponential().replace(/\.?0+e/, 'e')}
            onChange={(e) => {
              const parsed = Number(e.target.value)
              if (e.target.value !== '' && !isNaN(parsed)) updateField(path, parsed)
            }}
            className="h-7 font-mono"
            placeholder="如 1e-6"
          />
        ) : (
          <NumberInput value={val as number | undefined} onChange={(v) => updateField(path, v)} className="h-7" />
        )}
      </div>
    )
  }

  /** 递归渲染一个对象节点: 叶子字段排列为 grid，子对象递归展开 */
  const renderNode = (obj: Record<string, unknown>, parentPath: string): React.ReactNode => {
    const leafKeys: string[] = []
    const objKeys: string[] = []

    for (const k of Object.keys(obj)) {
      if (isPlainObject(obj[k])) objKeys.push(k)
      else leafKeys.push(k)
    }

    return (
      <>
        {leafKeys.length > 0 && (
          <div className={`grid ${isEditable ? 'grid-cols-3' : 'grid-cols-4'} gap-2`}>
            {leafKeys.map(k => renderField(k, obj[k], parentPath ? `${parentPath}.${k}` : k))}
          </div>
        )}
        {objKeys.map((k, idx) => {
          const childPath = parentPath ? `${parentPath}.${k}` : k
          const needSep = idx > 0 || leafKeys.length > 0
          return (
            <div key={childPath}>
              <div className={`${needSep ? 'mt-2 mb-1 border-t border-dashed border-gray-200 pt-2' : 'mb-1'}`}>
                <span className="text-xs text-gray-500 font-mono">{k}</span>
              </div>
              {renderNode(obj[k] as Record<string, unknown>, childPath)}
            </div>
          )
        })}
      </>
    )
  }

  // 分离顶层: 叶子字段归入 "基础参数", 对象字段各自建 card
  const asRecord = chipParams as unknown as Record<string, unknown>
  const topLeafKeys = Object.keys(asRecord).filter(k => k !== 'name' && !isPlainObject(asRecord[k]))
  const topObjKeys = Object.keys(asRecord).filter(k => k !== 'name' && isPlainObject(asRecord[k]))

  return (
    <BaseCard
      collapsible
      gradient
      title={title || `芯片参数: ${chipName}`}
      expanded={expanded}
      onExpandChange={onExpandChange}
      contentClassName="p-2"
    >
      <div className="space-y-2">
        {/* 基础参数 (顶层叶子字段) */}
        {topLeafKeys.length > 0 && (
          <BaseCard
            collapsible
            gradient
            title="基础参数"
            defaultExpanded={false}
            contentClassName="p-2"
          >
            <div className={`grid ${isEditable ? 'grid-cols-3' : 'grid-cols-4'} gap-2`}>
              {topLeafKeys.map(k => renderField(k, asRecord[k], k))}
            </div>
          </BaseCard>
        )}

        {/* 各顶层对象节点 - 每个都是独立的可折叠 card */}
        {topObjKeys.map((sectionKey) => (
          <BaseCard
            key={sectionKey}
            collapsible
            gradient
            title={SECTION_LABELS[sectionKey] || sectionKey}
            defaultExpanded={false}
            contentClassName="p-2"
          >
            {renderNode(asRecord[sectionKey] as Record<string, unknown>, sectionKey)}
          </BaseCard>
        ))}
      </div>
    </BaseCard>
  )
}

export default ChipParamEditor
