/**
 * 参数遍历配置面板
 */

import React, { useState, useMemo } from 'react'
import { Plus, Trash2, AlertCircle, Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { NumberInput } from '@/components/ui/number-input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Alert, AlertDescription } from '@/components/ui/alert'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import type { SweepParam, SweepableParameter } from './sweepTypes'
import { BIND_GROUP_COLORS } from './sweepTypes'
import { getParameterDescription } from './parameterDescriptions'
import {
  calculateSweepValues,
  calculateTotalCombinationsWithBinding,
  validateSweepParams,
  getExistingBindGroups,
  getNextBindGroupId
} from './sweepHelpers'

interface SweepConfigPanelProps {
  // 可遍历参数列表
  sweepableParams: SweepableParameter[]
  // 已添加的参数
  sweepParams: SweepParam[]
  onSweepParamsChange: (params: SweepParam[]) => void
  // 基础配置摘要（只读显示）
  benchmarkName?: string
  topologyName?: string
}

export const SweepConfigPanel: React.FC<SweepConfigPanelProps> = ({
  sweepableParams,
  sweepParams,
  onSweepParamsChange,
  benchmarkName,
  topologyName,
}) => {
  // 搜索查询
  const [searchQuery, setSearchQuery] = useState('')

  // 添加参数到遍历列表
  const handleAddParameter = (paramKey: string) => {
    const paramMeta = sweepableParams.find(p => p.key === paramKey)
    if (!paramMeta) return

    // 检查是否已添加
    if (sweepParams.some(p => p.key === paramKey)) {
      return
    }

    const newParam: SweepParam = {
      key: paramMeta.key,
      label: paramMeta.label,
      start: paramMeta.defaultRange.min,
      end: paramMeta.defaultRange.max,
      step: paramMeta.defaultRange.step,
      values: calculateSweepValues(
        paramMeta.defaultRange.min,
        paramMeta.defaultRange.max,
        paramMeta.defaultRange.step
      ),
      currentValue: paramMeta.currentValue,
      unit: paramMeta.unit,
    }

    onSweepParamsChange([...sweepParams, newParam])
    // 清空搜索框
    setSearchQuery('')
  }

  // 删除参数
  const handleRemoveParameter = (key: string) => {
    onSweepParamsChange(sweepParams.filter(p => p.key !== key))
  }

  // 更新参数配置
  const handleUpdateParameter = (
    key: string,
    field: 'start' | 'end' | 'step',
    value: number
  ) => {
    onSweepParamsChange(
      sweepParams.map(p => {
        if (p.key !== key) return p

        const updated = { ...p, [field]: value }
        // 重新计算值列表
        updated.values = calculateSweepValues(updated.start, updated.end, updated.step)
        return updated
      })
    )
  }

  // 更新绑定组
  const handleUpdateBindGroup = (index: number, bindGroupId: string | undefined) => {
    onSweepParamsChange(
      sweepParams.map((p, idx) =>
        idx === index ? { ...p, bindGroupId } : p
      )
    )
  }

  // 获取已使用的绑定组
  const existingBindGroups = useMemo(
    () => getExistingBindGroups(sweepParams),
    [sweepParams]
  )

  // 计算总组合数（支持绑定）
  const totalCombinations = useMemo(
    () => calculateTotalCombinationsWithBinding(sweepParams),
    [sweepParams]
  )

  // 验证参数配置
  const validationErrors = useMemo(() => validateSweepParams(sweepParams), [sweepParams])

  // 过滤已添加的参数 + 搜索过滤
  const availableParams = useMemo(() => {
    const notAdded = sweepableParams.filter(
      p => !sweepParams.some(sp => sp.key === p.key)
    )

    if (!searchQuery.trim()) {
      return notAdded
    }

    // 搜索过滤
    const query = searchQuery.toLowerCase()
    return notAdded.filter(p => {
      return (
        p.key.toLowerCase().includes(query) ||
        p.label.toLowerCase().includes(query) ||
        p.currentValue.toString().includes(query) ||
        (p.unit && p.unit.toLowerCase().includes(query))
      )
    })
  }, [sweepableParams, sweepParams, searchQuery])

  return (
    <TooltipProvider>
      <div className="space-y-4">
      {/* 参数选择器（带搜索，参考 CrossRing 实现） */}
      <Select
        value=""
        onValueChange={(value) => {
          if (value) {
            handleAddParameter(value)
          }
        }}
      >
        <SelectTrigger className="w-full">
          <SelectValue placeholder="+ 添加遍历参数（可搜索）" />
        </SelectTrigger>
        <SelectContent
          className="w-full max-w-none"
          style={{ width: 'var(--radix-select-trigger-width)' }}
        >
          {/* 搜索输入框 */}
          <div className="px-2 py-1.5 border-b">
            <div className="relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
              <Input
                placeholder="搜索参数..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="h-8 pl-8 text-sm"
                onClick={(e) => e.stopPropagation()}
                onKeyDown={(e) => e.stopPropagation()}
              />
            </div>
            {searchQuery && (
              <div className="text-xs text-gray-500 mt-1">
                找到 {availableParams.length} 个参数
              </div>
            )}
          </div>

          {/* 参数列表（按配置文件结构分组） */}
          {availableParams.length === 0 ? (
            <div className="px-2 py-8 text-center text-sm text-gray-400">
              {searchQuery ? '未找到匹配的参数' : '没有更多可选参数'}
            </div>
          ) : (
            <>
              {/* Benchmark 配置文件 */}
              {(() => {
                const benchmarkParams = availableParams.filter(p =>
                  p.category === 'model' || p.category === 'inference'
                )
                if (benchmarkParams.length === 0) return null

                // 细分小节
                const sections = [
                  {
                    name: '基础参数',
                    params: benchmarkParams.filter(p =>
                      p.category === 'model' &&
                      !p.key.includes('moe_config') &&
                      !p.key.includes('mla_config') &&
                      !p.key.includes('attention')
                    )
                  },
                  {
                    name: '注意力配置',
                    params: benchmarkParams.filter(p =>
                      p.key.includes('attention') || p.key.includes('mla_config')
                    )
                  },
                  {
                    name: 'MoE 配置',
                    params: benchmarkParams.filter(p => p.key.includes('moe_config'))
                  },
                  {
                    name: '推理参数',
                    params: benchmarkParams.filter(p => p.category === 'inference')
                  }
                ].filter(s => s.params.length > 0)

                return (
                  <>
                    {/* Benchmark 配置文件标题 */}
                    <div className="sticky top-0 bg-white z-10 px-2 py-2 border-b-2 border-blue-500">
                      <div className="text-sm font-bold text-blue-700 flex items-center gap-2">
                        📄 Benchmark 配置文件
                        <span className="text-xs text-gray-500">({benchmarkParams.length}个参数)</span>
                      </div>
                    </div>

                    {/* 小节列表 */}
                    {sections.map((section, idx) => (
                      <React.Fragment key={section.name}>
                        {/* 小节标题 */}
                        <div className="px-3 py-1.5 bg-gray-50 border-b border-gray-200">
                          <div className="text-[12px] font-medium text-gray-600 flex items-center gap-2">
                            <span>▸</span>
                            {section.name}
                          </div>
                        </div>

                        {/* 参数列表 */}
                        {section.params.map(param => (
                          <SelectItem key={param.key} value={param.key} className="w-full">
                            <div className="flex items-center justify-between w-full gap-4 py-1 pl-6">
                              <div className="flex-1 min-w-0 flex items-center gap-2">
                                <span className="text-gray-400 text-xs">•</span>
                                <span className="font-medium text-[13px]">{param.label}</span>
                                {param.unit && (
                                  <span className="text-gray-400 text-[11px]">({param.unit})</span>
                                )}
                              </div>
                              <span className="text-gray-500 text-[11px] shrink-0 font-mono">
                                {param.currentValue}
                              </span>
                            </div>
                          </SelectItem>
                        ))}
                      </React.Fragment>
                    ))}
                  </>
                )
              })()}

              {/* 拓扑配置文件 */}
              {(() => {
                const topologyParams = availableParams.filter(p =>
                  p.category === 'hardware' || p.category === 'parallelism' || p.category === 'topology'
                )
                if (topologyParams.length === 0) return null

                // 提取芯片参数并按芯片类型分组
                const chipParams = topologyParams.filter(p =>
                  p.key.includes('chips.') || p.key.includes('hardware.chips') ||
                  (p.category === 'hardware' && !p.key.includes('interconnect'))
                )
                const chipTypes = new Set<string>()
                chipParams.forEach(p => {
                  const match = p.key.match(/chips\.([^.]+)\./)
                  if (match) chipTypes.add(match[1])
                })

                // 收集已分配的参数
                const assignedParams = new Set<string>()

                // 细分小节
                const sections: Array<{ name: string; params: SweepableParameter[] }> = []

                // 互联拓扑
                const interconnectParams = topologyParams.filter(p =>
                  (p.key.startsWith('topology.') || p.key.includes('interconnect')) &&
                  !p.key.includes('chips') &&
                  !p.key.includes('comm_latency')
                )
                if (interconnectParams.length > 0) {
                  sections.push({ name: '互联拓扑', params: interconnectParams })
                  interconnectParams.forEach(p => assignedParams.add(p.key))
                }

                // 为每种芯片类型创建一个小节
                Array.from(chipTypes).forEach(chipType => {
                  const params = chipParams.filter(p => p.key.includes(`chips.${chipType}.`))
                  if (params.length > 0) {
                    sections.push({ name: `芯片参数: ${chipType}`, params })
                    params.forEach(p => assignedParams.add(p.key))
                  }
                })

                // 其他芯片参数（不属于特定芯片类型的）
                const otherChipParams = chipParams.filter(p => !assignedParams.has(p.key))
                if (otherChipParams.length > 0) {
                  sections.push({ name: '芯片参数', params: otherChipParams })
                  otherChipParams.forEach(p => assignedParams.add(p.key))
                }

                // 通信延迟
                const commParams = topologyParams.filter(p => p.key.includes('comm_latency'))
                if (commParams.length > 0) {
                  sections.push({ name: '通信延迟', params: commParams })
                  commParams.forEach(p => assignedParams.add(p.key))
                }

                // 并行策略
                const parallelismParams = topologyParams.filter(p => p.category === 'parallelism')
                if (parallelismParams.length > 0) {
                  sections.push({ name: '并行策略', params: parallelismParams })
                  parallelismParams.forEach(p => assignedParams.add(p.key))
                }

                // 其他未分类的参数
                const unassignedParams = topologyParams.filter(p => !assignedParams.has(p.key))
                if (unassignedParams.length > 0) {
                  sections.push({ name: '其他参数', params: unassignedParams })
                }

                return (
                  <>
                    {/* 拓扑配置文件标题 */}
                    <div className="sticky top-0 bg-white z-10 px-2 py-2 border-b-2 border-green-500 mt-2">
                      <div className="text-sm font-bold text-green-700 flex items-center gap-2">
                        🏗️ 拓扑配置文件
                        <span className="text-xs text-gray-500">({topologyParams.length}个参数)</span>
                      </div>
                    </div>

                    {/* 小节列表 */}
                    {sections.map((section, idx) => (
                      <React.Fragment key={section.name}>
                        {/* 小节标题 */}
                        <div className="px-3 py-1.5 bg-gray-50 border-b border-gray-200">
                          <div className="text-[12px] font-medium text-gray-600 flex items-center gap-2">
                            <span>▸</span>
                            {section.name}
                          </div>
                        </div>

                        {/* 参数列表 */}
                        {section.params.map(param => (
                          <SelectItem key={param.key} value={param.key} className="w-full">
                            <div className="flex items-center justify-between w-full gap-4 py-1 pl-6">
                              <div className="flex-1 min-w-0 flex items-center gap-2">
                                <span className="text-gray-400 text-xs">•</span>
                                <span className="font-medium text-[13px]">{param.label}</span>
                                {param.unit && (
                                  <span className="text-gray-400 text-[11px]">({param.unit})</span>
                                )}
                              </div>
                              <span className="text-gray-500 text-[11px] shrink-0 font-mono">
                                {param.currentValue}
                              </span>
                            </div>
                          </SelectItem>
                        ))}
                      </React.Fragment>
                    ))}
                  </>
                )
              })()}
            </>
          )}
        </SelectContent>
      </Select>

      {/* 已添加参数列表 */}
      {sweepParams.length > 0 && (
        <div className="border rounded-lg overflow-hidden">
          <table className="w-full text-[13px]">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="px-3 py-2 text-left font-medium text-gray-600">参数</th>
                <th className="px-3 py-2 text-left font-medium text-gray-600">起始值</th>
                <th className="px-3 py-2 text-left font-medium text-gray-600">结束值</th>
                <th className="px-3 py-2 text-left font-medium text-gray-600">步长</th>
                <th className="px-3 py-2 text-center font-medium text-gray-600">值数量</th>
                <th className="px-3 py-2 text-center font-medium text-gray-600">绑定组</th>
                <th className="px-3 py-2 text-center font-medium text-gray-600">操作</th>
              </tr>
            </thead>
            <tbody>
              {sweepParams.map((param, idx) => (
                <tr
                  key={param.key}
                  className="border-b last:border-0"
                  style={{
                    backgroundColor: param.bindGroupId
                      ? BIND_GROUP_COLORS[param.bindGroupId]
                      : 'transparent',
                  }}
                >
                  <td className="px-3 py-2">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="cursor-help">
                          <div className="font-medium text-gray-700">{param.label}</div>
                          {param.unit && (
                            <div className="text-xs text-gray-400">{param.unit}</div>
                          )}
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-sm">
                          {getParameterDescription(param.key) || param.key}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </td>
                  <td className="px-3 py-2">
                    <NumberInput
                      value={param.start}
                      onChange={(value) =>
                        handleUpdateParameter(param.key, 'start', value || 0)
                      }
                      className="w-24"
                      size="sm"
                    />
                  </td>
                  <td className="px-3 py-2">
                    <NumberInput
                      value={param.end}
                      onChange={(value) =>
                        handleUpdateParameter(param.key, 'end', value || 0)
                      }
                      className="w-24"
                      size="sm"
                    />
                  </td>
                  <td className="px-3 py-2">
                    <NumberInput
                      value={param.step}
                      onChange={(value) =>
                        handleUpdateParameter(param.key, 'step', value || 1)
                      }
                      min={0.001}
                      className="w-24"
                      size="sm"
                    />
                  </td>
                  <td className="px-3 py-2 text-center">
                    <span className="inline-block px-2 py-0.5 bg-blue-50 text-blue-700 rounded text-xs font-medium">
                      {param.values.length}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-center">
                    <Select
                      value={param.bindGroupId || 'none'}
                      onValueChange={(value) => handleUpdateBindGroup(idx, value === 'none' ? undefined : value)}
                    >
                      <SelectTrigger className="w-24 h-8">
                        <SelectValue placeholder="无" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">无绑定</SelectItem>
                        {existingBindGroups.map(groupId => (
                          <SelectItem key={groupId} value={groupId}>
                            <div className="flex items-center gap-2">
                              <div
                                className="w-3 h-3 rounded"
                                style={{ backgroundColor: BIND_GROUP_COLORS[groupId] }}
                              />
                              组 {groupId}
                            </div>
                          </SelectItem>
                        ))}
                        <SelectItem value={getNextBindGroupId(existingBindGroups)}>
                          <div className="flex items-center gap-1">
                            <Plus className="h-3 w-3" />
                            新建组 {getNextBindGroupId(existingBindGroups)}
                          </div>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </td>
                  <td className="px-3 py-2 text-center">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleRemoveParameter(param.key)}
                      className="h-7 w-7 p-0"
                    >
                      <Trash2 className="h-3.5 w-3.5 text-red-500" />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* 组合数预览 */}
      {sweepParams.length > 0 && (
        <div className="flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="text-[13px] text-gray-600">
            总组合数:{' '}
            <span className="font-bold text-blue-700 text-lg ml-1">
              {totalCombinations}
            </span>
          </div>
          {totalCombinations > 100 && (
            <div className="text-xs text-orange-600 flex items-center">
              <AlertCircle className="h-3.5 w-3.5 mr-1" />
              组合数较多，评估耗时较长
            </div>
          )}
        </div>
      )}

      {/* 验证错误提示 */}
      {validationErrors.length > 0 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <ul className="list-disc pl-4 space-y-1">
              {validationErrors.map((error, idx) => (
                <li key={idx} className="text-[13px]">{error}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {/* 空状态提示 */}
      {sweepParams.length === 0 && (
        <div className="text-center py-6 text-gray-400">
          <Search className="h-8 w-8 mx-auto mb-2 opacity-30" />
          <p className="text-sm">使用上方搜索框查找并添加要遍历的参数</p>
        </div>
      )}
    </div>
    </TooltipProvider>
  )
}
