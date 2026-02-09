/**
 * ModelPresetEditor - 模型预设编辑器
 *
 * 基于 ModelPreset 类型的模型配置编辑面板，
 * 动态渲染配置中实际存在的所有字段，
 * 支持预设加载、字段编辑、修改追踪、保存/另存为/重新加载功能。
 */

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { Save, Copy, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { NumberInput } from '@/components/ui/number-input'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import { BaseCard } from '@/components/common/BaseCard'
import type { ModelPreset } from '@/types/math_model'
import { getModelPresets, getModelPreset, updateModelPreset, saveModelPreset } from '@/api/math_model'

// ============================================
// Props
// ============================================

interface ModelPresetEditorProps {
  value: ModelPreset
  onChange: (config: ModelPreset) => void
  /** 当编辑器内部参数相对于加载的预设发生变化时回调 */
  onParamsModified?: (modified: boolean) => void
}

// 特性模块 key 和嵌套对象 key，从基础参数中排除
const FEATURE_KEYS = new Set(['MoE', 'MLA', 'DSA', 'NSA', 'RoPE'])
const SKIP_BASIC_KEYS = new Set(['name', ...FEATURE_KEYS])

// ============================================
// 组件
// ============================================

export const ModelPresetEditor: React.FC<ModelPresetEditorProps> = ({ value, onChange, onParamsModified }) => {
  const [presetList, setPresetList] = useState<Array<{ name: string; config: ModelPreset }>>([])
  const [selectedPreset, setSelectedPreset] = useState<string>('')
  const originalRef = useRef<ModelPreset | null>(null)

  const [sections, setSections] = useState<Record<string, boolean>>({
    basic: false, MoE: false, MLA: false, DSA: false, NSA: false, RoPE: false,
  })

  const [saveAsOpen, setSaveAsOpen] = useState(false)
  const [saveAsName, setSaveAsName] = useState('')

  // 加载预设列表
  useEffect(() => {
    getModelPresets()
      .then(({ presets }) => {
        setPresetList(presets)
        if (presets.length === 0) return
        const lastUsed = localStorage.getItem('tier6_last_model_preset')
        const target = presets.find(p => p.name === lastUsed)
          || presets.find(p => p.name === value.name)
          || presets[0]
        setSelectedPreset(target.name)
        originalRef.current = JSON.parse(JSON.stringify(target.config))
        onChange({ ...target.config })
      })
      .catch(() => toast.error('加载模型预设列表失败'))
  }, [])

  // 检测外部推送的预设变更 (如 Benchmark 加载联动)，同步内部状态
  useEffect(() => {
    if (!value.name || presetList.length === 0) return
    if (value.name === selectedPreset) return
    // 外部改变了 name，同步内部选中预设和快照
    // 支持按 config.name 匹配（文件内 name 字段）或按 preset.name 匹配（文件名）
    const match = presetList.find(p => p.name === value.name)
      || presetList.find(p => p.config.name === value.name)
    if (match) {
      setSelectedPreset(match.name)
      originalRef.current = JSON.parse(JSON.stringify(match.config))
      // 用预设列表的干净配置替换外部推送的对象，确保 value 和 originalRef 一致
      onChange({ ...match.config })
    }
  }, [value.name, presetList])

  const handlePresetChange = useCallback((name: string) => {
    setSelectedPreset(name)
    localStorage.setItem('tier6_last_model_preset', name)
    const match = presetList.find(p => p.name === name)
    if (match) {
      originalRef.current = JSON.parse(JSON.stringify(match.config))
      onChange({ ...match.config })
    }
  }, [presetList, onChange])

  // 修改检测 (支持嵌套路径如 "moe.num_routed_experts")
  const isFieldModified = useCallback((path: string): boolean => {
    if (!originalRef.current) return false
    const getVal = (obj: unknown, p: string): unknown => {
      let cur: unknown = obj
      for (const k of p.split('.')) {
        if (cur == null || typeof cur !== 'object') return undefined
        cur = (cur as Record<string, unknown>)[k]
      }
      return cur
    }
    const a = getVal(value, path)
    const b = getVal(originalRef.current, path)
    return JSON.stringify(a) !== JSON.stringify(b)
  }, [value])

  const toggleSection = (key: string) => setSections(prev => ({ ...prev, [key]: !prev[key] }))

  // 特性模块
  const FEATURE_LABELS = { MoE: 'MoE', MLA: 'MLA', DSA: 'DSA', NSA: 'NSA' } as const
  const FEATURE_DEFAULTS: Record<string, unknown> = {
    MoE: { num_routed_experts: 8, num_activated_experts: 2, intermediate_size: value.intermediate_size, num_shared_experts: 0 },
    MLA: { q_lora_rank: 1536, kv_lora_rank: 512, qk_nope_head_dim: 128, qk_rope_head_dim: 64, v_head_dim: 128 },
    DSA: { num_index_heads: 64, index_head_dim: 128, topk_index: 2048 },
    NSA: { compress_layers: 32, compress_ratio: 16, select_length: 64, select_num: 16, window_size: 512 },
  }

  const toggleFeature = (feature: 'MoE' | 'MLA' | 'DSA' | 'NSA') => {
    if (value[feature]) {
      const { [feature]: _, ...rest } = value
      onChange(rest as ModelPreset)
    } else {
      const restored = originalRef.current?.[feature]
      const data = restored
        ? JSON.parse(JSON.stringify(restored))
        : FEATURE_DEFAULTS[feature]
      onChange({ ...value, [feature]: data })
    }
  }

  const isFeatureModified = useCallback((feature: 'MoE' | 'MLA' | 'DSA' | 'NSA'): boolean => {
    if (!originalRef.current) return false
    const orig = originalRef.current[feature]
    const curr = value[feature]
    if ((!orig) !== (!curr)) return true
    if (!orig || !curr) return false
    return JSON.stringify(orig) !== JSON.stringify(curr)
  }, [value])

  // 向父组件报告参数级修改状态 (同一预设内的编辑)
  const isAnyParamModified = useMemo(() => {
    if (!originalRef.current) return false
    return JSON.stringify(value) !== JSON.stringify(originalRef.current)
  }, [value])

  useEffect(() => {
    onParamsModified?.(isAnyParamModified)
  }, [isAnyParamModified, onParamsModified])

  // 保存
  const handleSave = async () => {
    if (!selectedPreset) { toast.error('请先选择一个预设'); return }
    try {
      await updateModelPreset(selectedPreset, value)
      originalRef.current = JSON.parse(JSON.stringify(value))
      toast.success(`已保存: ${selectedPreset}`)
    } catch (e: unknown) {
      toast.error(`保存失败: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const handleSaveAs = async () => {
    const name = saveAsName.trim()
    if (!name) { toast.error('请输入预设名称'); return }
    try {
      await saveModelPreset(name, { ...value, name })
      originalRef.current = JSON.parse(JSON.stringify({ ...value, name }))
      onChange({ ...value, name })
      const { presets } = await getModelPresets()
      setPresetList(presets)
      setSelectedPreset(name)
      setSaveAsOpen(false)
      setSaveAsName('')
      toast.success(`已另存为: ${name}`)
    } catch (e: unknown) {
      toast.error(`另存为失败: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const handleReload = useCallback(async () => {
    if (!selectedPreset) { toast.error('请先选择模型预设'); return }
    try {
      const cfg = await getModelPreset(selectedPreset)
      originalRef.current = JSON.parse(JSON.stringify(cfg))
      onChange(cfg)
      toast.success('已重新加载')
    } catch (e) {
      toast.error(`重新加载失败: ${e instanceof Error ? e.message : String(e)}`)
    }
  }, [selectedPreset, onChange])

  // ============================================
  // 通用字段渲染器
  // ============================================

  const modBadge = (path: string) =>
    isFieldModified(path)
      ? <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
      : null

  /** 渲染单个字段 (自动根据值类型选择控件) */
  const renderField = (
    key: string,
    val: unknown,
    path: string,
    onUpdate: (key: string, val: unknown) => void,
  ) => {
    const modified = isFieldModified(path)

    return (
      <div key={path} className={`p-2 rounded ${modified ? 'bg-blue-50/50' : ''}`}>
        <div className="mb-1 flex items-center gap-1.5">
          <span className="text-[13px] text-gray-600 font-mono">{key}</span>
          {modBadge(path)}
        </div>
        {typeof val === 'boolean' ? (
          <Select value={val ? 'true' : 'false'} onValueChange={(v) => onUpdate(key, v === 'true')}>
            <SelectTrigger className="h-7 w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="true">true</SelectItem>
              <SelectItem value="false">false</SelectItem>
            </SelectContent>
          </Select>
        ) : typeof val === 'string' ? (
          <Input value={val} onChange={(e) => onUpdate(key, e.target.value)} className="h-7" />
        ) : typeof val === 'number' && val !== 0 && (Math.abs(val) < 0.001 || Math.abs(val) >= 1e7) ? (
          // 极小/极大数字用科学计数法文本输入
          <Input
            value={val.toExponential().replace(/\.?0+e/, 'e')}
            onChange={(e) => {
              const parsed = Number(e.target.value)
              if (e.target.value !== '' && !isNaN(parsed)) onUpdate(key, parsed)
            }}
            className="h-7 font-mono"
            placeholder="如 1e-6"
          />
        ) : (
          <NumberInput value={val as number | undefined} onChange={(v) => onUpdate(key, v)} />
        )}
      </div>
    )
  }

  /** 渲染一组嵌套对象的所有字段 */
  const renderSubSection = (
    section: string,
    obj: Record<string, unknown>,
    onUpdate: (key: string, val: unknown) => void,
    cols: number = 2,
  ) => {
    const keys = Object.keys(obj)
    return (
      <div className={`grid grid-cols-${cols} gap-2`}>
        {keys.map((k) => renderField(k, obj[k], `${section}.${k}`, onUpdate))}
      </div>
    )
  }

  // 基础参数: value 中排除 name 和特性模块 key 后的所有扁平字段
  const basicKeys = Object.keys(value).filter(k => !SKIP_BASIC_KEYS.has(k))

  const updateBasicField = (key: string, val: unknown) => {
    onChange({ ...value, [key]: val })
  }

  const updateSubField = (section: 'MoE' | 'MLA' | 'DSA' | 'NSA' | 'RoPE') =>
    (key: string, val: unknown) => {
      const sub = value[section]
      if (!sub && section !== 'RoPE') return
      onChange({ ...value, [section]: { ...(sub || {}), [key]: val } })
    }

  return (
    <div>
      {/* 预设选择 */}
      <div className="mb-3">
        <div className="mb-1 flex justify-between items-center">
          <span className="text-gray-500 text-xs">模型预设</span>
          <Button variant="link" size="sm" className="p-0 h-auto text-xs" onClick={() => {
            const allOpen = Object.values(sections).every(Boolean)
            const next: Record<string, boolean> = {}
            for (const k of Object.keys(sections)) next[k] = !allOpen
            setSections(next)
          }}>
            {Object.values(sections).every(Boolean)
              ? <><ChevronUp className="h-3 w-3 mr-1" />全部折叠</>
              : <><ChevronDown className="h-3 w-3 mr-1" />全部展开</>}
          </Button>
        </div>
        <Select value={selectedPreset} onValueChange={handlePresetChange}>
          <SelectTrigger className="w-full h-7">
            <SelectValue placeholder="选择模型预设..." />
          </SelectTrigger>
          <SelectContent>
            {presetList.map((p) => (
              <SelectItem key={p.name} value={p.name}>{p.name}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2 mb-3">
        {/* 基础参数 */}
        <BaseCard title="基础参数" collapsible expanded={sections.basic} onExpandChange={() => toggleSection('basic')} contentClassName="p-2" gradient>
          <div className="grid grid-cols-2 gap-2">
            {basicKeys.map((k) => renderField(
              k,
              (value as unknown as Record<string, unknown>)[k],
              k,
              updateBasicField,
            ))}
          </div>
          {/* 特性模块开关 */}
          <div className="mt-3 pt-2 border-t border-dashed border-gray-200">
            <div className="mb-1.5 flex items-center gap-1.5">
              <span className="text-xs text-gray-500">特性模块</span>
              {(['MoE', 'MLA', 'DSA', 'NSA'] as const).some(f => isFeatureModified(f)) && (
                <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
              )}
            </div>
            <div className="flex gap-2">
              {(['MoE', 'MLA', 'DSA', 'NSA'] as const).map((f) => {
                const active = !!value[f]
                const modified = isFeatureModified(f)
                return (
                  <Button
                    key={f}
                    variant="outline"
                    size="sm"
                    className={`h-7 px-3 text-xs font-medium transition-shadow ${
                      active
                        ? 'bg-blue-50 text-blue-700 border-blue-300 hover:bg-blue-100'
                        : 'bg-gray-50 text-gray-400 border-gray-200 hover:bg-gray-100 hover:text-gray-500'
                    } ${modified ? 'ring-1 ring-blue-400 ring-offset-1 shadow-[0_0_8px_rgba(59,130,246,0.5)]' : ''}`}
                    onClick={() => toggleFeature(f)}
                  >
                    {FEATURE_LABELS[f]}
                  </Button>
                )
              })}
            </div>
          </div>
        </BaseCard>

        {/* MoE 参数 */}
        {value.MoE && (
          <BaseCard title="MoE 参数" collapsible expanded={sections.MoE} onExpandChange={() => toggleSection('MoE')} contentClassName="p-2" gradient>
            {renderSubSection('MoE', value.MoE as unknown as Record<string, unknown>, updateSubField('MoE'))}
          </BaseCard>
        )}

        {/* MLA 参数 */}
        {value.MLA && (
          <BaseCard title="MLA 参数" collapsible expanded={sections.MLA} onExpandChange={() => toggleSection('MLA')} contentClassName="p-2" gradient>
            {renderSubSection('MLA', value.MLA as unknown as Record<string, unknown>, updateSubField('MLA'))}
          </BaseCard>
        )}

        {/* DSA 参数 */}
        {value.DSA && (
          <BaseCard title="DSA 参数" collapsible expanded={sections.DSA} onExpandChange={() => toggleSection('DSA')} contentClassName="p-2" gradient>
            {renderSubSection('DSA', value.DSA as unknown as Record<string, unknown>, updateSubField('DSA'), 3)}
          </BaseCard>
        )}

        {/* NSA 参数 */}
        {value.NSA && (
          <BaseCard title="NSA 参数" collapsible expanded={sections.NSA} onExpandChange={() => toggleSection('NSA')} contentClassName="p-2" gradient>
            {renderSubSection('NSA', value.NSA as unknown as Record<string, unknown>, updateSubField('NSA'), 3)}
          </BaseCard>
        )}

        {/* RoPE 参数 */}
        {value.RoPE && (
          <BaseCard title="RoPE 参数" collapsible expanded={sections.RoPE} onExpandChange={() => toggleSection('RoPE')} contentClassName="p-2" gradient>
            {renderSubSection('RoPE', value.RoPE as unknown as Record<string, unknown>, updateSubField('RoPE'))}
          </BaseCard>
        )}
      </div>

      {/* 操作按钮 */}
      <div className="flex gap-2">
        <Button variant="outline" size="sm" onClick={handleSave} disabled={!selectedPreset}>
          <Save className="h-3.5 w-3.5 mr-1" />保存
        </Button>
        <Button variant="outline" size="sm" onClick={() => { setSaveAsName(value.name); setSaveAsOpen(true) }}>
          <Copy className="h-3.5 w-3.5 mr-1" />另存为
        </Button>
        <Button variant="outline" size="sm" onClick={handleReload} disabled={!originalRef.current}>
          <RefreshCw className="h-3.5 w-3.5 mr-1" />重新加载
        </Button>
      </div>

      {/* 另存为弹窗 */}
      <Dialog open={saveAsOpen} onOpenChange={setSaveAsOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>另存为新预设</DialogTitle>
          </DialogHeader>
          <div className="py-4">
            <label className="text-sm font-medium mb-2 block">预设名称</label>
            <Input
              value={saveAsName}
              onChange={(e) => setSaveAsName(e.target.value)}
              placeholder="请输入预设名称"
              onKeyDown={(e) => { if (e.key === 'Enter' && saveAsName.trim()) handleSaveAs() }}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => { setSaveAsOpen(false); setSaveAsName('') }}>取消</Button>
            <Button onClick={handleSaveAs} disabled={!saveAsName.trim()}>保存</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
