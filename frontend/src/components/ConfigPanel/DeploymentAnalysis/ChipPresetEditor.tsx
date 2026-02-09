/**
 * ChipPresetEditor - 芯片预设编辑器
 *
 * 动态渲染配置中实际存在的所有字段 (递归支持深层嵌套)，
 * 支持加载/编辑/保存/另存为芯片预设配置，
 * 带修改追踪和蓝色高亮显示。
 */

import React, { useState, useEffect, useCallback, useRef } from 'react'
import { Save, Copy, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { NumberInput } from '@/components/ui/number-input'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { BaseCard } from '@/components/common/BaseCard'
import type { ChipPreset } from '@/types/tier6'
import { getChipPresets, getChipPreset, saveChipPreset, updateChipPreset } from '@/api/tier6'

// ==================== Props ====================

interface ChipPresetEditorProps {
  value: ChipPreset
  onChange: (config: ChipPreset) => void
}

// ==================== Helpers ====================

function deepClone<T>(obj: T): T { return structuredClone(obj) }

function setNested(obj: Record<string, unknown>, path: string, val: unknown): void {
  const keys = path.split('.')
  let cur: Record<string, unknown> = obj
  for (let i = 0; i < keys.length - 1; i++) {
    if (cur[keys[i]] == null || typeof cur[keys[i]] !== 'object') cur[keys[i]] = {}
    cur = cur[keys[i]] as Record<string, unknown>
  }
  cur[keys[keys.length - 1]] = val
}

function getNested(obj: unknown, path: string): unknown {
  let cur: unknown = obj
  for (const k of path.split('.')) {
    if (cur == null || typeof cur !== 'object') return undefined
    cur = (cur as Record<string, unknown>)[k]
  }
  return cur
}

function errMsg(err: unknown): string { return err instanceof Error ? err.message : String(err) }

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return v != null && typeof v === 'object' && !Array.isArray(v)
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

// ==================== Component ====================

export const ChipPresetEditor: React.FC<ChipPresetEditorProps> = ({ value, onChange }) => {
  const [presetList, setPresetList] = useState<Array<{ name: string; config: ChipPreset }>>([])
  const [selectedPreset, setSelectedPreset] = useState<string>('')
  const originalRef = useRef<ChipPreset | null>(null)
  const [sections, setSections] = useState<Record<string, boolean>>({ basic: false })
  const [saveAsOpen, setSaveAsOpen] = useState(false)
  const [saveAsName, setSaveAsName] = useState('')

  // 加载预设列表，自动选择上次使用的或第一个
  useEffect(() => {
    getChipPresets().then(({ presets }) => {
      setPresetList(presets)
      if (presets.length === 0) return
      const lastUsed = localStorage.getItem('tier6_last_chip_preset')
      const target = presets.find(p => p.name === lastUsed)
        || presets.find(p => p.name === value.name)
        || presets[0]
      setSelectedPreset(target.name)
      originalRef.current = deepClone(target.config)
      onChange(target.config)
    }).catch((err) => toast.error(`加载芯片预设失败: ${errMsg(err)}`))
  }, [])

  // 通用字段更新 (支持嵌套路径)
  const updateField = useCallback((path: string, val: unknown) => {
    const u = deepClone(value) as unknown as Record<string, unknown>
    setNested(u, path, val)
    onChange(u as unknown as ChipPreset)
  }, [value, onChange])

  // 修改检测
  const isFieldModified = useCallback((path: string): boolean => {
    if (!originalRef.current) return false
    const a = getNested(value, path)
    const b = getNested(originalRef.current, path)
    return JSON.stringify(a) !== JSON.stringify(b)
  }, [value])

  // 切换预设
  const handlePresetChange = async (name: string) => {
    try {
      const config = await getChipPreset(name)
      setSelectedPreset(name)
      localStorage.setItem('tier6_last_chip_preset', name)
      originalRef.current = deepClone(config); onChange(config)
    } catch (err) { toast.error(`加载预设失败: ${errMsg(err)}`) }
  }

  // 保存
  const handleSave = async () => {
    if (!selectedPreset) { toast.error('未选择预设，请使用另存为'); return }
    try {
      await updateChipPreset(selectedPreset, value)
      originalRef.current = deepClone(value)
      const { presets } = await getChipPresets(); setPresetList(presets)
      toast.success(`已保存: ${selectedPreset}`)
    } catch (err) { toast.error(`保存失败: ${errMsg(err)}`) }
  }

  // 另存为
  const handleSaveAs = async () => {
    const name = saveAsName.trim()
    if (!name) { toast.error('请输入预设名称'); return }
    try {
      const cfg = deepClone(value); cfg.name = name
      await saveChipPreset(cfg)
      setSelectedPreset(name); originalRef.current = deepClone(cfg); onChange(cfg)
      const { presets } = await getChipPresets(); setPresetList(presets)
      setSaveAsOpen(false); setSaveAsName('')
      toast.success(`已另存为: ${name}`)
    } catch (err) { toast.error(`另存为失败: ${errMsg(err)}`) }
  }

  // 重新加载
  const handleReload = async () => {
    if (!selectedPreset) return
    try {
      const config = await getChipPreset(selectedPreset)
      originalRef.current = deepClone(config); onChange(config)
      toast.success('已重新加载')
    } catch (err) { toast.error(`重新加载失败: ${errMsg(err)}`) }
  }

  // ==================== 动态渲染 ====================

  const modBadge = (path: string) =>
    isFieldModified(path)
      ? <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
      : null

  /** 渲染单个叶子字段 */
  const renderField = (key: string, val: unknown, path: string) => {
    const modified = isFieldModified(path)
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
          <NumberInput value={val as number | undefined} onChange={(v) => updateField(path, v)} />
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
          <div className="grid grid-cols-3 gap-2">
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
  const asRecord = value as unknown as Record<string, unknown>
  const topLeafKeys = Object.keys(asRecord).filter(k => k !== 'name' && !isPlainObject(asRecord[k]))
  const topObjKeys = Object.keys(asRecord).filter(k => k !== 'name' && isPlainObject(asRecord[k]))

  // 确保 sections 包含所有顶层对象 key (预设切换后可能变化)
  const ensuredSections = { ...sections }
  for (const k of topObjKeys) {
    if (!(k in ensuredSections)) ensuredSections[k] = false
  }

  const toggleSection = (key: string) => setSections(prev => ({ ...prev, [key]: !prev[key] }))

  const allSectionKeys = ['basic', ...topObjKeys]
  const allOpen = allSectionKeys.every(k => ensuredSections[k])

  return (
    <div className="space-y-2">
      {/* 预设选择器 */}
      <div className="mb-2">
        <div className="mb-1 flex justify-between items-center">
          <span className="text-xs text-gray-500">芯片预设</span>
          <Button variant="link" size="sm" className="p-0 h-auto text-xs" onClick={() => {
            const next: Record<string, boolean> = {}
            for (const k of allSectionKeys) next[k] = !allOpen
            setSections(next)
          }}>
            {allOpen
              ? <><ChevronUp className="h-3 w-3 mr-1" />全部折叠</>
              : <><ChevronDown className="h-3 w-3 mr-1" />全部展开</>}
          </Button>
        </div>
        <Select value={selectedPreset} onValueChange={handlePresetChange}>
          <SelectTrigger className="w-full h-7"><SelectValue placeholder="选择芯片预设" /></SelectTrigger>
          <SelectContent>
            {presetList.map((p) => <SelectItem key={p.name} value={p.name}>{p.name}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>

      {/* 基础参数 (顶层叶子字段) */}
      <BaseCard title="基础参数" collapsible expanded={ensuredSections.basic} onExpandChange={() => toggleSection('basic')} contentClassName="p-2" gradient>
        <div className="grid grid-cols-3 gap-2">
          {topLeafKeys.map(k => renderField(k, asRecord[k], k))}
        </div>
      </BaseCard>

      {/* 各顶层对象节点各自一个 card */}
      {topObjKeys.map(sectionKey => (
        <BaseCard
          key={sectionKey}
          title={SECTION_LABELS[sectionKey] || sectionKey}
          collapsible
          expanded={ensuredSections[sectionKey] ?? false}
          onExpandChange={() => toggleSection(sectionKey)}
          contentClassName="p-2"
          gradient
        >
          {renderNode(asRecord[sectionKey] as Record<string, unknown>, sectionKey)}
        </BaseCard>
      ))}

      {/* 操作按钮 */}
      <div className="flex gap-2 pt-1">
        <Button variant="outline" size="sm" onClick={handleSave}><Save className="h-3.5 w-3.5 mr-1" />保存</Button>
        <Button variant="outline" size="sm" onClick={() => { setSaveAsName(value.name); setSaveAsOpen(true) }}><Copy className="h-3.5 w-3.5 mr-1" />另存为</Button>
        <Button variant="outline" size="sm" onClick={handleReload}><RefreshCw className="h-3.5 w-3.5 mr-1" />重新加载</Button>
      </div>

      {/* 另存为弹窗 */}
      <Dialog open={saveAsOpen} onOpenChange={setSaveAsOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader><DialogTitle>另存为新预设</DialogTitle></DialogHeader>
          <div className="py-4">
            <label className="text-sm font-medium mb-2 block">预设名称</label>
            <Input
              value={saveAsName} onChange={(e) => setSaveAsName(e.target.value)}
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

export default ChipPresetEditor
