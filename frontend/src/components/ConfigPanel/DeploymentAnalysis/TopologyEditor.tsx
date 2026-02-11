/**
 * TopologyEditor - 拓扑配置编辑器
 *
 * grouped_pods 格式: pods[].racks[].boards[].chips[]
 * 层级配置只读展示，互联参数可编辑，支持预设管理(保存/另存为/重载)
 */
import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { Save, Copy, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { NumberInput } from '@/components/ui/number-input'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { HelpTooltip } from '@/components/ui/info-tooltip'
import { BaseCard } from '@/components/common/BaseCard'
import type { TopologyConfig, TopologyListItem } from '@/types/math_model'
import { getTopologies, getTopology, createTopology, updateTopology } from '@/api/math_model'
import { countChips, countPods, countRacks, countBoards } from '@/utils/llmDeployment/topologyFormat'
import { deepClone, errMsg } from '@/utils/nestedObjectEditor'

interface TopologyEditorProps {
  value: TopologyConfig
  onChange: (config: TopologyConfig) => void
  onParamsModified?: (modified: boolean) => void
}

const IC_LABELS: Record<string, { label: string; tip: string }> = {
  c2c: { label: 'C2C (Chip-to-Chip)', tip: 'Die 间直连，同封装内芯片互联' },
  b2b: { label: 'B2B (Board-to-Board)', tip: '板间互联，同机架内跨板通信' },
  r2r: { label: 'R2R (Rack-to-Rack)', tip: '机架间互联，同 Pod 内跨机架通信' },
  p2p: { label: 'P2P (Pod-to-Pod)', tip: '跨 Pod 互联，集群间通信' },
}

const COMM_FIELDS: Array<{ key: string; label: string; tip: string; step?: number }> = [
  { key: 'bandwidth_utilization', label: 'Bandwidth Utilization', tip: 'Actual BW / Peak BW (0~1)', step: 0.01 },
  { key: 'sync_latency_us', label: 'Sync Latency (us)', tip: 'Synchronization barrier latency' },
  { key: 'switch_latency_us', label: 'Switch Latency (us)', tip: 'Switch forwarding latency' },
  { key: 'cable_latency_us', label: 'Cable Latency (us)', tip: 'Cable transmission latency' },
  { key: 'memory_read_latency_us', label: 'DDR Read Latency (us)', tip: 'HBM/GDDR read latency' },
  { key: 'memory_write_latency_us', label: 'DDR Write Latency (us)', tip: 'HBM/GDDR write latency' },
  { key: 'noc_latency_us', label: 'NoC Latency (us)', tip: 'Network-on-Chip latency' },
  { key: 'die_to_die_latency_us', label: 'Die-to-Die Latency (us)', tip: 'Die-to-Die interconnect latency' },
]

function getICValue(cfg: TopologyConfig, lvl: string, f: 'bandwidth_gbps' | 'latency_us'): number | undefined {
  const links = cfg.interconnect?.links
  if (!links) return undefined
  const e = (links as Record<string, { bandwidth_gbps: number; latency_us: number } | undefined>)[lvl]
  return e?.[f]
}

/** 统计每种芯片的总数 */
function getChipBreakdown(value: TopologyConfig): Array<{ name: string; count: number }> {
  const chipMap = new Map<string, number>()
  for (const podGroup of value.pods || []) {
    const pc = podGroup.count ?? 1
    for (const rackGroup of podGroup.racks ?? []) {
      const rc = rackGroup.count ?? 1
      for (const board of rackGroup.boards ?? []) {
        const bc = board.count ?? 1
        for (const chip of board.chips ?? []) {
          chipMap.set(chip.name, (chipMap.get(chip.name) || 0) + pc * rc * bc * (chip.count ?? 1))
        }
      }
    }
  }
  return Array.from(chipMap.entries()).map(([name, count]) => ({ name, count }))
}

/** 计算各层级的 link 连接数 (全互联假设) */
function countLevelLinks(value: TopologyConfig): { p2p: number; r2r: number; b2b: number; c2c: number } {
  const pods = value.pods || []
  let c2c = 0, b2b = 0, r2r = 0
  const mesh = (n: number) => n * (n - 1) / 2

  // p2p: 所有 pod 实例间的连接
  let totalPodInstances = 0
  for (const pg of pods) totalPodInstances += pg.count ?? 1
  const p2p = mesh(totalPodInstances)

  for (const pg of pods) {
    const pc = pg.count ?? 1
    // r2r: 同一 pod 内 rack 实例间的连接
    let racksInPod = 0
    for (const rg of pg.racks ?? []) racksInPod += rg.count ?? 1
    r2r += pc * mesh(racksInPod)

    for (const rg of pg.racks ?? []) {
      const rc = rg.count ?? 1
      // b2b: 同一 rack 内 board 实例间的连接
      let boardsInRack = 0
      for (const bd of rg.boards ?? []) boardsInRack += bd.count ?? 1
      b2b += pc * rc * mesh(boardsInRack)

      for (const bd of rg.boards ?? []) {
        const bc = bd.count ?? 1
        // c2c: 同一 board 内 chip 实例间的连接
        let chipsInBoard = 0
        for (const ch of bd.chips ?? []) chipsInBoard += ch.count ?? 1
        c2c += pc * rc * bc * mesh(chipsInBoard)
      }
    }
  }
  return { p2p, r2r, b2b, c2c }
}


export const TopologyEditor: React.FC<TopologyEditorProps> = ({ value, onChange, onParamsModified }) => {
  const [presets, setPresets] = useState<TopologyListItem[]>([])
  const [selectedPreset, setSelectedPreset] = useState<string>('')
  const [snapshot, setSnapshot] = useState<TopologyConfig | null>(null)
  const [saveAsOpen, setSaveAsOpen] = useState(false)
  const [saveAsName, setSaveAsName] = useState('')
  const [loading, setLoading] = useState(false)
  const [sections, setSections] = useState({ hierarchy: false, interconnect: false })
  const initRef = useRef(false)

  const [refreshing, setRefreshing] = useState(false)

  // 加载预设列表，自动选择上次使用的或第一个
  useEffect(() => {
    getTopologies()
      .then(async (res) => {
        setPresets(res.topologies)
        if (initRef.current || res.topologies.length === 0) return
        initRef.current = true
        const lastUsed = localStorage.getItem('tier6_last_topology_preset')
        // 优先使用父组件传入的 value.name（来自 Benchmark 联动），其次 localStorage
        const target = (value.name ? res.topologies.find((t) => t.name === value.name) : null)
          || res.topologies.find((t) => t.name === lastUsed)
          || res.topologies[0]
        setSelectedPreset(target.name)
        try {
          const cfg = await getTopology(target.name)
          setSnapshot(deepClone(cfg)); onChange(cfg)
        } catch (e) { toast.error(`加载拓扑失败: ${errMsg(e)}`) }
      })
      .catch((e) => toast.error(`加载拓扑列表失败: ${errMsg(e)}`))
  }, [])

  // 刷新预设列表（手动触发）
  const handleRefreshList = useCallback(async () => {
    setRefreshing(true)
    try {
      const res = await getTopologies()
      setPresets(res.topologies)
      toast.success(`已刷新拓扑预设列表 (${res.topologies.length} 个)`)
    } catch (e) {
      toast.error(`刷新拓扑预设列表失败: ${errMsg(e)}`)
    } finally {
      setRefreshing(false)
    }
  }, [])

  // 检测外部推送的预设变更（如 Benchmark 加载联动），同步内部状态
  useEffect(() => {
    if (!value.name || !initRef.current) return
    if (value.name === selectedPreset) return
    // 外部改变了 name，从后端加载该预设的干净配置，重置 snapshot 和 value
    setSelectedPreset(value.name)
    getTopology(value.name)
      .then(cfg => {
        setSnapshot(deepClone(cfg))
        // 用预设列表的干净配置替换外部推送的对象，确保 value 和 snapshot 一致
        onChange(cfg)
      })
      .catch(err => {
        console.error('加载拓扑失败:', err)
        // 如果加载失败，使用外部传入的 value 作为 snapshot
        setSnapshot(deepClone(value))
      })
  }, [value.name, onChange])

  const handlePresetChange = useCallback(async (name: string) => {
    setSelectedPreset(name)
    localStorage.setItem('tier6_last_topology_preset', name)
    setLoading(true)
    try {
      const cfg = await getTopology(name)
      setSnapshot(deepClone(cfg)); onChange(cfg)
    } catch (e) { toast.error(`加载拓扑失败: ${errMsg(e)}`) }
    finally { setLoading(false) }
  }, [onChange])

  const handleSave = useCallback(async () => {
    if (!selectedPreset) { toast.error('请先选择拓扑预设'); return }
    try {
      await updateTopology(selectedPreset, value)
      setSnapshot(deepClone(value)); toast.success(`已保存: ${selectedPreset}`)
    } catch (e) { toast.error(`保存失败: ${errMsg(e)}`) }
  }, [selectedPreset, value])

  const handleSaveAs = useCallback(async () => {
    const name = saveAsName.trim()
    if (!name) { toast.error('请输入拓扑名称'); return }
    try {
      const cfg = { ...value, name }
      const result = await createTopology(cfg)
      setPresets((p) => [...p, { name: result.name }])
      setSelectedPreset(result.name); setSnapshot(deepClone(cfg)); onChange(cfg)
      toast.success(`已另存为: ${result.name}`)
      setSaveAsOpen(false); setSaveAsName('')
    } catch (e) { toast.error(`另存为失败: ${errMsg(e)}`) }
  }, [saveAsName, value, onChange])

  const handleReload = useCallback(async () => {
    if (!selectedPreset) { toast.error('请先选择拓扑预设'); return }
    setLoading(true)
    try {
      const cfg = await getTopology(selectedPreset)
      setSnapshot(deepClone(cfg)); onChange(cfg); toast.success('已重新加载')
    } catch (e) { toast.error(`重新加载失败: ${errMsg(e)}`) }
    finally { setLoading(false) }
  }, [selectedPreset, onChange])

  const isModified = useCallback(
    (getter: (c: TopologyConfig) => unknown): boolean => {
      if (!snapshot) return false
      return JSON.stringify(getter(value)) !== JSON.stringify(getter(snapshot))
    }, [value, snapshot]
  )

  const isAnyParamModified = useMemo(() => {
    if (!snapshot) return false
    const check = (getter: (c: TopologyConfig) => unknown) =>
      JSON.stringify(getter(value)) !== JSON.stringify(getter(snapshot))
    return check(c => c.pods)
      || check(c => c.interconnect?.links)
      || check(c => c.interconnect?.comm_params)
  }, [value, snapshot])

  useEffect(() => {
    onParamsModified?.(isAnyParamModified)
  }, [isAnyParamModified, onParamsModified])

  const getCommValue = useCallback((key: string): number | undefined => {
    const cp = value.interconnect?.comm_params as Record<string, unknown> | undefined
    return cp ? cp[key] as number | undefined : undefined
  }, [value])

  const updateComm = useCallback((key: string, val: number | undefined) => {
    if (val === undefined) return
    const cp = (value.interconnect?.comm_params || {}) as Record<string, unknown>
    const ic = value.interconnect || {}
    onChange({ ...value, interconnect: { ...ic, comm_params: { ...cp, [key]: val } } })
  }, [value, onChange])

  const updateIC = useCallback(
    (lvl: string, field: 'bandwidth_gbps' | 'latency_us', val: number | undefined) => {
      if (val === undefined) return
      const links = value.interconnect?.links ?? {}
      const entry = (links as Record<string, Record<string, number>>)[lvl] ?? {}
      const ic = value.interconnect || {}
      onChange({
        ...value,
        interconnect: { ...ic, links: { ...links, [lvl]: { ...entry, [field]: val } } },
      })
    }, [value, onChange]
  )

  const mc = (mod: boolean) => mod ? 'bg-blue-50/50 rounded px-1 -mx-1' : ''
  const modBadge = (mod: boolean) => mod
    ? <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
    : null

  const totalChips = useMemo(() => countChips(value), [value.pods])
  const chipBreakdown = useMemo(() => getChipBreakdown(value), [value.pods])
  const levelLinks = useMemo(() => countLevelLinks(value), [value.pods])

  return (
    <div className="space-y-3">
      {/* 预设选择 */}
      <div>
        <div className="mb-1 flex justify-between items-center">
          <span className="text-gray-500 text-xs">拓扑预设</span>
          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-[11px]">{totalChips} chips</span>
            <Button variant="link" size="sm" className="p-0 h-auto text-xs" onClick={() => {
              const allOpen = Object.values(sections).every(Boolean)
              setSections({ hierarchy: !allOpen, interconnect: !allOpen })
            }}>
              {Object.values(sections).every(Boolean)
                ? <><ChevronUp className="h-3 w-3 mr-1" />全部折叠</>
                : <><ChevronDown className="h-3 w-3 mr-1" />全部展开</>}
            </Button>
          </div>
        </div>
        <div className="flex gap-1.5">
          <Select value={selectedPreset} onValueChange={handlePresetChange}>
            <SelectTrigger className="flex-1 h-7"><SelectValue placeholder="选择拓扑预设..." /></SelectTrigger>
            <SelectContent>
              {presets.map((p) => (
                <SelectItem key={p.name} value={p.name}>{p.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="ghost" size="sm" className="h-7 w-7 p-0 shrink-0" onClick={handleRefreshList} disabled={refreshing || loading} title="刷新预设列表">
            <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </div>

      {/* 层级配置 (只读) */}
      <BaseCard title="层级配置" collapsible expanded={sections.hierarchy} onExpandChange={() => setSections(s => ({ ...s, hierarchy: !s.hierarchy }))} contentClassName="p-2" gradient>
        <div className="space-y-1.5 text-xs text-gray-600">
          {/* 统计概览 */}
          <div className="grid grid-cols-4 gap-2">
            <div className="bg-gray-50 rounded px-2 py-1 text-center">
              <span className="text-gray-400 block text-[10px]">Pod</span>
              <span className="font-medium">{countPods(value)}</span>
            </div>
            <div className="bg-gray-50 rounded px-2 py-1 text-center">
              <span className="text-gray-400 block text-[10px]">Rack</span>
              <span className="font-medium">{countRacks(value)}</span>
            </div>
            <div className="bg-gray-50 rounded px-2 py-1 text-center">
              <span className="text-gray-400 block text-[10px]">Board</span>
              <span className="font-medium">{countBoards(value)}</span>
            </div>
            <div className="bg-gray-50 rounded px-2 py-1 text-center relative group cursor-default">
              <span className="text-gray-400 block text-[10px]">Chip</span>
              <span className="font-medium">{totalChips}</span>
              {chipBreakdown.length > 0 && (
                <div className="absolute left-1/2 -translate-x-1/2 top-full mt-1 z-10 hidden group-hover:block bg-gray-800 text-white text-[11px] rounded px-2 py-1 whitespace-nowrap shadow-lg">
                  {chipBreakdown.map(({ name, count }) => (
                    <div key={name}>{name} x{count}</div>
                  ))}
                </div>
              )}
            </div>
          </div>
          {/* Link 连接数 */}
          <div className="grid grid-cols-4 gap-1.5">
            {([
              { key: 'p2p', label: 'P2P', count: levelLinks.p2p },
              { key: 'r2r', label: 'R2R', count: levelLinks.r2r },
              { key: 'b2b', label: 'B2B', count: levelLinks.b2b },
              { key: 'c2c', label: 'C2C', count: levelLinks.c2c },
            ] as const).map(({ key, label, count }) => (
              <div key={key} className="bg-gray-50 rounded px-1.5 py-0.5 text-center">
                <span className="text-gray-400 text-[10px]">{label} Links</span>
                <span className={`block font-mono text-[11px] ${count > 0 ? 'text-gray-700' : 'text-gray-300'}`}>{count}</span>
              </div>
            ))}
          </div>
        </div>
      </BaseCard>

      {/* 互联配置 */}
      <BaseCard title="互联配置" collapsible expanded={sections.interconnect} onExpandChange={() => setSections(s => ({ ...s, interconnect: !s.interconnect }))} contentClassName="p-2" gradient>
        <div className="space-y-3">
          {(['c2c', 'b2b', 'r2r', 'p2p'] as const).map((lvl) => {
            const m = IC_LABELS[lvl]
            const bwMod = isModified((c) => c.interconnect?.links?.[lvl]?.bandwidth_gbps)
            const latMod = isModified((c) => c.interconnect?.links?.[lvl]?.latency_us)
            return (
              <div key={lvl}>
                <HelpTooltip label={m.label} content={m.tip} labelClassName="text-gray-600 text-xs font-medium cursor-help" />
                <div className="grid grid-cols-2 gap-2 mt-1">
                  <div className={mc(bwMod)}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <span className="text-gray-400 text-[11px]">BW (GB/s)</span>
                      {modBadge(bwMod)}
                    </div>
                    <NumberInput min={0} step={1} value={getICValue(value, lvl, 'bandwidth_gbps')} onChange={(v) => updateIC(lvl, 'bandwidth_gbps', v)} className="h-7" />
                  </div>
                  <div className={mc(latMod)}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <span className="text-gray-400 text-[11px]">Latency (us)</span>
                      {modBadge(latMod)}
                    </div>
                    <NumberInput min={0} step={0.1} value={getICValue(value, lvl, 'latency_us')} onChange={(v) => updateIC(lvl, 'latency_us', v)} className="h-7" />
                  </div>
                </div>
              </div>
            )
          })}

          {/* 通信参数 */}
          <div className="border-t border-gray-200 pt-2 mt-2">
            <span className="text-gray-500 text-xs font-medium">通信参数</span>
            <div className="grid grid-cols-2 gap-2 mt-1.5">
              {COMM_FIELDS.map(({ key, label, tip, step }) => {
                const mod = isModified((c) => (c.interconnect?.comm_params as Record<string, unknown> | undefined)?.[key])
                return (
                  <div key={key} className={mc(mod)}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <HelpTooltip label={label} content={tip} labelClassName="text-gray-400 text-[11px] cursor-help" />
                      {modBadge(mod)}
                    </div>
                    <NumberInput min={0} step={step ?? 0.01} value={getCommValue(key)} onChange={(v) => updateComm(key, v)} className="h-7" />
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </BaseCard>

      {/* 操作按钮 */}
      <div className="flex gap-2">
        <Button variant="outline" size="sm" onClick={handleSave} disabled={loading || !selectedPreset}>
          <Save className="h-3.5 w-3.5 mr-1" />保存
        </Button>
        <Button variant="outline" size="sm" onClick={() => { setSaveAsName(value.name ?? ''); setSaveAsOpen(true) }} disabled={loading}>
          <Copy className="h-3.5 w-3.5 mr-1" />另存为
        </Button>
        <Button variant="outline" size="sm" onClick={handleReload} disabled={loading || !selectedPreset}>
          <RefreshCw className="h-3.5 w-3.5 mr-1" />重新加载
        </Button>
      </div>

      {/* 另存为弹窗 */}
      <Dialog open={saveAsOpen} onOpenChange={setSaveAsOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader><DialogTitle>另存为新拓扑</DialogTitle></DialogHeader>
          <div className="py-4">
            <label className="text-sm font-medium mb-2 block">拓扑名称</label>
            <Input value={saveAsName} onChange={(e) => setSaveAsName(e.target.value)} placeholder="如: P1-R1-B2-C16"
              onKeyDown={(e) => { if (e.key === 'Enter' && saveAsName.trim()) handleSaveAs() }} />
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

export default TopologyEditor
