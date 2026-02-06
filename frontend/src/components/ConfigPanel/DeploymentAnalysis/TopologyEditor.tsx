/**
 * TopologyEditor - 拓扑配置编辑器
 *
 * 支持从后端加载拓扑预设，编辑结构和互联参数，
 * 提供保存/另存为/重载功能，修改字段蓝色高亮。
 */
import React, { useState, useEffect, useCallback, useRef } from 'react'
import { Save, Copy, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { NumberInput } from '@/components/ui/number-input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { HelpTooltip } from '@/components/ui/info-tooltip'
import { BaseCard } from '@/components/common/BaseCard'
import type { Tier6TopologyConfig, TopologyListItem } from '@/types/tier6'
import { getTopologies, getTopology, createTopology, updateTopology } from '@/api/tier6'

interface TopologyEditorProps {
  value: Tier6TopologyConfig
  onChange: (config: Tier6TopologyConfig) => void
}

const IC_LABELS: Record<string, { label: string; tip: string }> = {
  c2c: { label: 'C2C (Chip-to-Chip)', tip: 'Die 间直连，同封装内芯片互联' },
  b2b: { label: 'B2B (Board-to-Board)', tip: '板间互联，同机架内跨板通信' },
  r2r: { label: 'R2R (Rack-to-Rack)', tip: '机架间互联，同 Pod 内跨机架通信' },
  p2p: { label: 'P2P (Pod-to-Pod)', tip: '跨 Pod 互联，集群间通信' },
}

function deepClone<T>(obj: T): T { return JSON.parse(JSON.stringify(obj)) }

function getICValue(cfg: Tier6TopologyConfig, lvl: string, f: 'bandwidth_gbps' | 'latency_us'): number | undefined {
  const ic = cfg.hardware_params?.interconnect
  if (!ic) return undefined
  const e = (ic as Record<string, { bandwidth_gbps: number; latency_us: number } | undefined>)[lvl]
  return e?.[f]
}

function parseBoardSummary(rack: Record<string, unknown> | undefined): string[] {
  if (!rack) return []
  const boards = rack.boards as Array<Record<string, unknown>> | undefined
  if (!Array.isArray(boards)) return []
  return boards.map((board, idx) => {
    const chips = board.chips as Array<Record<string, unknown>> | undefined
    const cnt = (board.count as number) ?? 1
    const parts: string[] = []
    if (Array.isArray(chips)) {
      chips.forEach((c) => {
        const n = c.name as string | undefined
        const cc = c.count as number | undefined
        if (n && cc) parts.push(`${n} x${cc}`)
      })
    }
    return `Board ${idx + 1}: ${parts.length > 0 ? parts.join(', ') : 'N/A'} (count: ${cnt})`
  })
}

function errMsg(err: unknown): string { return err instanceof Error ? err.message : String(err) }

export const TopologyEditor: React.FC<TopologyEditorProps> = ({ value, onChange }) => {
  const [presets, setPresets] = useState<TopologyListItem[]>([])
  const [selectedPreset, setSelectedPreset] = useState<string>('')
  const [snapshot, setSnapshot] = useState<Tier6TopologyConfig | null>(null)
  const [saveAsOpen, setSaveAsOpen] = useState(false)
  const [saveAsName, setSaveAsName] = useState('')
  const [loading, setLoading] = useState(false)
  const [sections, setSections] = useState({ structure: true, board: false, interconnect: true })
  const initRef = useRef(false)

  // 加载预设列表，自动选择上次使用的或第一个
  useEffect(() => {
    getTopologies()
      .then(async (res) => {
        setPresets(res.topologies)
        if (initRef.current || res.topologies.length === 0) return
        initRef.current = true
        const lastUsed = localStorage.getItem('tier6_last_topology_preset')
        const target = res.topologies.find((t) => t.name === lastUsed)
          || (value.name ? res.topologies.find((t) => t.name === value.name) : null)
          || res.topologies[0]
        setSelectedPreset(target.name)
        try {
          const cfg = await getTopology(target.name)
          setSnapshot(deepClone(cfg)); onChange(cfg)
        } catch (e) { toast.error(`加载拓扑失败: ${errMsg(e)}`) }
      })
      .catch((e) => toast.error(`加载拓扑列表失败: ${errMsg(e)}`))
  }, [])

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
    (getter: (c: Tier6TopologyConfig) => unknown): boolean => {
      if (!snapshot) return false
      return JSON.stringify(getter(value)) !== JSON.stringify(getter(snapshot))
    }, [value, snapshot]
  )

  const updateIC = useCallback(
    (lvl: string, field: 'bandwidth_gbps' | 'latency_us', val: number | undefined) => {
      if (val === undefined) return
      const ic = value.hardware_params?.interconnect ?? {}
      const entry = (ic as Record<string, Record<string, number>>)[lvl] ?? {}
      onChange({
        ...value,
        hardware_params: { ...value.hardware_params, interconnect: { ...ic, [lvl]: { ...entry, [field]: val } } },
      })
    }, [value, onChange]
  )

  const mc = (mod: boolean) => mod ? 'bg-blue-50/60 rounded px-1 -mx-1' : ''
  const boardLines = parseBoardSummary(value.rack_config)

  return (
    <div className="space-y-3">
      {/* 预设选择 */}
      <div>
        <div className="mb-1 flex justify-between items-center">
          <span className="text-gray-500 text-xs">拓扑预设</span>
          <Button variant="link" size="sm" className="p-0 h-auto text-xs" onClick={() => {
            const allOpen = Object.values(sections).every(Boolean)
            setSections({ structure: !allOpen, board: !allOpen, interconnect: !allOpen })
          }}>
            {Object.values(sections).every(Boolean)
              ? <><ChevronUp className="h-3 w-3 mr-1" />全部折叠</>
              : <><ChevronDown className="h-3 w-3 mr-1" />全部展开</>}
          </Button>
        </div>
        <Select value={selectedPreset} onValueChange={handlePresetChange}>
          <SelectTrigger className="w-full h-7"><SelectValue placeholder="选择拓扑预设..." /></SelectTrigger>
          <SelectContent>
            {presets.map((p) => (
              <SelectItem key={p.name} value={p.name}>
                {p.name}{p.chip_count ? ` (${p.chip_count} chips)` : ''}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* 结构参数 */}
      <BaseCard title="结构参数" collapsible expanded={sections.structure} onExpandChange={() => setSections(s => ({ ...s, structure: !s.structure }))} contentClassName="p-2" gradient>
        <div className="grid grid-cols-2 gap-2">
          <div className={mc(isModified((c) => c.pod_count))}>
            <HelpTooltip label="Pod 数量" content="集群中 Pod 的数量" />
            <NumberInput min={1} value={value.pod_count} onChange={(v) => onChange({ ...value, pod_count: v })} className="h-7 mt-1" />
          </div>
          <div className={mc(isModified((c) => c.racks_per_pod))}>
            <HelpTooltip label="Rack/Pod" content="每个 Pod 中机架数量" />
            <NumberInput min={1} value={value.racks_per_pod} onChange={(v) => onChange({ ...value, racks_per_pod: v })} className="h-7 mt-1" />
          </div>
        </div>
      </BaseCard>

      {/* Board 配置 (只读) */}
      <BaseCard title="Board 配置" collapsible expanded={sections.board} onExpandChange={() => setSections(s => ({ ...s, board: !s.board }))} contentClassName="p-2" gradient>
        {boardLines.length > 0 ? (
          <div className="space-y-1">
            {boardLines.map((line, i) => <div key={i} className="text-xs text-gray-600 font-mono">{line}</div>)}
          </div>
        ) : (
          <div className="text-xs text-gray-400">暂无 Board 配置</div>
        )}
      </BaseCard>

      {/* 互联配置 */}
      <BaseCard title="互联配置" collapsible expanded={sections.interconnect} onExpandChange={() => setSections(s => ({ ...s, interconnect: !s.interconnect }))} contentClassName="p-2" gradient>
        <div className="space-y-3">
          {(['c2c', 'b2b', 'r2r', 'p2p'] as const).map((lvl) => {
            const m = IC_LABELS[lvl]
            const bwMod = isModified((c) => c.hardware_params?.interconnect?.[lvl]?.bandwidth_gbps)
            const latMod = isModified((c) => c.hardware_params?.interconnect?.[lvl]?.latency_us)
            return (
              <div key={lvl}>
                <HelpTooltip label={m.label} content={m.tip} labelClassName="text-gray-600 text-xs font-medium cursor-help" />
                <div className="grid grid-cols-2 gap-2 mt-1">
                  <div className={mc(bwMod)}>
                    <span className="text-gray-400 text-[11px]">BW (GB/s)</span>
                    <NumberInput min={0} step={1} value={getICValue(value, lvl, 'bandwidth_gbps')} onChange={(v) => updateIC(lvl, 'bandwidth_gbps', v)} className="h-7" />
                  </div>
                  <div className={mc(latMod)}>
                    <span className="text-gray-400 text-[11px]">Latency (us)</span>
                    <NumberInput min={0} step={0.1} value={getICValue(value, lvl, 'latency_us')} onChange={(v) => updateIC(lvl, 'latency_us', v)} className="h-7" />
                  </div>
                </div>
              </div>
            )
          })}
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
