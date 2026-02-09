/**
 * 拓扑状态管理 Context
 * 负责管理拓扑数据的加载、生成和配置
 */
import React, { createContext, useContext, useState, useCallback, useEffect, ReactNode } from 'react'
import { toast } from 'sonner'
import { HierarchicalTopology, ManualConnectionConfig } from '../types'
import { getTopology, generateTopology } from '../api/topology'

// ============================================
// 常量
// ============================================
const CONFIG_CACHE_KEY = 'tier6_topology_config_cache'

// ============================================
// 类型定义
// ============================================

// Rack 配置类型（用于部署分析）
// 芯片详细配置现在存储在顶层 chips 字典中（使用 Tier6 ChipPreset 格式）
export interface RackConfigForAnalysis {
  total_u: number
  boards: Array<{
    id: string
    name: string
    u_height: number
    count: number
    chips: Array<{
      name: string
      count: number
      preset_id?: string
    }>
  }>
}

// 互联参数配置
interface InterconnectParams {
  bandwidth_gbps: number
  latency_us: number
}

// 生成配置类型
export interface GenerateConfig {
  pod_count: number
  racks_per_pod: number
  rack_config?: {
    total_u: number
    boards: Array<{
      id: string
      name: string
      u_height: number
      count: number
      chips: Array<{ name: string; count: number }>
    }>
  }
  switch_config?: any
  manual_connections?: ManualConnectionConfig
  interconnect_config?: {
    c2c?: InterconnectParams  // Chip-to-Chip
    b2b?: InterconnectParams  // Board-to-Board
    r2r?: InterconnectParams  // Rack-to-Rack
    p2p?: InterconnectParams  // Pod-to-Pod
  }
}

// 拓扑状态接口
export interface TopologyContextType {
  topology: HierarchicalTopology | null
  setTopology: React.Dispatch<React.SetStateAction<HierarchicalTopology | null>>
  loading: boolean
  loadTopology: () => Promise<void>
  handleGenerate: (config: GenerateConfig) => Promise<void>
  // 用于部署分析的配置数据
  rackConfig: RackConfigForAnalysis | null
  podCount: number
  racksPerPod: number
}

// ============================================
// Context 创建
// ============================================
const TopologyContext = createContext<TopologyContextType | null>(null)

export const useTopology = () => {
  const context = useContext(TopologyContext)
  if (!context) {
    throw new Error('useTopology must be used within TopologyProvider')
  }
  return context
}

// ============================================
// Provider 实现
// ============================================
interface TopologyProviderProps {
  children: ReactNode
}

export const TopologyProvider: React.FC<TopologyProviderProps> = ({ children }) => {
  const [topology, setTopology] = useState<HierarchicalTopology | null>(null)
  const [loading, setLoading] = useState(true)
  // 用于部署分析的配置数据
  const [rackConfig, setRackConfig] = useState<RackConfigForAnalysis | null>(null)
  const [podCount, setPodCount] = useState(1)
  const [racksPerPod, setRacksPerPod] = useState(1)

  // 加载拓扑
  const loadTopology = useCallback(async () => {
    setLoading(true)
    try {
      const cachedStr = localStorage.getItem(CONFIG_CACHE_KEY)
      if (cachedStr) {
        const cached = JSON.parse(cachedStr)
        const data = await generateTopology({
          pod_count: cached.podCount,
          racks_per_pod: cached.racksPerPod,
          rack_config: cached.rackConfig,
          switch_config: cached.switchConfig,
          manual_connections: cached.manualConnectionConfig,
          interconnect_config: cached.interconnect?.links || cached.hardwareParams?.interconnect,
        })
        setTopology(data)
        // 保存用于部署分析的配置数据
        setPodCount(cached.podCount || 1)
        setRacksPerPod(cached.racksPerPod || 1)
        if (cached.rackConfig) {
          setRackConfig(cached.rackConfig)
        }
      } else {
        const data = await getTopology()
        setTopology(data)
      }
    } catch (error) {
      console.error('加载拓扑失败:', error)
      toast.error('加载拓扑数据失败')
    } finally {
      setLoading(false)
    }
  }, [])

  // 生成拓扑
  const handleGenerate = useCallback(async (config: GenerateConfig) => {
    try {
      const data = await generateTopology(config)
      setTopology(data)
      // 更新用于部署分析的配置数据
      setPodCount(config.pod_count || 1)
      setRacksPerPod(config.racks_per_pod || 1)
      if (config.rack_config) {
        setRackConfig(config.rack_config as RackConfigForAnalysis)
      }
    } catch (error) {
      console.error('生成拓扑失败:', error)
      toast.error('生成拓扑失败')
    }
  }, [])

  // 初始加载
  useEffect(() => {
    loadTopology()
  }, [loadTopology])

  const contextValue: TopologyContextType = {
    topology,
    setTopology,
    loading,
    loadTopology,
    handleGenerate,
    rackConfig,
    podCount,
    racksPerPod,
  }

  return (
    <TopologyContext.Provider value={contextValue}>
      {children}
    </TopologyContext.Provider>
  )
}

export default TopologyContext
