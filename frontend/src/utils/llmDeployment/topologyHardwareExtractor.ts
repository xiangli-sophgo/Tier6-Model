/**
 * 拓扑硬件提取器
 *
 * 从拓扑配置中提取硬件信息，生成部署分析所需的 HardwareConfig
 */

import { HierarchicalTopology, ConnectionConfig } from '../../types'
import { FlexBoardChipConfig } from '../../components/ConfigPanel/shared'
import { ChipHardwareConfig, HardwareConfig, BoardConfig, RackConfig, PodConfig } from './types'
import { getChipConfig } from './presets'

/**
 * 芯片组信息
 */
export interface ChipGroupInfo {
  /** 芯片类型名称 */
  chipType: string
  /** 预设ID (如 'h100-sxm') */
  presetId?: string
  /** 芯片性能参数 */
  chipConfig: ChipHardwareConfig
  /** 该类型芯片总数 */
  totalCount: number
  /** 包含该类型芯片的 Board 数 */
  boardCount: number
  /** 每个 Board 的芯片数量 (用于计算节点内芯片数) */
  chipsPerBoard: number
}

/**
 * 拓扑硬件摘要
 */
export interface TopologyHardwareSummary {
  /** 按芯片类型分组的硬件配置 */
  chipGroups: ChipGroupInfo[]
  /** 拓扑层级统计 */
  totalPods: number
  totalRacks: number
  totalBoards: number
  totalChips: number
  /** 连接带宽信息 */
  intraNodeBandwidthGbps: number   // 节点内互联带宽 (Board层连接)
  interNodeBandwidthGbps: number   // 节点间互联带宽 (Rack/Pod层连接)
  intraNodeLatencyUs: number       // 节点内延迟
  interNodeLatencyUs: number       // 节点间延迟
}

/**
 * 从拓扑配置提取硬件摘要
 */
export function extractHardwareSummary(topology: HierarchicalTopology): TopologyHardwareSummary {
  // 统计层级数量
  const totalPods = topology.pods.length
  let totalRacks = 0
  let totalBoards = 0
  let totalChips = 0

  // 收集芯片配置并分组
  const chipGroupMap = new Map<string, {
    chipType: string
    presetId?: string
    chipConfig: ChipHardwareConfig
    totalCount: number
    boardCount: number
    chipsPerBoardList: number[]
  }>()

  for (const pod of topology.pods) {
    totalRacks += pod.racks.length
    for (const rack of pod.racks) {
      totalBoards += rack.boards.length
      for (const board of rack.boards) {
        // 遍历 Board 上的芯片，统计总数
        totalChips += board.chips.length
      }
    }
  }

  // 如果拓扑中没有扩展的芯片配置信息，尝试从默认配置推断
  // 这需要查看 topology 的原始配置数据

  // 提取连接带宽信息
  const { intraNodeBandwidth, interNodeBandwidth, intraNodeLatency, interNodeLatency } =
    extractBandwidthFromConnections(topology.connections)

  // 构建芯片组列表
  const chipGroups: ChipGroupInfo[] = []
  for (const [, group] of chipGroupMap) {
    const avgChipsPerBoard = group.chipsPerBoardList.length > 0
      ? Math.round(group.chipsPerBoardList.reduce((a, b) => a + b, 0) / group.chipsPerBoardList.length)
      : 8
    chipGroups.push({
      chipType: group.chipType,
      presetId: group.presetId,
      chipConfig: group.chipConfig,
      totalCount: group.totalCount,
      boardCount: group.boardCount,
      chipsPerBoard: avgChipsPerBoard,
    })
  }

  return {
    chipGroups,
    totalPods,
    totalRacks,
    totalBoards,
    totalChips,
    intraNodeBandwidthGbps: intraNodeBandwidth,
    interNodeBandwidthGbps: interNodeBandwidth,
    intraNodeLatencyUs: intraNodeLatency,
    interNodeLatencyUs: interNodeLatency,
  }
}

/**
 * 从连接配置中提取带宽信息
 */
function extractBandwidthFromConnections(connections: ConnectionConfig[]): {
  intraNodeBandwidth: number
  interNodeBandwidth: number
  intraNodeLatency: number
  interNodeLatency: number
} {
  let intraNodeBandwidth = 900   // 默认 NVLink 4.0
  let interNodeBandwidth = 400   // 默认 NDR 400G
  let intraNodeLatency = 1       // 默认 1us
  let interNodeLatency = 2       // 默认 2us

  // 统计不同类型连接的带宽
  const intraBandwidths: number[] = []
  const interBandwidths: number[] = []
  const intraLatencies: number[] = []
  const interLatencies: number[] = []

  for (const conn of connections) {
    if (conn.bandwidth) {
      // 根据连接类型分类
      // c2c/b2b: 节点内连接 (Chip间、Board间)
      // r2r/p2p: 节点间连接 (Rack间、Pod间)
      if (conn.type === 'c2c' || conn.type === 'b2b') {
        // 节点内连接
        intraBandwidths.push(conn.bandwidth)
        if (conn.latency) intraLatencies.push(conn.latency) // us
      } else if (conn.type === 'r2r' || conn.type === 'p2p') {
        // 节点间连接
        interBandwidths.push(conn.bandwidth)
        if (conn.latency) interLatencies.push(conn.latency) // us
      }
    }
  }

  // 取平均值
  if (intraBandwidths.length > 0) {
    intraNodeBandwidth = Math.round(intraBandwidths.reduce((a, b) => a + b, 0) / intraBandwidths.length)
  }
  if (interBandwidths.length > 0) {
    interNodeBandwidth = Math.round(interBandwidths.reduce((a, b) => a + b, 0) / interBandwidths.length)
  }
  if (intraLatencies.length > 0) {
    intraNodeLatency = Math.round(intraLatencies.reduce((a, b) => a + b, 0) / intraLatencies.length * 10) / 10
  }
  if (interLatencies.length > 0) {
    interNodeLatency = Math.round(interLatencies.reduce((a, b) => a + b, 0) / interLatencies.length * 10) / 10
  }

  return { intraNodeBandwidth, interNodeBandwidth, intraNodeLatency, interNodeLatency }
}

/**
 * 从 FlexBoardChipConfig 列表提取芯片组信息
 *
 * 优先从保存的配置中读取硬件参数，不依赖后端 API
 */
export function extractChipGroupsFromConfig(
  boards: Array<{ chips: FlexBoardChipConfig[], count: number }>,
  hardwareParams?: { chips?: Record<string, any> }
): ChipGroupInfo[] {
  const chipGroupMap = new Map<string, {
    chipType: string
    presetId?: string
    chipConfig: ChipHardwareConfig
    totalCount: number
    boardCount: number
    chipsPerBoardList: number[]
  }>()

  for (const board of boards) {
    const boardCount = board.count || 1
    for (const chip of board.chips) {
      // 使用芯片名称作为唯一 key，确保相同名称的芯片合并到一个组
      const key = chip.name
      const existing = chipGroupMap.get(key)

      // 从预设或默认值获取硬件参数
      let chipConfig: ChipHardwareConfig

      // SG2260E 默认微架构参数
      const defaultMicroArch = {
        cube_m: 16,
        cube_k: 32,
        cube_n: 8,
        sram_size_kb: 2048,
        sram_utilization: 0.45,
        lane_num: 16,
        align_bytes: 32,
        compute_dma_overlap_rate: 0.8,
      }

      // 默认芯片配置
      const defaultChipConfig: ChipHardwareConfig = {
        name: chip.name,
        num_cores: 256,
        compute_tflops_fp8: 1600, // 默认 1600 TFLOPs FP8
        compute_tflops_bf16: 800, // 默认 800 TFLOPs BF16
        memory_capacity_gb: 64,
        memory_bandwidth_gbps: 1200,
        memory_bandwidth_utilization: 0.85,
        lmem_capacity_mb: 128,
        lmem_bandwidth_gbps: 12000,
        ...defaultMicroArch,
      }

      // 方案1: 从拓扑配置的 hardware_params.chips 直接获取（优先级最高）
      if (hardwareParams?.chips?.[chip.name]) {
        chipConfig = hardwareParams.chips[chip.name] as ChipHardwareConfig
      }
      // 方案2: 从预设获取（后端预设 + 自定义预设）
      else if (chip.preset_id) {
        const preset = getChipConfig(chip.preset_id)
        if (preset) {
          chipConfig = preset
        } else {
          // 预设ID找不到，使用合理的默认值并警告
          console.warn(`芯片预设 '${chip.preset_id}' 未找到，使用默认参数`)
          chipConfig = defaultChipConfig
        }
      }
      // 方案3: 使用默认值
      else {
        console.warn(`芯片 '${chip.name}' 缺少硬件参数，请在拓扑设置的Chip层中完善配置`)
        chipConfig = defaultChipConfig
      }

      if (existing) {
        existing.totalCount += chip.count * boardCount
        existing.boardCount += boardCount
        existing.chipsPerBoardList.push(chip.count)
      } else {
        chipGroupMap.set(key, {
          chipType: chip.name,
          presetId: chip.preset_id,
          chipConfig,
          totalCount: chip.count * boardCount,
          boardCount: boardCount,
          chipsPerBoardList: [chip.count],
        })
      }
    }
  }

  const chipGroups: ChipGroupInfo[] = []
  for (const [, group] of chipGroupMap) {
    const avgChipsPerBoard = group.chipsPerBoardList.length > 0
      ? Math.round(group.chipsPerBoardList.reduce((a, b) => a + b, 0) / group.chipsPerBoardList.length)
      : 8
    chipGroups.push({
      chipType: group.chipType,
      presetId: group.presetId,
      chipConfig: group.chipConfig,
      totalCount: group.totalCount,
      boardCount: group.boardCount,
      chipsPerBoard: avgChipsPerBoard,
    })
  }

  return chipGroups
}

/**
 * 从拓扑配置生成硬件配置
 *
 * @param summary 拓扑硬件摘要
 * @param selectedChipType 选中的芯片类型 (当有多种芯片时需要指定)
 */
export function generateHardwareConfig(
  summary: TopologyHardwareSummary,
  selectedChipType?: string
): HardwareConfig | null {
  // 找到选中的芯片组
  let chipGroup: ChipGroupInfo | undefined
  if (selectedChipType) {
    chipGroup = summary.chipGroups.find(g => g.chipType === selectedChipType || g.presetId === selectedChipType)
  } else if (summary.chipGroups.length === 1) {
    chipGroup = summary.chipGroups[0]
  } else if (summary.chipGroups.length > 1) {
    // 多种芯片类型，取数量最多的
    chipGroup = summary.chipGroups.reduce((a, b) => a.totalCount > b.totalCount ? a : b)
  }

  if (!chipGroup) {
    return null
  }

  // 构建新格式配置
  const chips: Record<string, ChipHardwareConfig> = {
    [chipGroup.chipType]: chipGroup.chipConfig
  }


  // 如果有多种芯片类型，都添加进去
  if (summary.chipGroups.length > 1) {
    summary.chipGroups.forEach(group => {
      if (group.chipType !== chipGroup!.chipType) {
        chips[group.chipType] = group.chipConfig
      }
    })
  }

  // console.log(' generateHardwareConfig: 最终chips字典 =', chips)

  // 构建互联配置
  const interconnect = {
    c2c: {
      bandwidth_gbps: summary.intraNodeBandwidthGbps,
      latency_us: summary.intraNodeLatencyUs
    },
    b2b: {
      bandwidth_gbps: summary.intraNodeBandwidthGbps,
      latency_us: summary.intraNodeLatencyUs
    },
    r2r: {
      bandwidth_gbps: summary.interNodeBandwidthGbps,
      latency_us: summary.interNodeLatencyUs
    },
    p2p: {
      bandwidth_gbps: summary.interNodeBandwidthGbps,
      latency_us: summary.interNodeLatencyUs
    }
  }

  const result = {
    hardware_params: {
      chips,
      interconnect
    }
  }


  return result
}

/**
 * 从 ConfigPanel 的配置直接生成硬件配置
 *
 * 这是一个便捷函数，用于从 ConfigPanel 的配置直接生成 HardwareConfig
 */
export function generateHardwareConfigFromPanelConfig(
  podCount: number,
  racksPerPod: number,
  boards: Array<{ chips: FlexBoardChipConfig[], count: number }>,
  connections: ConnectionConfig[],
  selectedChipType?: string,
  hardwareParams?: { chips?: Record<string, any> }
): HardwareConfig | null {
  // 提取芯片组
  const chipGroups = extractChipGroupsFromConfig(boards, hardwareParams)

  // 计算总 Board 数
  const totalBoards = podCount * racksPerPod * boards.reduce((sum, b) => sum + (b.count || 1), 0)

  // 提取带宽信息
  const { intraNodeBandwidth, interNodeBandwidth, intraNodeLatency, interNodeLatency } =
    extractBandwidthFromConnections(connections)

  // 更新芯片组的 boardCount
  for (const group of chipGroups) {
    group.boardCount = Math.round(group.boardCount * podCount * racksPerPod)
  }

  // 构建摘要
  const summary: TopologyHardwareSummary = {
    chipGroups,
    totalPods: podCount,
    totalRacks: podCount * racksPerPod,
    totalBoards,
    totalChips: chipGroups.reduce((sum, g) => sum + g.totalCount * podCount * racksPerPod, 0),
    intraNodeBandwidthGbps: intraNodeBandwidth,
    interNodeBandwidthGbps: interNodeBandwidth,
    intraNodeLatencyUs: intraNodeLatency,
    interNodeLatencyUs: interNodeLatency,
  }

  return generateHardwareConfig(summary, selectedChipType)
}
