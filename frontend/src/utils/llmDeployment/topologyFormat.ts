/**
 * 拓扑配置格式工具
 *
 * grouped_pods 格式: pods[].racks[].boards[].chips[], 每级带 count
 */

import type { TopologyConfig, TopologyPodGroup } from '../../types/math_model'

/** 统计 grouped_pods 格式的芯片总数 */
export function countChips(config: TopologyConfig): number {
  const pods = config.pods
  if (!pods || pods.length === 0) return 0

  let total = 0
  for (const podGroup of pods) {
    const podCount = podGroup.count ?? 1
    for (const rackGroup of podGroup.racks ?? []) {
      const rackCount = rackGroup.count ?? 1
      for (const board of rackGroup.boards ?? []) {
        const boardCount = board.count ?? 1
        for (const chipGroup of board.chips ?? []) {
          const chipCount = chipGroup.count ?? 1
          total += podCount * rackCount * boardCount * chipCount
        }
      }
    }
  }
  return total
}

/** 统计 Board 总数 */
export function countBoards(config: TopologyConfig): number {
  const pods = config.pods
  if (!pods || pods.length === 0) return 0

  let total = 0
  for (const podGroup of pods) {
    const podCount = podGroup.count ?? 1
    for (const rackGroup of podGroup.racks ?? []) {
      const rackCount = rackGroup.count ?? 1
      for (const board of rackGroup.boards ?? []) {
        total += podCount * rackCount * (board.count ?? 1)
      }
    }
  }
  return total
}

/** 统计 Rack 总数 */
export function countRacks(config: TopologyConfig): number {
  const pods = config.pods
  if (!pods || pods.length === 0) return 0

  let total = 0
  for (const podGroup of pods) {
    const podCount = podGroup.count ?? 1
    for (const rackGroup of podGroup.racks ?? []) {
      total += podCount * (rackGroup.count ?? 1)
    }
  }
  return total
}

/** 统计 Pod 总数 */
export function countPods(config: TopologyConfig): number {
  const pods = config.pods
  if (!pods || pods.length === 0) return 0

  let total = 0
  for (const podGroup of pods) {
    total += podGroup.count ?? 1
  }
  return total
}

/** 创建默认的 pods 结构 (1 Pod, 1 Rack, 1 Board, 8 chips) */
export function createDefaultPods(chipName = 'SG2262', chipCount = 8): TopologyPodGroup[] {
  return [
    {
      count: 1,
      racks: [
        {
          count: 1,
          boards: [
            {
              count: 1,
              chips: [{ name: chipName, count: chipCount }],
            },
          ],
        },
      ],
    },
  ]
}
