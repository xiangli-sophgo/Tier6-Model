/**
 * 配置名称生成工具函数
 *
 * 用于根据配置内容生成标准化的名称
 */

// 重新导出 generateBenchmarkName，统一使用 benchmarkNaming.ts 的实现
export { generateBenchmarkName, parseBenchmarkParts } from './llmDeployment/benchmarkNaming'

/**
 * 生成 Topology 名称
 *
 * 格式: P{Pods}-R{Racks}-B{TotalBoards}-C{TotalChips}
 * 示例: P1-R4-B32-C256
 *
 * 从 grouped_pods 格式 (pods[].racks[].boards[].chips[]) 计算统计量。
 */
export function generateTopologyName(topology: { pods?: Array<{ count?: number; racks?: Array<{ count?: number; boards?: Array<{ count?: number; chips?: Array<{ count?: number }> }> }> }> }): string {
  const pods = topology.pods || []
  let totalPods = 0
  let totalRacks = 0
  let totalBoards = 0
  let totalChips = 0

  for (const podGroup of pods) {
    const pc = podGroup.count ?? 1
    totalPods += pc
    for (const rackGroup of podGroup.racks ?? []) {
      const rc = rackGroup.count ?? 1
      totalRacks += pc * rc
      for (const board of rackGroup.boards ?? []) {
        const bc = board.count ?? 1
        totalBoards += pc * rc * bc
        for (const chip of board.chips ?? []) {
          totalChips += pc * rc * bc * (chip.count ?? 1)
        }
      }
    }
  }

  return `P${totalPods}-R${totalRacks}-B${totalBoards}-C${totalChips}`
}
