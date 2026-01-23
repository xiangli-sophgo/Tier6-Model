/**
 * useNodePositions Hook
 * 计算拓扑中所有节点（Pod、Rack、Board）的世界坐标
 */
import { useMemo } from 'react'
import * as THREE from 'three'
import { HierarchicalTopology, RACK_DIMENSIONS } from '../types'

export interface NodePositions {
  pods: Map<string, THREE.Vector3>      // Pod中心位置
  racks: Map<string, THREE.Vector3>     // Rack位置
  boards: Map<string, THREE.Vector3>    // Board世界坐标
}

/**
 * 计算所有节点的世界坐标
 * @param topology - 层级拓扑数据
 * @returns 节点位置映射（pods, racks, boards）
 */
export const useNodePositions = (topology: HierarchicalTopology | null): NodePositions => {
  return useMemo(() => {
    if (!topology) {
      return { pods: new Map(), racks: new Map(), boards: new Map() }
    }

    const rackSpacingX = 1.5
    const rackSpacingZ = 2
    const { uHeight, totalU } = RACK_DIMENSIONS
    const rackHeight = totalU * uHeight

    const pods = new Map<string, THREE.Vector3>()
    const racks = new Map<string, THREE.Vector3>()
    const boards = new Map<string, THREE.Vector3>()

    // 计算Pod布局参数
    const firstPod = topology.pods[0]
    let podSpacingX = 6
    let podSpacingZ = 4
    let podCols = 2

    if (firstPod) {
      const rackCols = firstPod.grid_size[1]
      const rackRows = firstPod.grid_size[0]
      podSpacingX = rackCols * rackSpacingX + 2
      podSpacingZ = rackRows * rackSpacingZ + 1

      const totalPods = topology.pods.length
      if (totalPods <= 2) podCols = totalPods
      else if (totalPods <= 4) podCols = 2
      else if (totalPods <= 6) podCols = 3
      else if (totalPods <= 9) podCols = 3
      else podCols = 4
    }

    const getPodGridPosition = (podIndex: number) => {
      const row = Math.floor(podIndex / podCols)
      const col = podIndex % podCols
      return { row, col }
    }

    // 首先计算所有Rack位置以找出中心
    let minX = Infinity, maxX = -Infinity
    let minZ = Infinity, maxZ = -Infinity

    topology.pods.forEach((pod, podIndex) => {
      const { row, col } = getPodGridPosition(podIndex)
      const podOffsetX = col * podSpacingX
      const podOffsetZ = row * podSpacingZ
      pod.racks.forEach(rack => {
        const x = podOffsetX + rack.position[1] * rackSpacingX
        const z = podOffsetZ + rack.position[0] * rackSpacingZ
        minX = Math.min(minX, x)
        maxX = Math.max(maxX, x)
        minZ = Math.min(minZ, z)
        maxZ = Math.max(maxZ, z)
      })
    })

    const centerX = (minX + maxX) / 2
    const centerZ = (minZ + maxZ) / 2

    // 设置所有节点位置
    topology.pods.forEach((pod, podIndex) => {
      const { row, col } = getPodGridPosition(podIndex)
      const podOffsetX = col * podSpacingX
      const podOffsetZ = row * podSpacingZ

      let podSumX = 0, podSumZ = 0, podCount = 0

      pod.racks.forEach(rack => {
        const rackX = podOffsetX + rack.position[1] * rackSpacingX - centerX
        const rackZ = podOffsetZ + rack.position[0] * rackSpacingZ - centerZ
        racks.set(rack.id, new THREE.Vector3(rackX, 0, rackZ))

        podSumX += rackX
        podSumZ += rackZ
        podCount++

        // 计算Board位置
        rack.boards.forEach(board => {
          const boardY = (board.u_position - 1) * uHeight + (board.u_height * uHeight) / 2 - rackHeight / 2
          boards.set(board.id, new THREE.Vector3(rackX, boardY, rackZ))
        })
      })

      // Pod中心
      if (podCount > 0) {
        pods.set(pod.id, new THREE.Vector3(podSumX / podCount, 0, podSumZ / podCount))
      }
    })

    return { pods, racks, boards }
  }, [topology])
}
