/**
 * 知识图谱力导向布局计算 Worker
 * 在后台线程执行 300 次 D3 force simulation tick，避免阻塞主线程
 */

import * as d3Force from 'd3-force'

// 知识图谱力导向布局参数
const KNOWLEDGE_FORCE_CONFIG = {
  chargeStrength: -400,
  chargeDistanceMax: 500,
  linkDistance: 80,
  linkStrength: 0.3,
  radialStrength: 0.1,
  radialMinRadius: 50,
  radialMaxRadius: 500,
  centerStrength: 0.05,
  collisionRadius: 35,
  collisionStrength: 1,
  collisionIterations: 4,
  warmupTicks: 300,
}

// 消息类型定义
interface WorkerInput {
  type: 'compute'
  data: {
    nodes: Array<{
      id: string
      name: string
      definition: string
      category: string
      x?: number
      y?: number
      vx?: number
      vy?: number
    }>
    relations: Array<{
      source: string
      target: string
    }>
    degreeMap: Record<string, number>
    maxDegree: number
    centerX: number
    centerY: number
  }
}

interface WorkerOutput {
  type: 'success' | 'error'
  data?: {
    nodes: Array<{
      id: string
      x: number
      y: number
      vx: number
      vy: number
    }>
    viewBox: {
      x: number
      y: number
      width: number
      height: number
    }
  }
  error?: string
}

// Worker 消息处理
self.onmessage = (e: MessageEvent<WorkerInput>) => {
  try {
    if (e.data.type !== 'compute') {
      throw new Error(`Unknown message type: ${e.data.type}`)
    }

    const { nodes: inputNodes, relations, degreeMap, maxDegree, centerX, centerY } = e.data.data

    // 创建节点副本（Worker 中的数据）
    const nodes = inputNodes.map(node => ({
      ...node,
      x: node.x ?? centerX,
      y: node.y ?? centerY,
      vx: node.vx ?? 0,
      vy: node.vy ?? 0,
    }))

    // 创建力导向模拟
    const simulation = d3Force.forceSimulation(nodes as any)
      .force('charge', d3Force.forceManyBody()
        .strength(KNOWLEDGE_FORCE_CONFIG.chargeStrength)
        .distanceMax(KNOWLEDGE_FORCE_CONFIG.chargeDistanceMax)
      )
      .force('link', d3Force.forceLink(relations.map(r => ({ source: r.source, target: r.target })))
        .id((d: any) => d.id)
        .distance(KNOWLEDGE_FORCE_CONFIG.linkDistance)
        .strength(KNOWLEDGE_FORCE_CONFIG.linkStrength)
      )
      .force('radial', d3Force.forceRadial(
        (d: any) => {
          const degree = degreeMap[d.id] || 0
          const radiusFactor = Math.pow(1 - degree / maxDegree, 1.2)
          return KNOWLEDGE_FORCE_CONFIG.radialMinRadius +
                 radiusFactor * (KNOWLEDGE_FORCE_CONFIG.radialMaxRadius - KNOWLEDGE_FORCE_CONFIG.radialMinRadius)
        },
        centerX,
        centerY
      ).strength(KNOWLEDGE_FORCE_CONFIG.radialStrength))
      .force('centerX', d3Force.forceX(centerX).strength(KNOWLEDGE_FORCE_CONFIG.centerStrength))
      .force('centerY', d3Force.forceY(centerY).strength(KNOWLEDGE_FORCE_CONFIG.centerStrength))
      .force('collision', d3Force.forceCollide()
        .radius(KNOWLEDGE_FORCE_CONFIG.collisionRadius)
        .strength(KNOWLEDGE_FORCE_CONFIG.collisionStrength)
        .iterations(KNOWLEDGE_FORCE_CONFIG.collisionIterations)
      )
      .stop()

    // Warmup：执行 300 次 tick（在 Worker 线程中，不阻塞主线程）
    for (let i = 0; i < KNOWLEDGE_FORCE_CONFIG.warmupTicks; i++) {
      simulation.tick()
    }

    // 清零速度确保静止
    nodes.forEach(node => {
      node.vx = 0
      node.vy = 0
    })

    // 计算适合所有节点的视口
    const padding = 100
    const minX = Math.min(...nodes.map(n => n.x ?? 0)) - padding
    const maxX = Math.max(...nodes.map(n => n.x ?? 0)) + padding
    const minY = Math.min(...nodes.map(n => n.y ?? 0)) - padding
    const maxY = Math.max(...nodes.map(n => n.y ?? 0)) + padding
    const width = Math.max(maxX - minX, 400)
    const height = Math.max(maxY - minY, 300)

    // 返回计算结果
    const output: WorkerOutput = {
      type: 'success',
      data: {
        nodes: nodes.map(n => ({
          id: n.id,
          x: n.x ?? centerX,
          y: n.y ?? centerY,
          vx: n.vx ?? 0,
          vy: n.vy ?? 0,
        })),
        viewBox: {
          x: minX,
          y: minY,
          width,
          height,
        },
      },
    }

    self.postMessage(output)
  } catch (error) {
    const output: WorkerOutput = {
      type: 'error',
      error: error instanceof Error ? error.message : String(error),
    }
    self.postMessage(output)
  }
}

// 导出空对象（避免 TS 报错）
export {}
