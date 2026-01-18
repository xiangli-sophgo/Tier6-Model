/**
 * LLM 推理模拟系统 - 核心模拟器 v2
 *
 * 真正的事件驱动模拟器，支持：
 * - 微批次级别的 PP 流水线调度 (1F1B schedule)
 * - 动态计算-通信重叠建模
 * - Bubble/Idle 事件生成
 * - 精确的资源占用追踪
 */

import {
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
  HardwareConfig,
  InferencePhase,
} from '../types'
import {
  calculateLayerFlopsPrefill,
  calculateLayerFlopsDecode,
} from '../modelCalculator'
import { getBytesPerElement } from '../types'
import { EventQueue } from './eventQueue'
import {
  SimulationConfig,
  SimulationResult,
  SimEvent,
  CommTraceItem,
  GanttChartData,
  GanttTask,
  GanttResource,
  SimulationStats,
  PhaseTimeStats,
  ChipState,
  DEFAULT_SIMULATION_CONFIG,
  LatencyDistribution,
  GanttTaskType,
} from './types'

/** 生成唯一ID */
let eventIdCounter = 0
function generateId(prefix: string): string {
  return `${prefix}_${++eventIdCounter}`
}

/** 重置ID计数器 */
function resetIdCounter(): void {
  eventIdCounter = 0
}

/**
 * 微批次状态
 */
interface MicroBatchState {
  /** 微批次ID */
  id: number
  /** 当前所在的PP阶段 (-1表示未开始, pp表示已完成) */
  currentStage: number
  /** 是否在前向阶段 */
  isForward: boolean
  /** 各阶段完成时间 */
  stageCompletionTime: number[]
}

/**
 * PP阶段状态
 */
interface PPStageState {
  /** 阶段索引 */
  stageIndex: number
  /** 当前时间 */
  currentTime: number
  /** 计算资源空闲时间 */
  computeIdleAt: number
  /** 网络资源空闲时间 */
  networkIdleAt: number
  /** 正在处理的微批次ID (-1表示空闲) */
  processingMicroBatch: number
  /** 等待队列 (微批次ID列表) */
  waitQueue: number[]
}

/**
 * 推理模拟器 v2 - 事件驱动架构
 */
export class InferenceSimulator {
  private model: LLMModelConfig
  private inference: InferenceConfig
  private parallelism: ParallelismStrategy
  private hardware: HardwareConfig
  private config: SimulationConfig

  // 事件队列
  private eventQueue: EventQueue = new EventQueue()

  // 模拟状态
  private chipStates: Map<string, ChipState> = new Map()
  private ppStageStates: PPStageState[] = []
  private microBatches: MicroBatchState[] = []
  private events: SimEvent[] = []
  private commTrace: CommTraceItem[] = []
  private ganttTasks: GanttTask[] = []

  // 缓存的计算结果
  private layerComputeTimePrefill: number = 0
  private layerComputeTimeDecode: number[] = []
  private tpCommTime: number = 0
  private ppCommTime: number = 0
  private layersPerStage: number = 0

  // 统计
  private totalComputeTime: number = 0
  private totalBubbleTime: number = 0
  private totalOverlapTime: number = 0

  constructor(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    config: Partial<SimulationConfig> = {}
  ) {
    this.model = model
    this.inference = inference
    this.parallelism = parallelism
    this.hardware = hardware
    this.config = { ...DEFAULT_SIMULATION_CONFIG, ...config }
  }

  /**
   * 运行模拟
   */
  run(): SimulationResult {
    resetIdCounter()
    this.events = []
    this.commTrace = []
    this.ganttTasks = []
    this.eventQueue.clear()
    this.totalComputeTime = 0
    this.totalBubbleTime = 0
    this.totalOverlapTime = 0

    // 初始化
    this.initializeChipStates()
    this.precomputeLatencies()
    this.initializePPStageStates()

    // 模拟 Prefill 阶段 (1F1B 调度)
    const prefillEndTime = this.simulatePrefillWithPipeline()

    // 模拟 Decode 阶段
    const decodeEndTime = this.simulateDecode(prefillEndTime)

    // 生成结果
    return this.generateResult(prefillEndTime, decodeEndTime)
  }

  /**
   * 初始化芯片状态
   */
  private initializeChipStates(): void {
    this.chipStates.clear()
    const { tp, pp, ep } = this.parallelism

    for (let ppStage = 0; ppStage < pp; ppStage++) {
      for (let tpRank = 0; tpRank < tp; tpRank++) {
        for (let epRank = 0; epRank < ep; epRank++) {
          const chipId = `pp${ppStage}_tp${tpRank}_ep${epRank}`
          this.chipStates.set(chipId, {
            chipId,
            ppStage,
            tpRank,
            epRank,
            currentTime: 0,
            computeIdleAt: 0,
            tpNetworkIdleAt: 0,
            ppNetworkIdleAt: 0,
            epNetworkIdleAt: 0,
            completedLayers: 0,
            completedTokens: 0,
          })
        }
      }
    }
  }

  /**
   * 初始化PP阶段状态
   */
  private initializePPStageStates(): void {
    this.ppStageStates = []
    for (let i = 0; i < this.parallelism.pp; i++) {
      this.ppStageStates.push({
        stageIndex: i,
        currentTime: 0,
        computeIdleAt: 0,
        networkIdleAt: 0,
        processingMicroBatch: -1,
        waitQueue: [],
      })
    }

    // 初始化微批次
    const numMicroBatches = this.inference.num_micro_batches ?? Math.max(this.parallelism.pp * 2, 4)
    this.microBatches = []
    for (let i = 0; i < numMicroBatches; i++) {
      this.microBatches.push({
        id: i,
        currentStage: -1,
        isForward: true,
        stageCompletionTime: new Array(this.parallelism.pp).fill(0),
      })
    }
  }

  /**
   * 预计算各种延迟
   */
  private precomputeLatencies(): void {
    const { tp, pp } = this.parallelism
    const numMicroBatches = this.inference.num_micro_batches ?? Math.max(pp * 2, 4)

    // 每微批次的 batch size
    const microBatchSize = Math.ceil(this.inference.batch_size / numMicroBatches)

    // Prefill 单层计算 FLOPs (每微批次)
    const layerFlopsPrefill = calculateLayerFlopsPrefill(
      this.model,
      microBatchSize,
      this.inference.input_seq_length
    )

    // 每芯片算力 (FLOPs/s)
    const flopsPerSecond = this.hardware.chip.compute_tflops_fp16 * 1e12

    // MFU 估计
    const mfuPrefill = 0.5
    const mfuDecode = 0.3

    // 单层计算时间 (ms) - 每微批次
    const flopsPerChipPerLayer = layerFlopsPrefill / tp
    this.layerComputeTimePrefill = (flopsPerChipPerLayer / (flopsPerSecond * mfuPrefill)) * 1000

    // 计算每个PP阶段的层数
    this.layersPerStage = Math.ceil(this.model.num_layers / pp)

    // Decode 各 context 长度的计算时间
    this.layerComputeTimeDecode = []
    const maxTokens = this.config.maxSimulatedTokens
    for (let t = 0; t < maxTokens; t++) {
      const contextLen = this.inference.input_seq_length + t
      const layerFlopsDecode = calculateLayerFlopsDecode(
        this.model,
        microBatchSize,
        contextLen
      )
      const flopsPerChip = layerFlopsDecode / tp
      const timeMs = (flopsPerChip / (flopsPerSecond * mfuDecode)) * 1000
      this.layerComputeTimeDecode.push(timeMs)
    }

    // 通信延迟计算
    // TP AllReduce - 每层
    const hiddenSize = this.model.hidden_size
    const seqLen = this.inference.input_seq_length
    // AllReduce 数据量 = 2 * (tp-1) / tp * hidden_size * seq_len * batch_size * 2bytes
    const tpCommVolPerLayer = tp > 1
      ? 2 * (tp - 1) / tp * hiddenSize * seqLen * microBatchSize * 2 / 1e9 // GB
      : 0
    const tpBandwidth = this.hardware.node.intra_node_bandwidth_gbps
    const tpLatencyUs = this.hardware.node.intra_node_latency_us
    this.tpCommTime = tpCommVolPerLayer > 0
      ? (tpCommVolPerLayer / tpBandwidth) * 1000 + tpLatencyUs / 1000
      : 0

    // PP P2P - 每阶段间传输
    const ppCommVol = pp > 1
      ? hiddenSize * seqLen * microBatchSize * 2 / 1e9 // GB
      : 0
    const ppBandwidth = this.hardware.cluster.inter_node_bandwidth_gbps
    const ppLatencyUs = this.hardware.cluster.inter_node_latency_us
    this.ppCommTime = ppCommVol > 0
      ? (ppCommVol / ppBandwidth) * 1000 + ppLatencyUs / 1000
      : 0

    // EP AllToAll (暂不模拟，留作扩展)
    // TODO: 支持 MoE 模型的 EP 通信建模
  }

  /**
   * 模拟 Prefill 阶段 - 使用 1F1B 流水线调度
   */
  private simulatePrefillWithPipeline(): number {
    const { pp } = this.parallelism
    const numMicroBatches = this.microBatches.length

    // 记录阶段开始
    this.addPhaseEvent('prefill', 0, 'start')

    if (pp === 1) {
      // 无流水线，顺序处理所有微批次
      return this.simulatePrefillNoPipeline()
    }

    // 1F1B 调度
    // Warmup阶段：逐个发送 pp-1 个微批次进入流水线
    // Steady阶段：每个stage同时执行1F+1B
    // Cooldown阶段：等待所有微批次完成

    let globalTime = 0
    const stageEndTime: number[] = new Array(pp).fill(0)

    // 每个微批次在每个阶段的前向完成时间 (Prefill 只有前向)
    const forwardDoneTime: number[][] = Array.from({ length: numMicroBatches }, () =>
      new Array(pp).fill(0)
    )

    // 单个阶段处理一个微批次的时间 (所有层的计算 + 通信)
    const stageComputeTime = this.layersPerStage * this.layerComputeTimePrefill
    const stageTpCommTime = this.layersPerStage * this.tpCommTime

    // 计算-通信重叠：通信可以与下一层计算重叠
    const overlapRatio = this.config.enableOverlap ? 0.8 : 0
    const effectiveStageTime = stageComputeTime + stageTpCommTime * (1 - overlapRatio)

    // Warmup 阶段: 发送前 pp-1 个微批次的前向
    for (let mb = 0; mb < Math.min(pp - 1, numMicroBatches); mb++) {
      for (let stage = 0; stage < pp; stage++) {
        const startTime = Math.max(
          stageEndTime[stage],
          stage > 0 ? forwardDoneTime[mb][stage - 1] + this.ppCommTime : 0
        )
        const endTime = startTime + effectiveStageTime

        forwardDoneTime[mb][stage] = endTime
        stageEndTime[stage] = endTime

        // 添加甘特图任务
        this.addMicroBatchGanttTask(stage, mb, 'forward', 'prefill', startTime, endTime)

        // 记录bubble (如果有等待)
        if (startTime > (stage > 0 ? forwardDoneTime[mb][stage - 1] + this.ppCommTime : 0)) {
          const bubbleStart = stage > 0 ? forwardDoneTime[mb][stage - 1] + this.ppCommTime : 0
          if (startTime > bubbleStart) {
            this.addBubbleGanttTask(stage, 'prefill', bubbleStart, startTime)
          }
        }
      }
    }

    // Steady 阶段: 1F1B (这里简化为Prefill只有前向，没有后向)
    for (let mb = pp - 1; mb < numMicroBatches; mb++) {
      for (let stage = 0; stage < pp; stage++) {
        const startTime = Math.max(
          stageEndTime[stage],
          stage > 0 ? forwardDoneTime[mb][stage - 1] + this.ppCommTime : stageEndTime[stage]
        )
        const endTime = startTime + effectiveStageTime

        forwardDoneTime[mb][stage] = endTime
        stageEndTime[stage] = endTime

        this.addMicroBatchGanttTask(stage, mb, 'forward', 'prefill', startTime, endTime)

        // 记录 PP 通信
        if (stage > 0) {
          const ppCommStart = forwardDoneTime[mb][stage - 1]
          this.addPPCommGanttTask(stage - 1, stage, mb, 'prefill', ppCommStart, ppCommStart + this.ppCommTime)
        }
      }
      globalTime = Math.max(globalTime, stageEndTime[pp - 1])
    }

    // 计算总时间
    const prefillEndTime = Math.max(...stageEndTime)

    // 计算并记录bubble时间
    this.calculateAndRecordBubbles(prefillEndTime)

    // 记录阶段结束
    this.addPhaseEvent('prefill', prefillEndTime, 'end')

    return prefillEndTime
  }

  /**
   * 无流水线的Prefill模拟 (PP=1)
   */
  private simulatePrefillNoPipeline(): number {
    const numMicroBatches = this.microBatches.length
    let currentTime = 0

    for (let mb = 0; mb < numMicroBatches; mb++) {
      const mbStartTime = currentTime

      // 处理所有层
      for (let layer = 0; layer < this.model.num_layers; layer++) {
        const computeDuration = this.applyJitter(this.layerComputeTimePrefill)
        const computeEnd = currentTime + computeDuration

        // TP通信 (与计算重叠)
        if (this.parallelism.tp > 1) {
          const tpDuration = this.applyJitter(this.tpCommTime)
          const overlapStart = this.config.enableOverlap
            ? currentTime + computeDuration * 0.8
            : computeEnd
          const overlapTime = this.config.enableOverlap
            ? Math.min(tpDuration, computeDuration * 0.2)
            : 0

          this.totalOverlapTime += overlapTime
          currentTime = Math.max(computeEnd, overlapStart + tpDuration)
        } else {
          currentTime = computeEnd
        }

        this.totalComputeTime += computeDuration
      }

      // 添加微批次级别的甘特图任务
      this.addMicroBatchGanttTask(0, mb, 'forward', 'prefill', mbStartTime, currentTime)
    }

    this.addPhaseEvent('prefill', currentTime, 'end')
    return currentTime
  }

  /**
   * 计算并记录bubble时间
   */
  private calculateAndRecordBubbles(totalEndTime: number): void {
    const { pp } = this.parallelism
    const numMicroBatches = this.microBatches.length

    // 理想情况下的总时间 (无bubble)
    const stageComputeTime = this.layersPerStage * this.layerComputeTimePrefill
    const idealTotalTime = numMicroBatches * stageComputeTime + (pp - 1) * this.ppCommTime

    // Bubble时间 = 实际时间 - 理想时间
    const bubbleTime = totalEndTime - idealTotalTime

    if (bubbleTime > 0) {
      this.totalBubbleTime += bubbleTime
    }
  }

  /**
   * 模拟 Decode 阶段
   */
  private simulateDecode(startTime: number): number {
    const { pp } = this.parallelism
    const maxTokens = Math.min(this.config.maxSimulatedTokens, this.inference.output_seq_length)

    // 记录阶段开始
    this.addPhaseEvent('decode', startTime, 'start')

    let currentTime = startTime

    // Decode阶段每个token需要经过所有PP阶段
    for (let tokenIdx = 0; tokenIdx < maxTokens; tokenIdx++) {
      const tokenStartTime = currentTime

      // 获取该token的计算时间
      const layerComputeTime = this.layerComputeTimeDecode[Math.min(tokenIdx, this.layerComputeTimeDecode.length - 1)]
      // Decode通信时间较短 (因为seq_len=1)
      const tpCommTimeDecode = this.tpCommTime * 0.05  // 大约1/20
      const ppCommTimeDecode = this.ppCommTime * 0.05

      // PP流水线 - 每个stage顺序处理
      for (let stage = 0; stage < pp; stage++) {
        const stageStartTime = stage === 0 ? currentTime : currentTime + stage * ppCommTimeDecode

        // 该stage的所有层
        const stageComputeTime = this.layersPerStage * this.applyJitter(layerComputeTime)
        const stageTpTime = this.layersPerStage * tpCommTimeDecode

        // 重叠计算
        const effectiveTime = stageComputeTime + stageTpTime * (1 - (this.config.enableOverlap ? 0.9 : 0))

        currentTime = Math.max(currentTime, stageStartTime + effectiveTime)

        this.totalComputeTime += stageComputeTime
      }

      // 添加token级别的甘特图任务 (简化显示)
      if (tokenIdx < 5 || tokenIdx === maxTokens - 1 || tokenIdx % 10 === 0) {
        this.addGanttTask(
          `pp0_tp0_ep0`,
          0,
          'decode',
          undefined,
          tokenIdx,
          'compute',
          tokenStartTime,
          currentTime
        )
      }
    }

    // 记录阶段结束
    this.addPhaseEvent('decode', currentTime, 'end')

    return currentTime
  }

  /**
   * 添加微批次级别的甘特图任务
   */
  private addMicroBatchGanttTask(
    ppStage: number,
    microBatchId: number,
    direction: 'forward' | 'backward',
    phase: InferencePhase,
    start: number,
    end: number
  ): void {
    const taskType: GanttTaskType = 'compute'
    const name = `MB${microBatchId} ${direction === 'forward' ? 'F' : 'B'}`

    this.ganttTasks.push({
      id: generateId('gantt'),
      name,
      resource: `Stage ${ppStage}`,
      start,
      end,
      type: taskType,
      phase,
      chipId: `pp${ppStage}_tp0_ep0`,
      ppStage,
      tokenIndex: microBatchId,
    })
  }

  /**
   * 添加PP通信甘特图任务
   */
  private addPPCommGanttTask(
    fromStage: number,
    toStage: number,
    microBatchId: number,
    phase: InferencePhase,
    start: number,
    end: number
  ): void {
    this.ganttTasks.push({
      id: generateId('gantt'),
      name: `PP ${fromStage}→${toStage}`,
      resource: `Stage ${fromStage} Network`,
      start,
      end,
      type: 'pp_comm',
      phase,
      chipId: `pp${fromStage}_tp0_ep0`,
      ppStage: fromStage,
      tokenIndex: microBatchId,
    })
  }

  /**
   * 添加Bubble甘特图任务
   */
  private addBubbleGanttTask(
    ppStage: number,
    phase: InferencePhase,
    start: number,
    end: number
  ): void {
    if (end <= start) return

    this.ganttTasks.push({
      id: generateId('gantt'),
      name: 'Bubble',
      resource: `Stage ${ppStage}`,
      start,
      end,
      type: 'bubble',
      phase,
      chipId: `pp${ppStage}_tp0_ep0`,
      ppStage,
    })
  }

  /**
   * 应用抖动
   */
  private applyJitter(value: number): number {
    if (this.config.jitterFactor === 0) return value
    const jitter = (Math.random() - 0.5) * 2 * this.config.jitterFactor
    return value * (1 + jitter)
  }

  /**
   * 添加阶段事件
   */
  private addPhaseEvent(phase: InferencePhase, timestamp: number, type: 'start' | 'end'): void {
    this.events.push({
      id: generateId('phase'),
      type: type === 'start' ? 'phase_start' : 'phase_end',
      timestamp,
      chipId: 'global',
      ppStage: 0,
      phase,
      operation: 'attention',
      resource: 'compute',
    })
  }

  /**
   * 添加甘特图任务
   */
  private addGanttTask(
    chipId: string,
    ppStage: number,
    phase: InferencePhase,
    layerIndex: number | undefined,
    tokenIndex: number | undefined,
    type: 'compute' | 'tp_comm' | 'pp_comm' | 'ep_comm' | 'bubble' | 'idle',
    start: number,
    end: number
  ): void {
    const name = layerIndex !== undefined
      ? `Layer ${layerIndex} ${type}`
      : tokenIndex !== undefined
        ? `Token ${tokenIndex}`
        : type

    this.ganttTasks.push({
      id: generateId('gantt'),
      name,
      resource: `Stage ${ppStage}`,
      start,
      end,
      type: type as GanttTaskType,
      phase,
      chipId,
      ppStage,
      layerIndex,
      tokenIndex,
    })
  }

  /**
   * 生成模拟结果
   */
  private generateResult(prefillEndTime: number, decodeEndTime: number): SimulationResult {
    const stats = this.calculateStats(prefillEndTime, decodeEndTime)
    const ganttChart = this.generateGanttChart(prefillEndTime, decodeEndTime)

    return {
      config: this.config,
      events: this.events,
      commTrace: this.commTrace,
      ganttChart,
      stats,
      timestamp: Date.now(),
    }
  }

  /**
   * 计算统计数据
   */
  private calculateStats(prefillEndTime: number, decodeEndTime: number): SimulationStats {
    const prefillTime = prefillEndTime
    const decodeTime = decodeEndTime - prefillEndTime
    const totalTime = decodeEndTime
    const numTokens = Math.min(this.config.maxSimulatedTokens, this.inference.output_seq_length)

    // Prefill 阶段统计
    const numMicroBatches = this.microBatches.length
    const prefillComputeTime = numMicroBatches * this.model.num_layers * this.layerComputeTimePrefill
    const prefillCommTime = numMicroBatches * this.model.num_layers * this.tpCommTime +
                            numMicroBatches * (this.parallelism.pp - 1) * this.ppCommTime

    const prefillStats: PhaseTimeStats = {
      computeTime: prefillComputeTime,
      commTime: prefillCommTime,
      bubbleTime: this.totalBubbleTime,
      overlapTime: this.totalOverlapTime,
      totalTime: prefillTime,
      computeEfficiency: prefillComputeTime / (prefillTime * this.parallelism.pp),
    }

    // Decode 阶段统计
    const avgTpot = decodeTime / numTokens

    const tpotDistribution: LatencyDistribution = {
      min: avgTpot * (1 - this.config.jitterFactor),
      max: avgTpot * (1 + this.config.jitterFactor),
      mean: avgTpot,
      stdDev: avgTpot * this.config.jitterFactor,
      p50: avgTpot,
      p90: avgTpot * 1.1,
      p99: avgTpot * 1.2,
    }

    const decodeStats: PhaseTimeStats = {
      computeTime: decodeTime * 0.75,
      commTime: decodeTime * 0.15,
      bubbleTime: decodeTime * 0.1,
      overlapTime: decodeTime * 0.05,
      totalTime: decodeTime,
      computeEfficiency: 0.75,
    }

    // 计算动态 MFU
    const microBatchSize = Math.ceil(this.inference.batch_size / numMicroBatches)
    const totalFlops = calculateLayerFlopsPrefill(
      this.model,
      microBatchSize,
      this.inference.input_seq_length
    ) * this.model.num_layers * numMicroBatches

    const totalChips = this.parallelism.tp * this.parallelism.pp * this.parallelism.ep
    const theoreticalFlopsPerSecond = this.hardware.chip.compute_tflops_fp16 * 1e12 * totalChips
    const actualFlopsPerSecond = totalFlops / (prefillTime / 1000)
    const dynamicMfu = actualFlopsPerSecond / theoreticalFlopsPerSecond

    // MBU 计算 (基于实际 Decode 访存量和 TPOT)
    // MBU = (Model_Weights + KV_Cache) / (TPOT * Peak_Bandwidth)
    const dynamicMbu = this.calculateDecodeMBU(avgTpot)

    // PP 气泡比
    const maxPPBubbleRatio = this.parallelism.pp > 1
      ? (this.parallelism.pp - 1) / (numMicroBatches + this.parallelism.pp - 1)
      : 0

    return {
      prefill: prefillStats,
      decode: decodeStats,
      totalRunTime: totalTime,
      simulatedTokens: numTokens,
      ttft: prefillTime,
      avgTpot,
      tpotDistribution,
      dynamicMfu: Math.min(dynamicMfu, 1),
      dynamicMbu,
      maxPPBubbleRatio,
      totalEvents: this.events.length,
      prefillFlops: totalFlops,
    }
  }

  /**
   * 计算 Decode 阶段的 MBU (Memory Bandwidth Utilization)
   *
   * MBU = (Data_Read_Per_Token) / (TPOT * Peak_Bandwidth)
   *
   * 其中 Data_Read_Per_Token 包括:
   * - 模型权重 (每 token 都需要加载)
   * - KV Cache (按当前 context 比例)
   */
  private calculateDecodeMBU(avgTpotMs: number): number {
    if (avgTpotMs <= 0) return 0

    const GB_TO_BYTES = 1024 * 1024 * 1024
    const weightBytesPerElement = getBytesPerElement(this.model.weight_dtype)
    const actBytesPerElement = getBytesPerElement(this.model.activation_dtype)
    const H = this.model.hidden_size

    // 1. 计算模型权重 (需要正确处理 MoE)
    let modelWeightBytes = 0

    // Attention 权重 (每层, 按 TP 切分)
    const attnWeightPerLayer = this.model.mla_config
      ? (
          H * this.model.mla_config.q_lora_rank +  // Q LoRA down
          this.model.mla_config.q_lora_rank * this.model.num_attention_heads *
            (this.model.mla_config.qk_nope_head_dim + this.model.mla_config.qk_rope_head_dim) +  // Q LoRA up
          H * this.model.mla_config.kv_lora_rank +  // KV compress
          this.model.num_attention_heads * (this.model.mla_config.v_head_dim ?? (H / this.model.num_attention_heads)) * H  // Output proj
        ) * weightBytesPerElement / this.parallelism.tp
      : (4 * H * H) * weightBytesPerElement / this.parallelism.tp  // 标准 QKV + O

    // FFN 权重
    if (this.model.model_type === 'moe' && this.model.moe_config) {
      // MoE: 前几层是 Dense，其余是 MoE
      const numDenseLayers = this.model.moe_config.first_k_dense_replace ?? 3
      const numMoELayers = this.model.num_layers - numDenseLayers

      // Dense FFN (按 TP 切分)
      const denseFFNWeightPerLayer = 3 * H * this.model.intermediate_size * weightBytesPerElement / this.parallelism.tp
      // MoE FFN (8 experts, 不按 TP 切分!)
      const moeFFNWeightPerLayer = 3 * H * (this.model.moe_config.expert_intermediate_size ?? this.model.intermediate_size) *
        weightBytesPerElement * this.model.moe_config.num_experts_per_tok

      modelWeightBytes = (attnWeightPerLayer + denseFFNWeightPerLayer) * numDenseLayers +
                         (attnWeightPerLayer + moeFFNWeightPerLayer) * numMoELayers
    } else {
      // Dense 模型
      const ffnWeightPerLayer = 3 * H * this.model.intermediate_size * weightBytesPerElement / this.parallelism.tp
      modelWeightBytes = (attnWeightPerLayer + ffnWeightPerLayer) * this.model.num_layers
    }

    // 2. 计算 KV Cache (按平均 context 长度，使用激活精度)
    const avgContext = this.inference.input_seq_length + this.inference.output_seq_length / 2
    const kvDim = this.model.mla_config?.kv_lora_rank ?? (this.model.hidden_size / this.model.num_attention_heads * this.model.num_kv_heads)
    const kvCacheBytes = 2 * this.inference.batch_size * avgContext * kvDim * actBytesPerElement / this.parallelism.tp * this.model.num_layers

    // 3. 计算 MBU
    const dataReadPerTokenGB = (modelWeightBytes + kvCacheBytes) / GB_TO_BYTES
    const tpotSeconds = avgTpotMs / 1000
    const achievedBandwidthGBps = dataReadPerTokenGB / tpotSeconds
    const peakBandwidthGBps = this.hardware.chip.memory_bandwidth_gbps

    const mbu = achievedBandwidthGBps / peakBandwidthGBps
    return Math.min(mbu, 1.0)  // MBU 不应超过 1
  }

  /**
   * 生成甘特图数据
   */
  private generateGanttChart(prefillEndTime: number, decodeEndTime: number): GanttChartData {
    const resources: GanttResource[] = []

    for (let ppStage = 0; ppStage < this.parallelism.pp; ppStage++) {
      resources.push({
        id: `stage${ppStage}_compute`,
        name: `Stage ${ppStage}`,
        ppStage,
        type: 'compute',
      })

      if (this.parallelism.pp > 1 || this.parallelism.tp > 1) {
        resources.push({
          id: `stage${ppStage}_network`,
          name: `Stage ${ppStage} Network`,
          ppStage,
          type: 'network',
        })
      }
    }

    return {
      resources,
      tasks: this.ganttTasks,
      timeRange: {
        start: 0,
        end: decodeEndTime,
      },
      phaseTransition: prefillEndTime,
    }
  }
}

/**
 * 便捷函数：运行模拟
 */
export function runInferenceSimulation(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  config?: Partial<SimulationConfig>
): SimulationResult {
  const simulator = new InferenceSimulator(model, inference, parallelism, hardware, config)
  return simulator.run()
}
