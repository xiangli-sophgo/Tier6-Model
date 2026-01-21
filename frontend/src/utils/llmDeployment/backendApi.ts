/**
 * 后端 API 调用层
 *
 * 所有延迟、吞吐、MFU/MBU 计算均由后端完成
 */

import {
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
  HardwareConfig,
  SimulationResult,
  SimulationStats,
} from './types';

// API 基础地址
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

/**
 * 调用后端模拟 API
 */
export async function simulateBackend(
  topology: any,
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  config?: any
): Promise<SimulationResult> {
  const response = await fetch(`${API_BASE_URL}/api/simulate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      topology,
      model,
      inference,
      parallelism,
      hardware,
      config,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: '未知错误' }));
    throw new Error(`后端模拟失败: ${error.detail || response.statusText}`);
  }

  return await response.json();
}

/**
 * 单个模拟配置
 */
export interface SimulationConfig {
  topology: any;
  model: LLMModelConfig;
  inference: InferenceConfig;
  parallelism: ParallelismStrategy;
  hardware: HardwareConfig;
  config?: any;
}

/**
 * 批量模拟选项
 */
export interface BatchSimulateOptions {
  /** 最大并发数（默认 5） */
  concurrency?: number;
  /** 进度回调 (已完成数, 总数, 最新结果, 结果索引) */
  onProgress?: (completed: number, total: number, result: SimulationResult, index: number) => void;
  /** 取消信号 */
  abortSignal?: AbortSignal;
}

/**
 * 批量模拟（并行调用，支持取消）
 *
 * @param configs 多个配置
 * @param options 选项（并发数、进度回调、取消信号）
 */
export async function batchSimulate(
  configs: SimulationConfig[],
  onProgress?: (current: number, total: number, result: SimulationResult) => void,
  options?: BatchSimulateOptions
): Promise<SimulationResult[]> {
  const { concurrency = 5, abortSignal } = options || {};
  const total = configs.length;
  const results: SimulationResult[] = new Array(total);
  let completed = 0;
  let nextIndex = 0;

  // 检查是否已取消
  const checkAborted = () => {
    if (abortSignal?.aborted) {
      throw new DOMException('评估已取消', 'AbortError');
    }
  };

  // 执行单个模拟
  const runOne = async (index: number): Promise<void> => {
    checkAborted();
    const cfg = configs[index];

    try {
      const response = await fetch(`${API_BASE_URL}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topology: cfg.topology,
          model: cfg.model,
          inference: cfg.inference,
          parallelism: cfg.parallelism,
          hardware: cfg.hardware,
          config: cfg.config,
        }),
        signal: abortSignal,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: '未知错误' }));
        throw new Error(`后端模拟失败: ${error.detail || response.statusText}`);
      }

      results[index] = await response.json();
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw error; // 重新抛出取消错误
      }
      console.error(`方案 ${index + 1} 模拟失败:`, error);
      results[index] = {
        ganttChart: { resources: [], tasks: [], timeRange: { start: 0, end: 0 } },
        stats: createErrorStats(error instanceof Error ? error.message : '未知错误'),
        timestamp: Date.now() / 1000,
      };
    }

    completed++;
    // 调用两种回调格式
    onProgress?.(completed, total, results[index]);
    options?.onProgress?.(completed, total, results[index], index);
  };

  // 工作线程
  const worker = async (): Promise<void> => {
    while (true) {
      checkAborted();
      const index = nextIndex++;
      if (index >= total) break;
      await runOne(index);
    }
  };

  // 启动并发工作线程
  const workers = Array(Math.min(concurrency, total))
    .fill(null)
    .map(() => worker());

  await Promise.all(workers);
  return results;
}

/**
 * 创建错误统计信息
 */
function createErrorStats(errorMessage: string): SimulationStats {
  const emptyPhaseStats = {
    computeTime: 0,
    commTime: 0,
    bubbleTime: 0,
    overlapTime: 0,
    totalTime: 0,
    computeEfficiency: 0,
  };

  return {
    prefill: emptyPhaseStats,
    decode: emptyPhaseStats,
    totalRunTime: 0,
    simulatedTokens: 0,
    ttft: Infinity,
    avgTpot: Infinity,
    dynamicMfu: 0,
    dynamicMbu: 0,
    maxPpBubbleRatio: 0,
    totalEvents: 0,
    prefillFlops: 0,
    errorReason: errorMessage,
  };
}

/**
 * 验证配置
 */
export async function validateConfig(
  topology: any,
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): Promise<{
  valid: boolean;
  errors?: string[];
  required_chips?: number;
  available_chips?: number;
}> {
  const response = await fetch(`${API_BASE_URL}/api/validate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      topology,
      model,
      inference,
      parallelism,
      hardware,
    }),
  });

  if (!response.ok) {
    throw new Error(`配置验证失败: ${response.statusText}`);
  }

  return await response.json();
}
