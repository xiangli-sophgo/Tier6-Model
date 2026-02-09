/**
 * Math Model 后端 API 模块
 *
 * 提供与 math_model 仿真引擎的接口
 */

import type {
  ChipPreset,
  ModelPreset,
  BenchmarkListItem,
  BenchmarkConfig,
  TopologyListItem,
  TopologyConfig,
  SimulateRequest,
  SimulateResponse,
  ValidationResult,
  EvaluationRequest,
  TaskStatus,
  TaskResults,
  Experiment,
} from '../types/math_model';

// API 统一使用 /api 前缀（通过 vite proxy 转发）
const API_BASE = '';

// ==================== 芯片预设接口 ====================

/**
 * 获取芯片预设列表（含完整配置）
 */
export async function getChipPresets(): Promise<{ presets: Array<{ name: string; config: ChipPreset }> }> {
  const res = await fetch(`${API_BASE}/api/presets/chips`);
  if (!res.ok) throw new Error(`Failed to fetch chip presets: ${res.statusText}`);
  return res.json();
}

/**
 * 获取芯片预设详情
 */
export async function getChipPreset(name: string): Promise<ChipPreset> {
  const res = await fetch(`${API_BASE}/api/presets/chips/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`Failed to fetch chip preset: ${res.statusText}`);
  return res.json();
}

/**
 * 创建芯片预设（另存为）
 */
export async function saveChipPreset(config: ChipPreset): Promise<{ message: string }> {
  const res = await fetch(`${API_BASE}/api/presets/chips`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`Failed to save chip preset: ${res.statusText}`);
  return res.json();
}

/**
 * 更新芯片预设（保存）
 */
export async function updateChipPreset(name: string, config: ChipPreset): Promise<{ message: string }> {
  const res = await fetch(`${API_BASE}/api/presets/chips/${encodeURIComponent(name)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config }),
  });
  if (!res.ok) throw new Error(`Failed to update chip preset: ${res.statusText}`);
  return res.json();
}

/**
 * 删除芯片预设
 */
export async function deleteChipPreset(name: string): Promise<{ message: string }> {
  const res = await fetch(`${API_BASE}/api/presets/chips/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Failed to delete chip preset: ${res.statusText}`);
  return res.json();
}

// ==================== 模型预设接口 ====================

/**
 * 获取模型预设列表（含完整配置）
 */
export async function getModelPresets(): Promise<{ presets: Array<{ name: string; config: ModelPreset }> }> {
  const res = await fetch(`${API_BASE}/api/presets/models`);
  if (!res.ok) throw new Error(`Failed to fetch model presets: ${res.statusText}`);
  return res.json();
}

/**
 * 获取模型预设详情
 */
export async function getModelPreset(name: string): Promise<ModelPreset> {
  const res = await fetch(`${API_BASE}/api/presets/models/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`Failed to fetch model preset: ${res.statusText}`);
  return res.json();
}

/**
 * 创建模型预设（另存为）
 */
export async function saveModelPreset(name: string, config: ModelPreset): Promise<{ message: string }> {
  const res = await fetch(`${API_BASE}/api/presets/models`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, config }),
  });
  if (!res.ok) throw new Error(`Failed to save model preset: ${res.statusText}`);
  return res.json();
}

/**
 * 更新模型预设（保存）
 */
export async function updateModelPreset(name: string, config: ModelPreset): Promise<{ message: string }> {
  const res = await fetch(`${API_BASE}/api/presets/models/${encodeURIComponent(name)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config }),
  });
  if (!res.ok) throw new Error(`Failed to update model preset: ${res.statusText}`);
  return res.json();
}

/**
 * 删除模型预设
 */
export async function deleteModelPreset(name: string): Promise<{ message: string }> {
  const res = await fetch(`${API_BASE}/api/presets/models/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Failed to delete model preset: ${res.statusText}`);
  return res.json();
}

// ==================== Benchmark 接口 ====================

/**
 * 获取 Benchmark 列表
 */
export async function getBenchmarks(): Promise<{ benchmarks: BenchmarkListItem[] }> {
  const res = await fetch(`${API_BASE}/api/benchmarks`);
  if (!res.ok) throw new Error(`Failed to fetch benchmarks: ${res.statusText}`);
  return res.json();
}

/**
 * 获取 Benchmark 详情
 */
export async function getBenchmark(id: string): Promise<BenchmarkConfig> {
  const res = await fetch(`${API_BASE}/api/benchmarks/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`Failed to fetch benchmark: ${res.statusText}`);
  return res.json();
}

/**
 * 创建 Benchmark
 */
export async function createBenchmark(config: BenchmarkConfig): Promise<{ id: string }> {
  const res = await fetch(`${API_BASE}/api/benchmarks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`Failed to create benchmark: ${res.statusText}`);
  return res.json();
}

/**
 * 更新 Benchmark
 */
export async function updateBenchmark(id: string, config: BenchmarkConfig): Promise<void> {
  const res = await fetch(`${API_BASE}/api/benchmarks/${encodeURIComponent(id)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`Failed to update benchmark: ${res.statusText}`);
}

/**
 * 删除 Benchmark
 */
export async function deleteBenchmark(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/benchmarks/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Failed to delete benchmark: ${res.statusText}`);
}

// ==================== Topology 接口 ====================

/**
 * 获取拓扑列表
 */
export async function getTopologies(): Promise<{ topologies: TopologyListItem[] }> {
  const res = await fetch(`${API_BASE}/api/topologies`);
  if (!res.ok) throw new Error(`Failed to fetch topologies: ${res.statusText}`);
  return res.json();
}

/**
 * 获取拓扑详情（含解析后的芯片参数）
 */
export async function getTopology(name: string): Promise<TopologyConfig> {
  const res = await fetch(`${API_BASE}/api/topologies/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`Failed to fetch topology: ${res.statusText}`);
  return res.json();
}

/**
 * 创建拓扑
 */
export async function createTopology(config: TopologyConfig): Promise<{ name: string }> {
  const res = await fetch(`${API_BASE}/api/topologies`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`Failed to create topology: ${res.statusText}`);
  return res.json();
}

/**
 * 更新拓扑
 */
export async function updateTopology(name: string, config: TopologyConfig): Promise<void> {
  const res = await fetch(`${API_BASE}/api/topologies/${encodeURIComponent(name)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`Failed to update topology: ${res.statusText}`);
}

/**
 * 删除拓扑
 */
export async function deleteTopology(name: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/topologies/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Failed to delete topology: ${res.statusText}`);
}

// ==================== 仿真接口 ====================

/**
 * 同步仿真
 */
export async function simulate(request: SimulateRequest): Promise<SimulateResponse> {
  const res = await fetch(`${API_BASE}/api/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(`Simulation failed: ${res.statusText}`);
  return res.json();
}

/**
 * 配置验证
 */
export async function validateConfig(config: Record<string, unknown>): Promise<ValidationResult> {
  const res = await fetch(`${API_BASE}/api/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`Validation failed: ${res.statusText}`);
  return res.json();
}

// ==================== 评估任务接口 ====================

/**
 * 提交评估任务
 */
export async function submitEvaluation(request: EvaluationRequest): Promise<{ task_id: string; experiment_id: number }> {
  const res = await fetch(`${API_BASE}/api/evaluation/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(`Failed to submit evaluation: ${res.statusText}`);
  return res.json();
}

/**
 * 获取任务状态
 */
export async function getTaskStatus(taskId: string): Promise<TaskStatus> {
  const res = await fetch(`${API_BASE}/api/evaluation/tasks/${encodeURIComponent(taskId)}`);
  if (!res.ok) throw new Error(`Failed to fetch task status: ${res.statusText}`);
  return res.json();
}

/**
 * 获取任务结果
 */
export async function getTaskResults(taskId: string): Promise<TaskResults> {
  const res = await fetch(`${API_BASE}/api/evaluation/tasks/${encodeURIComponent(taskId)}/results`);
  if (!res.ok) throw new Error(`Failed to fetch task results: ${res.statusText}`);
  return res.json();
}

/**
 * 取消任务
 */
export async function cancelTask(taskId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/evaluation/tasks/${encodeURIComponent(taskId)}/cancel`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Failed to cancel task: ${res.statusText}`);
}

// ==================== 实验管理接口 ====================

/**
 * 获取实验列表
 */
export async function getExperiments(params?: { skip?: number; limit?: number }): Promise<{ experiments: Experiment[]; total: number }> {
  const url = new URL(`${API_BASE}/api/evaluation/experiments`);
  if (params?.skip !== undefined) url.searchParams.set('skip', String(params.skip));
  if (params?.limit !== undefined) url.searchParams.set('limit', String(params.limit));
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`Failed to fetch experiments: ${res.statusText}`);
  return res.json();
}

/**
 * 获取实验详情
 */
export async function getExperiment(id: number): Promise<Experiment> {
  const res = await fetch(`${API_BASE}/api/evaluation/experiments/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch experiment: ${res.statusText}`);
  return res.json();
}

/**
 * 更新实验
 */
export async function updateExperiment(id: number, data: { name?: string; description?: string }): Promise<void> {
  const res = await fetch(`${API_BASE}/api/evaluation/experiments/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Failed to update experiment: ${res.statusText}`);
}

/**
 * 删除实验
 */
export async function deleteExperiment(id: number): Promise<void> {
  const res = await fetch(`${API_BASE}/api/evaluation/experiments/${id}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Failed to delete experiment: ${res.statusText}`);
}

/**
 * 批量删除实验
 */
export async function batchDeleteExperiments(ids: number[]): Promise<void> {
  const res = await fetch(`${API_BASE}/api/evaluation/experiments/batch-delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ experiment_ids: ids }),
  });
  if (!res.ok) throw new Error(`Failed to batch delete experiments: ${res.statusText}`);
}

/**
 * 导出实验
 */
export async function exportExperiments(ids: number[]): Promise<Blob> {
  const url = new URL(`${API_BASE}/api/evaluation/experiments/export`);
  url.searchParams.set('experiment_ids', ids.join(','));
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`Failed to export experiments: ${res.statusText}`);
  return res.blob();
}

// ==================== 导出类型 ====================

export type {
  ChipPreset,
  GmemConfig,
  LmemConfig,
  MacPerLaneConfig,
  EuPerLaneConfig,
  ModelPreset,
  MoEConfig,
  MLAConfig,
  DSAConfig,
  NSAConfig,
  RoPEConfig,
  BenchmarkListItem,
  BenchmarkConfig,
  BackendInferenceConfig,
  TopologyListItem,
  TopologyConfig,
  CommLatencyConfig,
  EvaluationRequest,
  ManualParallelism,
  TaskStatus,
  TaskResults,
  EvaluationResult,
  Experiment,
  SimulateRequest,
  SimulateResponse,
  ValidationResult,
} from '../types/math_model';
