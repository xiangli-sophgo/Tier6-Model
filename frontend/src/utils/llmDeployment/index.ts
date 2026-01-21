/**
 * LLM 部署分析系统 - 导出入口
 *
 * 提供完整的 LLM 推理部署方案分析能力：
 * - 模型参数化：根据 LLM 配置自动计算通信量、显存需求、计算量
 * - 硬件参数化：配置芯片算力、显存、带宽
 * - 显存分析：分析各项显存占用，检查是否超限
 * - 后端模拟：调用后端精确仿真器进行延迟、吞吐、MFU/MBU 计算
 * - 方案搜索：自动搜索满足约束的所有方案（串行调用后端）
 * - 评分排序：基于后端仿真结果进行方案评分和排序
 *
 * 注意：所有延迟估算、瓶颈分析、吞吐量计算均由后端完成
 */

// ============================================
// 类型定义
// ============================================
export * from './types';

// ============================================
// 预设配置
// ============================================
export * from './presets';

// ============================================
// 模型计算器
// ============================================
export {
  // 参数量计算
  calculateModelParams,
  calculateParamsPerLayer,
  // 显存计算
  calculateModelMemory,
  calculateKVCacheMemory,
  calculateActivationMemory,
  calculateOverheadMemory,
  analyzeMemory,
  // FLOPs 计算
  calculateLayerFlopsPrefill,
  calculateLayerFlopsDecode,
  calculatePrefillFlops,
  calculateDecodeFlopsPerToken,
  calculateTotalInferenceFlops,
  calculateFlopsPerToken,
} from './modelCalculator';

// ============================================
// 通信计算器
// ============================================
export {
  // TP 通信量
  calculateTPCommVolumePrefill,
  calculateTPCommVolumeDecode,
  // PP 通信量
  calculatePPCommVolumePrefill,
  calculatePPCommVolumeDecode,
  // EP 通信量
  calculateEPCommVolumePrefill,
  calculateEPCommVolumeDecode,
  // SP 通信量
  calculateSPCommVolumePrefill,
  calculateSPCommVolumeDecode,
  // DP 通信量 (训练)
  calculateDPGradientSyncVolume,
  // 综合分析
  analyzeCommunication,
} from './commCalculator';

// ============================================
// 后端 API 调用层
// ============================================
export {
  simulateBackend,
  batchSimulate,
  validateConfig,
} from './backendApi';

// ============================================
// 方案搜索器（后端版本）
// ============================================
export {
  searchWithFixedChips,
  progressiveSearch,
  type SearchOptions,
  type SearchResult,
  type InfeasibleResult,
  type FullSearchResult,
} from './planSearcherBackend';

// ============================================
// 仿真评分计算器
// ============================================
export {
  calculateSimulationScore,
  formatDeviation,
  isSignificantDeviation,
} from './simulationScorer';

// ============================================
// 结果适配器
// ============================================
export {
  adaptSimulationResult,
} from './resultAdapter';

// ============================================
// 芯片映射器 (拓扑融合)
// ============================================
export {
  // 自动映射
  autoMapChipsToParallelism,
  // 通信组生成
  generateCommunicationGroups,
  // 辅助函数
  getCollectiveOpDescription,
  getParallelismTypeDescription,
} from './chipMapper';

// ============================================
// 流量映射器 (拓扑融合)
// ============================================
export {
  // 流量映射
  mapTrafficToLinks,
  // 完整分析
  analyzeTopologyTraffic,
  // 热力图辅助
  getHeatmapColor,
  getHeatmapWidth,
} from './trafficMapper';

// ============================================
// 推理模拟相关类型
// ============================================
// 注意: 前端模拟器已移除，所有模拟均由后端执行
export type {
  SimulationConfig,
  SimulationResult,
  GanttChartData,
  GanttTask,
  SimulationStats,
  SimulationScoreResult,
  FormulaVsSimComparison,
} from './types';
