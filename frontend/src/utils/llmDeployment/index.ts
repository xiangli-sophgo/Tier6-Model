/**
 * LLM 部署分析系统 - 导出入口
 *
 * 提供完整的 LLM 推理部署方案分析能力：
 * - 模型参数化：根据 LLM 配置自动计算通信量、显存需求、计算量
 * - 硬件参数化：配置芯片算力、显存、带宽，计算真实延迟
 * - 延迟估算：预估 Prefill/Decode 延迟，识别瓶颈
 * - 显存分析：分析各项显存占用，检查是否超限
 * - 方案搜索：自动搜索满足约束的所有方案，评分排序
 * - 多方案对比：对比多个方案的各项指标
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
// 延迟估算器
// ============================================
export {
  // 计算延迟
  estimatePrefillComputeLatency,
  estimateDecodeComputeLatency,
  // 访存延迟
  estimateMemoryLatency,
  estimateDecodeMemoryLatency,
  // 通信延迟
  estimateCommLatency,
  estimatePrefillCommLatency,
  estimateDecodeCommLatency,
  // 流水线
  calculatePPBubbleRatio,
  calculatePPEfficiency,
  // 综合分析
  analyzeLatency,
  analyzeBottleneckRoofline,
  // 吞吐量
  estimateTokenThroughput,
  estimateRequestThroughput,
  estimateMFU,
  estimateTheoreticalMaxThroughput,
} from './latencyEstimator';

// ============================================
// 方案分析器
// ============================================
export {
  // 可行性检查
  checkFeasibility,
  // 吞吐分析
  analyzeThroughput,
  // 利用率分析
  analyzeUtilization,
  // 评分
  calculateOverallScore,
  // 建议生成
  generateSuggestions,
  // 完整分析
  analyzePlan,
  quickAnalyze,
} from './planAnalyzer';

// ============================================
// 方案搜索器
// ============================================
export {
  // 完整搜索
  searchOptimalPlan,
  // 快速搜索
  quickSearch,
  // 固定芯片数搜索
  searchWithFixedChips,
  // 渐进式搜索
  progressiveSearch,
} from './planSearcher';

// ============================================
// 方案对比器
// ============================================
export {
  // 对比分析
  comparePlans,
  analyzeAndCompare,
  // 报告生成
  generateComparisonReport,
  // 成本效益
  calculateCostPerToken,
  compareCostEfficiency,
  // 场景适配
  analyzeScenarioFit,
  // 敏感性分析
  analyzeSensitivity,
  // 数据导出
  exportComparisonToCSV,
  exportComparisonToJSON,
} from './planComparator';

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
// 推理模拟器
// ============================================
export {
  // 核心模拟器
  InferenceSimulator,
  runInferenceSimulation,
  // 事件队列
  EventQueue,
  TaskDependencyGraph,
  // 类型 (常用)
  type SimulationConfig,
  type SimulationResult,
  type GanttChartData,
  type GanttTask,
  type SimEvent,
  type CommTraceItem,
  type SimulationStats,
  // 默认配置
  DEFAULT_SIMULATION_CONFIG,
} from './simulation';
