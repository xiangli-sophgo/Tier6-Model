/**
 * LLM 部署分析系统 - 导出入口
 *
 * 提供 LLM 推理部署方案分析能力：
 * - 预设配置：模型、硬件预设
 * - 后端 API：调用后端精确仿真器进行延迟、吞吐、MFU/MBU 计算
 * - 芯片映射：自动映射芯片到并行组
 * - 流量映射：分析拓扑流量
 *
 * 注意：所有评估计算（参数量、显存、通信量、延迟、吞吐等）均由后端完成
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
// 后端 API 调用层
// ============================================
export {
  simulateBackend,
  batchSimulate,
  validateConfig,
} from './backendApi';

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
  InfeasibleResult,
} from './types';
