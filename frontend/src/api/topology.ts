/**
 * 拓扑 API 模块
 *
 * 使用本地生成器和IndexedDB存储，无需后端服务
 */

import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  ConnectionConfig,
  GlobalSwitchConfig,
  ManualConnectionConfig,
  ManualConnection,
} from '../types';

import {
  topologyGenerator,
  TopologyGenerateRequest,
} from '../utils/topologyGenerator';

import {
  listConfigs as storageListConfigs,
  getConfig as storageGetConfig,
  saveConfig as storageSaveConfig,
  deleteConfig as storageDeleteConfig,
  getManualConnections as storageGetManualConnections,
  saveManualConnections as storageSaveManualConnections,
  addManualConnection as storageAddManualConnection,
  deleteManualConnection as storageDeleteManualConnection,
  clearManualConnections as storageClearManualConnections,
  getChipTypes as storageGetChipTypes,
  getRackDimensions as storageGetRackDimensions,
  getLevelConnectionDefaults as storageGetLevelConnectionDefaults,
  SavedConfig,
} from '../utils/storage';

// ============================================
// 拓扑生成 API
// ============================================

/**
 * 获取完整拓扑数据
 */
export async function getTopology(): Promise<HierarchicalTopology> {
  return topologyGenerator.getCachedTopology();
}

/**
 * 生成新的拓扑
 */
export async function generateTopology(config: {
  pod_count?: number;
  racks_per_pod?: number;
  board_configs?: {
    u1: { count: number; chips: { npu: number; cpu: number } };
    u2: { count: number; chips: { npu: number; cpu: number } };
    u4: { count: number; chips: { npu: number; cpu: number } };
  };
  rack_config?: {
    total_u: number;
    boards: Array<{
      id: string;
      name: string;
      u_height: number;
      count: number;
      chips: Array<{ name: string; count: number }>;
    }>;
  };
  switch_config?: GlobalSwitchConfig;
  manual_connections?: ManualConnectionConfig;
}): Promise<HierarchicalTopology> {
  return topologyGenerator.generate(config as TopologyGenerateRequest);
}

/**
 * 获取指定Pod
 */
export async function getPod(podId: string): Promise<PodConfig> {
  const pod = topologyGenerator.getPod(podId);
  if (!pod) {
    throw new Error(`Pod '${podId}' not found`);
  }
  return pod;
}

/**
 * 获取指定Rack
 */
export async function getRack(rackId: string): Promise<RackConfig> {
  const rack = topologyGenerator.getRack(rackId);
  if (!rack) {
    throw new Error(`Rack '${rackId}' not found`);
  }
  return rack;
}

/**
 * 获取指定Board
 */
export async function getBoard(boardId: string): Promise<BoardConfig> {
  const board = topologyGenerator.getBoard(boardId);
  if (!board) {
    throw new Error(`Board '${boardId}' not found`);
  }
  return board;
}

/**
 * 获取连接数据
 */
export async function getConnections(
  level?: string,
  parentId?: string
): Promise<ConnectionConfig[]> {
  if (level) {
    return topologyGenerator.getConnectionsForLevel(level, parentId);
  }
  return topologyGenerator.getCachedTopology().connections;
}

// ============================================
// 配置接口
// ============================================

/**
 * 获取Chip类型配置
 */
export async function getChipTypes(): Promise<{
  types: { id: string; name: string; color: string }[];
}> {
  return storageGetChipTypes();
}

/**
 * 获取Rack尺寸配置
 */
export async function getRackDimensions(): Promise<{
  width: number;
  depth: number;
  u_height: number;
  total_u: number;
  full_height: number;
}> {
  return storageGetRackDimensions();
}

/**
 * 获取各层级连接的默认带宽和延迟配置
 */
export async function getLevelConnectionDefaults(): Promise<{
  datacenter: { bandwidth: number; latency: number };
  pod: { bandwidth: number; latency: number };
  rack: { bandwidth: number; latency: number };
  board: { bandwidth: number; latency: number };
}> {
  return storageGetLevelConnectionDefaults();
}

// ============================================
// 配置保存/加载 API
// ============================================

export type { SavedConfig } from '../utils/storage';

/**
 * 获取所有保存的配置
 */
export async function listConfigs(): Promise<SavedConfig[]> {
  return storageListConfigs();
}

/**
 * 获取指定配置
 */
export async function getConfig(name: string): Promise<SavedConfig> {
  const config = await storageGetConfig(name);
  if (!config) {
    throw new Error(`配置 '${name}' 不存在`);
  }
  return config;
}

/**
 * 保存配置
 */
export async function saveConfig(config: SavedConfig): Promise<SavedConfig> {
  return storageSaveConfig(config);
}

/**
 * 删除配置
 */
export async function deleteConfig(name: string): Promise<void> {
  return storageDeleteConfig(name);
}

// ============================================
// 手动连接 API
// ============================================

/**
 * 获取手动连接配置
 */
export async function getManualConnections(): Promise<ManualConnectionConfig> {
  return storageGetManualConnections();
}

/**
 * 保存手动连接配置
 */
export async function saveManualConnections(config: ManualConnectionConfig): Promise<ManualConnectionConfig> {
  return storageSaveManualConnections(config);
}

/**
 * 添加单个手动连接
 */
export async function addManualConnection(connection: ManualConnection): Promise<ManualConnectionConfig> {
  return storageAddManualConnection(connection);
}

/**
 * 删除单个手动连接
 */
export async function deleteManualConnection(connectionId: string): Promise<void> {
  return storageDeleteManualConnection(connectionId);
}

/**
 * 清空手动连接（可按层级清空）
 */
export async function clearManualConnections(hierarchyLevel?: string): Promise<void> {
  return storageClearManualConnections(hierarchyLevel);
}
