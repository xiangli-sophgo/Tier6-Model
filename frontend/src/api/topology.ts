/**
 * 拓扑 API 模块
 *
 * 拓扑生成使用本地生成器，配置存储使用后端 API
 */

import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  ConnectionConfig,
  GlobalSwitchConfig,
  ManualConnectionConfig,
} from '../types';

import {
  topologyGenerator,
  TopologyGenerateRequest,
} from '../utils/topologyGenerator';

import {
  getChipTypes as storageGetChipTypes,
  getRackDimensions as storageGetRackDimensions,
  getLevelConnectionDefaults as storageGetLevelConnectionDefaults,
} from '../utils/storage';

// ============================================
// 拓扑生成 API（本地生成）
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
  interconnect_config?: {
    c2c?: { bandwidth_gbps: number; latency_us: number };
    b2b?: { bandwidth_gbps: number; latency_us: number };
    r2r?: { bandwidth_gbps: number; latency_us: number };
    p2p?: { bandwidth_gbps: number; latency_us: number };
  };
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
// 静态配置接口
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

