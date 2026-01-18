/**
 * 流量映射模块
 *
 * 将通信组的流量分配到物理链路上，使用BFS最短路径算法
 */

import { HierarchicalTopology } from '../../types';
import {
  CommunicationGroup,
  LinkTraffic,
  TopologyTrafficResult,
  ParallelismStrategy,
  CommunicationAnalysis,
} from './types';
import { autoMapChipsToParallelism, generateCommunicationGroups } from './chipMapper';

// ============================================
// 图结构与路径查找
// ============================================

/** 图的邻接表表示 */
interface Graph {
  nodes: Set<string>;
  edges: Map<string, Map<string, { bandwidth: number; latency: number }>>;
}

/**
 * 从拓扑构建图
 */
function buildGraphFromTopology(topology: HierarchicalTopology): Graph {
  const nodes = new Set<string>();
  const edges = new Map<string, Map<string, { bandwidth: number; latency: number }>>();

  // 添加所有连接
  for (const conn of topology.connections) {
    nodes.add(conn.source);
    nodes.add(conn.target);

    // 添加双向边
    if (!edges.has(conn.source)) {
      edges.set(conn.source, new Map());
    }
    if (!edges.has(conn.target)) {
      edges.set(conn.target, new Map());
    }

    const bandwidth = conn.bandwidth ?? 100; // 默认100 Gbps
    const latency = conn.latency ?? 1; // 默认1 us

    edges.get(conn.source)!.set(conn.target, { bandwidth, latency });
    edges.get(conn.target)!.set(conn.source, { bandwidth, latency });
  }

  return { nodes, edges };
}

/**
 * BFS 查找最短路径
 */
function findShortestPath(
  graph: Graph,
  source: string,
  target: string
): { path: string[]; found: boolean } {
  if (source === target) {
    return { path: [source], found: true };
  }

  const visited = new Set<string>();
  const queue: { node: string; path: string[] }[] = [{ node: source, path: [source] }];
  visited.add(source);

  while (queue.length > 0) {
    const { node, path } = queue.shift()!;
    const neighbors = graph.edges.get(node);

    if (!neighbors) continue;

    for (const [neighbor] of neighbors) {
      if (visited.has(neighbor)) continue;

      const newPath = [...path, neighbor];

      if (neighbor === target) {
        return { path: newPath, found: true };
      }

      visited.add(neighbor);
      queue.push({ node: neighbor, path: newPath });
    }
  }

  return { path: [], found: false };
}

// ============================================
// 通信对生成
// ============================================

/** 通信对 */
interface CommPair {
  source: string;
  target: string;
  messageSizeMb: number;
  groupId: string;
}

/**
 * 根据集合操作类型生成通信对
 */
function generateCommPairs(group: CommunicationGroup): CommPair[] {
  const pairs: CommPair[] = [];
  const { members, collectiveOp, messageSizeMb, id } = group;
  const n = members.length;

  if (n < 2) return pairs;

  switch (collectiveOp) {
    case 'allreduce':
      // Ring AllReduce: 每个节点发送到下一个节点
      // 实际通信量: 2 * (n-1)/n * M (reduce-scatter + all-gather)
      // 简化为每对通信 M/n
      for (let i = 0; i < n; i++) {
        const next = (i + 1) % n;
        pairs.push({
          source: members[i],
          target: members[next],
          messageSizeMb: messageSizeMb / n * 2, // Ring的两阶段
          groupId: id,
        });
      }
      break;

    case 'p2p':
      // 流水线: 相邻节点间点对点通信
      for (let i = 0; i < n - 1; i++) {
        pairs.push({
          source: members[i],
          target: members[i + 1],
          messageSizeMb: messageSizeMb,
          groupId: id,
        });
      }
      break;

    case 'alltoall':
      // All-to-All: 每对节点间都有通信
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i !== j) {
            pairs.push({
              source: members[i],
              target: members[j],
              messageSizeMb: messageSizeMb / (n * n),
              groupId: id,
            });
          }
        }
      }
      break;

    case 'allgather':
      // AllGather: Ring模式
      for (let i = 0; i < n; i++) {
        const next = (i + 1) % n;
        pairs.push({
          source: members[i],
          target: members[next],
          messageSizeMb: messageSizeMb / n,
          groupId: id,
        });
      }
      break;

    case 'reduce_scatter':
      // ReduceScatter: Ring模式
      for (let i = 0; i < n; i++) {
        const next = (i + 1) % n;
        pairs.push({
          source: members[i],
          target: members[next],
          messageSizeMb: messageSizeMb / n,
          groupId: id,
        });
      }
      break;
  }

  return pairs;
}

// ============================================
// 流量映射
// ============================================

/**
 * 将通信组流量映射到链路
 */
export function mapTrafficToLinks(
  groups: CommunicationGroup[],
  topology: HierarchicalTopology
): LinkTraffic[] {
  // 构建图
  const graph = buildGraphFromTopology(topology);

  // 创建链路带宽映射
  const linkBandwidth = new Map<string, number>();
  for (const conn of topology.connections) {
    const key1 = `${conn.source}->${conn.target}`;
    const key2 = `${conn.target}->${conn.source}`;
    const bandwidth = conn.bandwidth ?? 100;
    linkBandwidth.set(key1, bandwidth);
    linkBandwidth.set(key2, bandwidth);
  }

  // 累计每条链路的流量
  const linkTrafficMap = new Map<string, { trafficMb: number; groups: Set<string> }>();

  // 处理每个通信组
  for (const group of groups) {
    if (group.messageSizeMb === 0 || group.frequency === 0) continue;

    const pairs = generateCommPairs(group);

    for (const pair of pairs) {
      // 查找最短路径
      const { path, found } = findShortestPath(graph, pair.source, pair.target);

      if (!found || path.length < 2) continue;

      // 将流量分配到路径上的每条边
      for (let i = 0; i < path.length - 1; i++) {
        const edgeKey = `${path[i]}->${path[i + 1]}`;

        if (!linkTrafficMap.has(edgeKey)) {
          linkTrafficMap.set(edgeKey, { trafficMb: 0, groups: new Set() });
        }

        const linkData = linkTrafficMap.get(edgeKey)!;
        linkData.trafficMb += pair.messageSizeMb * group.frequency;
        linkData.groups.add(pair.groupId);
      }
    }
  }

  // 转换为 LinkTraffic 数组
  const result: LinkTraffic[] = [];

  for (const [key, data] of linkTrafficMap) {
    const [source, target] = key.split('->');
    const bandwidth = linkBandwidth.get(key) ?? 100;

    // 计算利用率: 流量(MB) / (带宽(Gbps) * 1000 / 8) = 流量(MB) / (带宽 * 125)
    // 假设1秒内完成所有通信
    const utilizationPercent = (data.trafficMb / (bandwidth * 125)) * 100;

    result.push({
      source,
      target,
      trafficMb: data.trafficMb,
      bandwidthGbps: bandwidth,
      utilizationPercent: Math.min(utilizationPercent, 100), // 上限100%
      contributingGroups: Array.from(data.groups),
    });
  }

  // 按流量大小排序
  result.sort((a, b) => b.trafficMb - a.trafficMb);

  return result;
}

// ============================================
// 主入口函数
// ============================================

/**
 * 执行完整的拓扑流量分析
 */
export function analyzeTopologyTraffic(
  topology: HierarchicalTopology,
  parallelism: ParallelismStrategy,
  commAnalysis: CommunicationAnalysis
): TopologyTrafficResult {
  // Step 1: 芯片映射
  const { mapping } = autoMapChipsToParallelism(topology, parallelism);

  // Step 2: 生成通信组
  const groups = generateCommunicationGroups(mapping, parallelism, commAnalysis);

  // Step 3: 映射流量到链路
  const linkTraffic = mapTrafficToLinks(groups, topology);

  // Step 4: 分析瓶颈
  const bottleneckThreshold = 80; // 80% 利用率
  const bottleneckLinks = linkTraffic
    .filter(lt => lt.utilizationPercent > bottleneckThreshold)
    .map(lt => `${lt.source}->${lt.target}`);

  const utilizations = linkTraffic.map(lt => lt.utilizationPercent);
  const maxUtilization = utilizations.length > 0 ? Math.max(...utilizations) : 0;
  const avgUtilization = utilizations.length > 0
    ? utilizations.reduce((a, b) => a + b, 0) / utilizations.length
    : 0;

  return {
    chipMapping: mapping,
    communicationGroups: groups,
    linkTraffic,
    bottleneckLinks,
    maxUtilization,
    avgUtilization,
  };
}

/**
 * 获取热力图颜色
 * 利用率: 0-30% 绿色, 30-60% 黄色, 60-80% 橙色, 80-100% 红色
 */
export function getHeatmapColor(utilizationPercent: number): string {
  const u = Math.min(Math.max(utilizationPercent, 0), 100);

  if (u < 30) {
    // 绿色到黄绿色
    const t = u / 30;
    const r = Math.round(100 * t);
    const g = 200;
    const b = Math.round(100 * (1 - t));
    return `rgb(${r}, ${g}, ${b})`;
  } else if (u < 60) {
    // 黄绿色到黄色
    const t = (u - 30) / 30;
    const r = Math.round(100 + 155 * t);
    const g = 200;
    const b = 0;
    return `rgb(${r}, ${g}, ${b})`;
  } else if (u < 80) {
    // 黄色到橙色
    const t = (u - 60) / 20;
    const r = 255;
    const g = Math.round(200 - 100 * t);
    const b = 0;
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // 橙色到红色
    const t = (u - 80) / 20;
    const r = 255;
    const g = Math.round(100 - 100 * t);
    const b = 0;
    return `rgb(${r}, ${g}, ${b})`;
  }
}

/**
 * 获取链路宽度
 * 根据流量大小映射到 2-6px
 */
export function getHeatmapWidth(trafficMb: number, maxTrafficMb: number): number {
  if (maxTrafficMb === 0) return 2;
  const ratio = trafficMb / maxTrafficMb;
  return 2 + ratio * 4; // 2-6px
}
