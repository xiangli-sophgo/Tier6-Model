/**
 * 层级拓扑生成器
 *
 * 生成数据中心层级拓扑: Pod -> Rack -> Board -> Chip
 * 移植自 Python 后端 topology.py
 */

import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  ChipConfig,
  ConnectionConfig,
  SwitchInstance,
  GlobalSwitchConfig,
  ManualConnectionConfig,
  SwitchLayerConfig,
  SwitchTypeConfig,
} from '../types';

// 各层级连接的默认参数配置 (带宽: GB/s, 延迟: us)
export const LEVEL_CONNECTION_DEFAULTS = {
  datacenter: { bandwidth: 50.0, latency: 5.0 },      // Pod间: 50 GB/s, 5 us
  pod: { bandwidth: 50.0, latency: 2.0 },             // Rack间: 50 GB/s, 2 us (InfiniBand)
  rack: { bandwidth: 64.0, latency: 15.0 },           // Board间: 64 GB/s, 15 us (PCIe)
  board: { bandwidth: 900.0, latency: 1.0 },          // Chip间: 900 GB/s, 1 us (NVLink)
};

// 直连拓扑类型
type DirectTopologyType = 'none' | 'full_mesh' | 'full_mesh_2d' | 'ring' | 'torus_2d' | 'torus_3d';

// 板卡配置（灵活模式）
interface FlexBoardConfig {
  id: string;
  name: string;
  u_height: number;
  count: number;
  chips: Array<{ name: string; count: number }>;
}

// Rack配置（灵活模式）
interface FlexRackConfig {
  total_u: number;
  boards: FlexBoardConfig[];
}

// 板卡配置（传统模式）
interface BoardConfigByType {
  u1: { count: number; chips: { npu: number; cpu: number } };
  u2: { count: number; chips: { npu: number; cpu: number } };
  u4: { count: number; chips: { npu: number; cpu: number } };
}

// 互联参数配置
export interface InterconnectParams {
  bandwidth_gbps: number;
  latency_us: number;
}

// 层级互联配置
export interface InterconnectConfig {
  c2c?: InterconnectParams;  // Chip-to-Chip (板内)
  b2b?: InterconnectParams;  // Board-to-Board (机架内)
  r2r?: InterconnectParams;  // Rack-to-Rack (Pod内)
  p2p?: InterconnectParams;  // Pod-to-Pod (数据中心内)
}

// 生成请求参数
export interface TopologyGenerateRequest {
  pod_count?: number;
  racks_per_pod?: number;
  rack_config?: FlexRackConfig;
  switch_config?: GlobalSwitchConfig;
  manual_connections?: ManualConnectionConfig;
  interconnect_config?: InterconnectConfig;  // 互联参数配置
}

/**
 * 根据节点 ID 自动推断连接类型
 */
function inferConnectionType(source: string, target: string): 'c2c' | 'b2b' | 'r2r' | 'p2p' {
  // 判断最深层级
  if (source.includes('/chip_') && target.includes('/chip_')) {
    return 'c2c';  // Chip-to-Chip
  } else if (source.includes('/board_') && target.includes('/board_')
             && !source.includes('/chip_') && !target.includes('/chip_')) {
    return 'b2b';  // Board-to-Board
  } else if (source.includes('/rack_') && target.includes('/rack_')
             && !source.includes('/board_')) {
    return 'r2r';  // Rack-to-Rack
  } else if (source.startsWith('pod_') && target.startsWith('pod_')
             && source.split('/').length === 1 && target.split('/').length === 1) {
    return 'p2p';  // Pod-to-Pod
  }

  // 默认返回 c2c（后备方案）
  return 'c2c';
}

/**
 * 层级拓扑生成器类
 */
export class HierarchicalTopologyGenerator {
  private cachedTopology: HierarchicalTopology | null = null;

  /**
   * 生成默认的示例拓扑
   */
  generateDefault(): HierarchicalTopology {
    return this.generate({
      pod_count: 1,
      racks_per_pod: 4,
      rack_config: {
        total_u: 42,
        boards: [
          { id: 'board_1', name: 'Board', u_height: 2, count: 8, chips: [{ name: 'NPU', count: 8 }] },
        ],
      },
    });
  }

  /**
   * 根据配置生成拓扑
   */
  generate(request: TopologyGenerateRequest): HierarchicalTopology {
    const {
      pod_count = 1,
      racks_per_pod = 4,
      rack_config,
      switch_config,
      manual_connections,
      interconnect_config,
    } = request;

    // 合并用户配置与默认值，统一转换为 { bandwidth, latency } 格式
    const toConnParams = (cfg: InterconnectParams | undefined, defaults: { bandwidth: number; latency: number }) =>
      cfg ? { bandwidth: cfg.bandwidth_gbps, latency: cfg.latency_us } : defaults;

    const effectiveInterconnect = {
      c2c: toConnParams(interconnect_config?.c2c, LEVEL_CONNECTION_DEFAULTS.board),
      b2b: toConnParams(interconnect_config?.b2b, LEVEL_CONNECTION_DEFAULTS.rack),
      r2r: toConnParams(interconnect_config?.r2r, LEVEL_CONNECTION_DEFAULTS.pod),
      p2p: toConnParams(interconnect_config?.p2p, LEVEL_CONNECTION_DEFAULTS.datacenter),
    };

    const pods: PodConfig[] = [];
    let connections: ConnectionConfig[] = [];

    for (let podIdx = 0; podIdx < pod_count; podIdx++) {
      const podId = `pod_${podIdx}`;
      const racks: RackConfig[] = [];

      // 计算Rack网格布局
      let gridCols: number;
      if (racks_per_pod <= 2) {
        gridCols = racks_per_pod;
      } else if (racks_per_pod <= 4) {
        gridCols = 2;
      } else if (racks_per_pod <= 6) {
        gridCols = 3;
      } else if (racks_per_pod <= 9) {
        gridCols = 3;
      } else if (racks_per_pod <= 12) {
        gridCols = 4;
      } else if (racks_per_pod <= 16) {
        gridCols = 4;
      } else {
        gridCols = Math.ceil(Math.sqrt(racks_per_pod));
      }
      const gridRows = Math.ceil(racks_per_pod / gridCols);

      for (let rackIdx = 0; rackIdx < racks_per_pod; rackIdx++) {
        const rackId = `rack_${rackIdx}`;
        const rackFullId = `${podId}/${rackId}`;
        const boards: BoardConfig[] = [];

        // 获取Rack总U数
        const rackTotalU = rack_config?.total_u ?? 42;

        // 计算Switch预留空间
        let switchReservedU = 0;
        let switchPosition: 'top' | 'middle' | 'bottom' = 'top';
        let switchUHeight = 1;
        if (switch_config) {
          const interBoardCfg = switch_config.inter_board;
          if (interBoardCfg?.enabled) {
            switchPosition = interBoardCfg.switch_position ?? 'top';
            switchUHeight = interBoardCfg.switch_u_height ?? 1;
            switchReservedU = switchUHeight;
          }
        }

        // 根据Switch位置确定Board的起始U位置
        let boardStartU: number;
        if (switchPosition === 'bottom') {
          boardStartU = switchReservedU + 1;
        } else {
          boardStartU = 1;
        }

        // 使用 rack_config 生成板卡配置
        if (rack_config && rack_config.boards && rack_config.boards.length > 0) {
          console.log(`🏗️ [TopologyGen] Rack ${rackIdx}: 使用rack_config配置`, {
            rackTotalU,
            boardStartU,
            switchReservedU,
            switchPosition,
            boards: rack_config.boards,
          })
          let currentU = boardStartU;
          let boardIdxLocal = 0;

          for (const flexBoard of rack_config.boards) {
            const uHeight = flexBoard.u_height ?? 2;
            const boardName = flexBoard.name ?? 'Board';
            const boardCount = flexBoard.count ?? 1;
            const flexChips = flexBoard.chips ?? [];

            console.log(`  📦 [Board配置] ${boardName}: count=${boardCount}, uHeight=${uHeight}, currentU=${currentU}`)

            for (let i = 0; i < boardCount; i++) {
              if (currentU + uHeight - 1 > rackTotalU) {
                console.warn(`  ⚠️ [跳过Board] 超出容量: currentU=${currentU}, uHeight=${uHeight}, rackTotalU=${rackTotalU}`)
                break; // 超出机柜容量
              }

              const boardFullId = `${rackFullId}/board_${boardIdxLocal}`;

              // 使用灵活配置生成Chip
              const chips = this.generateBoardChipsFlex(boardFullId, flexChips);

              boards.push({
                id: boardFullId,
                u_position: currentU,
                u_height: uHeight,
                label: `${boardName}-${boardIdxLocal}`,
                chips,
              });

              // 生成Chip间连接
              const interChipCfg = switch_config?.inter_chip;
              const interChipEnabled = interChipCfg?.enabled ?? false;
              const interChipTopo = (interChipCfg?.direct_topology ?? 'none') as DirectTopologyType;
              const keepDirect = interChipCfg?.keep_direct_topology ?? false;

              if (!interChipEnabled || keepDirect) {
                const chipConnections = this.generateChipConnections(chips, interChipTopo, effectiveInterconnect.c2c);
                connections.push(...chipConnections);
              }

              currentU += uHeight;
              boardIdxLocal++;
            }
          }
        }

        // middle模式：在Board中间插入Switch空间
        if (switchPosition === 'middle' && switchReservedU > 0 && boards.length > 0) {
          const halfCount = Math.floor(boards.length / 2);
          if (halfCount > 0) {
            const sortedBoards = [...boards].sort((a, b) => a.u_position - b.u_position);
            const splitBoard = sortedBoards[halfCount - 1];
            const splitU = splitBoard.u_position + splitBoard.u_height;

            for (const board of boards) {
              if (board.u_position >= splitU) {
                board.u_position += switchReservedU;
              }
            }
          }
        }

        racks.push({
          id: rackFullId,
          position: [Math.floor(rackIdx / gridCols), rackIdx % gridCols],
          label: `Rack-${rackIdx}`,
          total_u: rackTotalU,
          boards,
        });

        // 如果没有启用inter_board Switch，或者配置了keep_direct_topology，生成Board间直连
        const interBoardCfg = switch_config?.inter_board;
        const interBoardEnabled = interBoardCfg?.enabled ?? false;
        const interBoardTopo = (interBoardCfg?.direct_topology ?? 'full_mesh') as DirectTopologyType;
        const keepDirectBoard = interBoardCfg?.keep_direct_topology ?? false;

        if ((!interBoardEnabled || keepDirectBoard) && boards.length > 1) {
          const boardConnections = this.generateBoardConnections(boards, interBoardTopo);
          connections.push(...boardConnections);
        }
      }

      pods.push({
        id: podId,
        label: `Pod-${podIdx}`,
        grid_size: [gridRows, gridCols],
        racks,
      });
    }

    // ============================================
    // Switch生成
    // ============================================
    let switches: SwitchInstance[] = [];

    if (switch_config) {
      const switchTypesList = switch_config.switch_types ?? [];

      // 0. Board层Switch（Chip间）
      const interChipConfig = switch_config.inter_chip;
      if (interChipConfig?.enabled && interChipConfig.layers?.length > 0) {
        for (const pod of pods) {
          for (const rack of pod.racks) {
            for (const board of rack.boards) {
              const chipIds = board.chips.map(c => c.id);
              if (chipIds.length > 0) {
                const [boardSwitches, boardSwitchConns] = this.generateSwitchConnections({
                  switchLayers: interChipConfig.layers,
                  switchTypes: switchTypesList,
                  devices: chipIds,
                  redundancy: interChipConfig.downlink_redundancy ?? 1,
                  parentId: board.id,
                  hierarchyLevel: 'inter_chip',
                  connectionMode: interChipConfig.connection_mode ?? 'full_mesh',
                  customConnections: interChipConfig.custom_connections,
                });
                switches.push(...boardSwitches);
                connections.push(...boardSwitchConns);
              }
            }
          }
        }
      }

      // 1. Rack层Switch（Board间）
      const interBoardConfig = switch_config.inter_board;
      if (interBoardConfig?.enabled && interBoardConfig.layers?.length > 0) {
        const switchPositionB = interBoardConfig.switch_position ?? 'top';
        const switchUHeightB = interBoardConfig.switch_u_height ?? 1;

        for (const pod of pods) {
          for (const rack of pod.racks) {
            let deviceIds: string[];

            if (interChipConfig?.enabled && interChipConfig.connect_to_upper_level !== false) {
              const topBoardSwitches = this.getTopLayerSwitches(switches, 'inter_chip');
              deviceIds = topBoardSwitches
                .filter(s => s.parent_id?.startsWith(rack.id))
                .map(s => s.id);
            } else {
              deviceIds = rack.boards.map(b => b.id);
            }

            if (deviceIds.length === 0) continue;

            const [rackSwitches, rackSwitchConns] = this.generateSwitchConnections({
              switchLayers: interBoardConfig.layers,
              switchTypes: switchTypesList,
              devices: deviceIds,
              redundancy: interBoardConfig.downlink_redundancy ?? 1,
              parentId: rack.id,
              hierarchyLevel: 'inter_board',
              connectionMode: interBoardConfig.connection_mode ?? 'full_mesh',
              customConnections: interBoardConfig.custom_connections,
            });

            // 分配u_position
            let switchU: number;
            if (switchPositionB === 'bottom') {
              switchU = 1;
            } else if (switchPositionB === 'middle') {
              if (rack.boards.length > 0) {
                const halfCount = Math.floor(rack.boards.length / 2);
                if (halfCount > 0) {
                  const sortedBoards = [...rack.boards].sort((a, b) => a.u_position - b.u_position);
                  const splitBoard = sortedBoards[halfCount - 1];
                  switchU = splitBoard.u_position + splitBoard.u_height;
                } else {
                  switchU = 1;
                }
              } else {
                switchU = 1;
              }
            } else {
              const maxBoardU = Math.max(...rack.boards.map(b => b.u_position + b.u_height - 1), 0);
              switchU = maxBoardU + 1;
            }

            for (const sw of rackSwitches) {
              sw.u_position = switchU;
              sw.u_height = switchUHeightB;
            }

            switches.push(...rackSwitches);
            connections.push(...rackSwitchConns);
          }
        }
      }

      // 2. Pod层Switch（Rack间）
      const interRackConfig = switch_config.inter_rack;
      if (interRackConfig?.enabled && interRackConfig.layers?.length > 0) {
        for (const pod of pods) {
          let deviceIds: string[];

          if (interBoardConfig?.enabled && interBoardConfig.connect_to_upper_level !== false) {
            const topRackSwitches = this.getTopLayerSwitches(switches, 'inter_board');
            deviceIds = topRackSwitches
              .filter(s => s.parent_id?.startsWith(pod.id))
              .map(s => s.id);
          } else {
            deviceIds = pod.racks.map(r => r.id);
          }

          if (deviceIds.length > 0) {
            const [podSwitches, podSwitchConns] = this.generateSwitchConnections({
              switchLayers: interRackConfig.layers,
              switchTypes: switchTypesList,
              devices: deviceIds,
              redundancy: interRackConfig.downlink_redundancy ?? 1,
              parentId: pod.id,
              hierarchyLevel: 'inter_rack',
              connectionMode: interRackConfig.connection_mode ?? 'full_mesh',
              customConnections: interRackConfig.custom_connections,
            });
            switches.push(...podSwitches);
            connections.push(...podSwitchConns);
          }
        }
      }

      // 3. 数据中心层Switch（Pod间）
      const dcLevelConfig = switch_config.inter_pod;
      if (dcLevelConfig?.enabled && dcLevelConfig.layers?.length > 0) {
        let deviceIds: string[];

        if (interRackConfig?.enabled && interRackConfig.connect_to_upper_level !== false) {
          const topPodSwitches = this.getTopLayerSwitches(switches, 'inter_rack');
          deviceIds = topPodSwitches.map(s => s.id);
        } else {
          deviceIds = pods.map(p => p.id);
        }

        if (deviceIds.length > 0) {
          const [dcSwitches, dcSwitchConns] = this.generateSwitchConnections({
            switchLayers: dcLevelConfig.layers,
            switchTypes: switchTypesList,
            devices: deviceIds,
            redundancy: dcLevelConfig.downlink_redundancy ?? 1,
            parentId: '',
            hierarchyLevel: 'inter_pod',
            connectionMode: dcLevelConfig.connection_mode ?? 'full_mesh',
            customConnections: dcLevelConfig.custom_connections,
          });
          switches.push(...dcSwitches);
          connections.push(...dcSwitchConns);
        }
      }
    }

    // 处理手动连接
    let manualConfig: ManualConnectionConfig | undefined;
    if (manual_connections?.enabled) {
      manualConfig = manual_connections;
      const mode = manual_connections.mode ?? 'append';
      const manualConnList = manual_connections.connections ?? [];

      if (mode === 'replace') {
        const levelsWithManual = new Set(manualConnList.map(c => c.hierarchy_level));
        connections = connections.filter(c => !this.isConnectionInLevel(c, levelsWithManual, pods));
      }

      for (const mc of manualConnList) {
        // 判断是否为自定义连接（手动指定了带宽/延迟）
        const hasCustomParams = mc.bandwidth !== undefined && mc.latency !== undefined;

        if (hasCustomParams) {
          // 手动指定了参数，使用 custom 类型
          connections.push({
            source: mc.source,
            target: mc.target,
            type: 'custom',
            bandwidth: mc.bandwidth,
            latency: mc.latency,
            is_manual: true,
          });
        } else {
          // 没有指定参数，自动推断类型（引用 interconnect 配置）
          connections.push({
            source: mc.source,
            target: mc.target,
            type: inferConnectionType(mc.source, mc.target),
            is_manual: true,
          });
        }
      }
    }

    const topology: HierarchicalTopology = {
      pods,
      connections,
      switches,
      switch_config,
      manual_connections: manualConfig,
    };

    this.cachedTopology = topology;
    return topology;
  }

  /**
   * 判断连接是否属于指定层级
   */
  private isConnectionInLevel(
    connection: ConnectionConfig,
    levels: Set<string>,
    _pods: PodConfig[]
  ): boolean {
    const source = connection.source;
    const target = connection.target;

    const sourceParts = source.split('/');
    const targetParts = target.split('/');

    if (levels.has('datacenter')) {
      if (sourceParts.length === 1 && source.startsWith('pod_') &&
          targetParts.length === 1 && target.startsWith('pod_')) {
        return true;
      }
    }

    if (levels.has('pod')) {
      if (sourceParts.length === 2 && source.includes('rack_') &&
          targetParts.length === 2 && target.includes('rack_')) {
        return true;
      }
    }

    if (levels.has('rack')) {
      if (sourceParts.length === 3 && source.includes('board_') &&
          targetParts.length === 3 && target.includes('board_')) {
        return true;
      }
    }

    if (levels.has('board')) {
      if (source.includes('chip_') && target.includes('chip_')) {
        return true;
      }
    }

    return false;
  }

  /**
   * 生成板卡上的芯片（传统模式）
   */
  private generateBoardChips(
    boardId: string,
    chipCounts: { npu: number; cpu: number }
  ): ChipConfig[] {
    const chips: ChipConfig[] = [];

    const totalChips = chipCounts.npu + chipCounts.cpu;
    if (totalChips === 0) return chips;

    const cols = Math.ceil(Math.sqrt(totalChips));

    const chipList: Array<{ type: 'npu' | 'cpu'; label: string }> = [];

    for (let i = 0; i < chipCounts.npu; i++) {
      chipList.push({ type: 'npu', label: `NPU-${i}` });
    }
    for (let i = 0; i < chipCounts.cpu; i++) {
      chipList.push({ type: 'cpu', label: `CPU-${i}` });
    }

    for (let idx = 0; idx < chipList.length; idx++) {
      const chipInfo = chipList[idx];
      const row = Math.floor(idx / cols);
      const col = idx % cols;

      chips.push({
        id: `${boardId}/chip_${idx}`,
        type: chipInfo.type as any,
        position: [row, col],
        label: chipInfo.label,
      });
    }

    return chips;
  }

  /**
   * 使用灵活配置生成板卡上的芯片
   */
  private generateBoardChipsFlex(
    boardId: string,
    flexChips: Array<{ name: string; count: number }>
  ): ChipConfig[] {
    const chips: ChipConfig[] = [];

    const totalChips = flexChips.reduce((sum, fc) => sum + (fc.count ?? 1), 0);
    if (totalChips === 0) return chips;

    const cols = Math.ceil(Math.sqrt(totalChips));

    const chipList: Array<{ type: string; label: string }> = [];

    for (const fc of flexChips) {
      const chipName = fc.name ?? 'CHIP';
      const chipCount = fc.count ?? 1;
      let chipType = chipName.toLowerCase();
      if (!['npu', 'cpu'].includes(chipType)) {
        chipType = 'npu';
      }

      for (let i = 0; i < chipCount; i++) {
        chipList.push({ type: chipType, label: `${chipName}-${i}` });
      }
    }

    for (let idx = 0; idx < chipList.length; idx++) {
      const chipInfo = chipList[idx];
      const row = Math.floor(idx / cols);
      const col = idx % cols;

      chips.push({
        id: `${boardId}/chip_${idx}`,
        type: chipInfo.type as any,
        position: [row, col],
        label: chipInfo.label,
      });
    }

    return chips;
  }

  /**
   * 生成芯片间的连接
   * @param chips 芯片配置列表
   * @param topologyType 拓扑类型
   */
  private generateChipConnections(
    chips: ChipConfig[],
    topologyType: DirectTopologyType = 'none'
  ): ConnectionConfig[] {
    const connections: ConnectionConfig[] = [];

    // 根据拓扑类型生成芯片间连接
    if (topologyType !== 'none' && chips.length > 1) {
      const chipIds = chips.map(c => c.id);
      const chipConnections = this.generateDirectConnections(chipIds, topologyType);
      connections.push(...chipConnections);
    }

    return connections;
  }

  /**
   * 生成Board间的连接
   * @param boards Board配置列表
   * @param topologyType 拓扑类型
   */
  private generateBoardConnections(
    boards: BoardConfig[],
    topologyType: DirectTopologyType = 'full_mesh'
  ): ConnectionConfig[] {
    const boardIds = boards.map(b => b.id);
    return this.generateDirectConnections(boardIds, topologyType);
  }

  /**
   * 根据拓扑类型生成直连
   * @param nodeIds 节点ID列表
   * @param topologyType 拓扑类型
   */
  private generateDirectConnections(
    nodeIds: string[],
    topologyType: DirectTopologyType
  ): ConnectionConfig[] {
    const connections: ConnectionConfig[] = [];
    const n = nodeIds.length;

    if (n < 2 || topologyType === 'none') {
      return connections;
    }

    if (topologyType === 'full_mesh') {
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          connections.push({
            source: nodeIds[i],
            target: nodeIds[j],
            type: inferConnectionType(nodeIds[i], nodeIds[j]),
          });
        }
      }
    } else if (topologyType === 'ring') {
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        connections.push({
          source: nodeIds[i],
          target: nodeIds[j],
          type: inferConnectionType(nodeIds[i], nodeIds[j]),
        });
      }
    } else if (topologyType === 'torus_2d') {
      const cols = Math.ceil(Math.sqrt(n));
      const rows = Math.ceil(n / cols);
      for (let i = 0; i < n; i++) {
        const row = Math.floor(i / cols);
        const col = i % cols;
        // 右邻居（环绕）
        const right = row * cols + (col + 1) % cols;
        if (right < n && right !== i) {
          connections.push({
            source: nodeIds[i],
            target: nodeIds[right],
            type: inferConnectionType(nodeIds[i], nodeIds[right]),
          });
        }
        // 下邻居（环绕）
        const down = ((row + 1) % rows) * cols + col;
        if (down < n && down !== i) {
          connections.push({
            source: nodeIds[i],
            target: nodeIds[down],
            type: inferConnectionType(nodeIds[i], nodeIds[down]),
          });
        }
      }
    } else if (topologyType === 'torus_3d') {
      const dim = Math.max(1, Math.round(Math.pow(n, 1/3)));
      for (let i = 0; i < n; i++) {
        const x = i % dim;
        const y = Math.floor(i / dim) % dim;
        const z = Math.floor(i / (dim * dim));
        // X方向邻居
        const nx = y * dim + ((x + 1) % dim) + z * dim * dim;
        if (nx < n && nx !== i && i < nx) {
          connections.push({
            source: nodeIds[i],
            target: nodeIds[nx],
            type: inferConnectionType(nodeIds[i], nodeIds[nx]),
          });
        }
        // Y方向邻居
        const ny = ((y + 1) % dim) * dim + x + z * dim * dim;
        if (ny < n && ny !== i && i < ny) {
          connections.push({
            source: nodeIds[i],
            target: nodeIds[ny],
            type: inferConnectionType(nodeIds[i], nodeIds[ny]),
          });
        }
        // Z方向邻居
        const nz = y * dim + x + ((z + 1) % dim) * dim * dim;
        if (nz < n && nz !== i && i < nz) {
          connections.push({
            source: nodeIds[i],
            target: nodeIds[nz],
            type: inferConnectionType(nodeIds[i], nodeIds[nz]),
          });
        }
      }
    } else if (topologyType === 'full_mesh_2d') {
      const cols = Math.ceil(Math.sqrt(n));
      const rows = Math.ceil(n / cols);
      // 同行全连接
      for (let row = 0; row < rows; row++) {
        const rowNodes: number[] = [];
        for (let i = row * cols; i < Math.min((row + 1) * cols, n); i++) {
          rowNodes.push(i);
        }
        for (let i = 0; i < rowNodes.length; i++) {
          for (let j = i + 1; j < rowNodes.length; j++) {
            connections.push({
              source: nodeIds[rowNodes[i]],
              target: nodeIds[rowNodes[j]],
              type: inferConnectionType(nodeIds[rowNodes[i]], nodeIds[rowNodes[j]]),
            });
          }
        }
      }
      // 同列全连接
      for (let col = 0; col < cols; col++) {
        const colNodes: number[] = [];
        for (let row = 0; row < rows; row++) {
          const idx = row * cols + col;
          if (idx < n) {
            colNodes.push(idx);
          }
        }
        for (let i = 0; i < colNodes.length; i++) {
          for (let j = i + 1; j < colNodes.length; j++) {
            connections.push({
              source: nodeIds[colNodes[i]],
              target: nodeIds[colNodes[j]],
              type: inferConnectionType(nodeIds[colNodes[i]], nodeIds[colNodes[j]]),
            });
          }
        }
      }
    }

    return connections;
  }

  /**
   * 获取缓存的拓扑数据
   */
  getCachedTopology(): HierarchicalTopology {
    if (this.cachedTopology === null) {
      return this.generateDefault();
    }
    return this.cachedTopology;
  }

  /**
   * 获取指定Pod
   */
  getPod(podId: string): PodConfig | undefined {
    const topology = this.getCachedTopology();
    return topology.pods.find(pod => pod.id === podId);
  }

  /**
   * 获取指定Rack
   */
  getRack(rackId: string): RackConfig | undefined {
    const topology = this.getCachedTopology();
    for (const pod of topology.pods) {
      for (const rack of pod.racks) {
        if (rack.id === rackId) {
          return rack;
        }
      }
    }
    return undefined;
  }

  /**
   * 获取指定Board
   */
  getBoard(boardId: string): BoardConfig | undefined {
    const topology = this.getCachedTopology();
    for (const pod of topology.pods) {
      for (const rack of pod.racks) {
        for (const board of rack.boards) {
          if (board.id === boardId) {
            return board;
          }
        }
      }
    }
    return undefined;
  }

  /**
   * 获取指定层级的连接
   */
  getConnectionsForLevel(level: string, parentId?: string): ConnectionConfig[] {
    const topology = this.getCachedTopology();
    const connections: ConnectionConfig[] = [];

    for (const conn of topology.connections) {
      if (level === 'rack') {
        if (conn.source.includes('/rack_') && !conn.source.includes('/board_')) {
          if (!parentId || conn.source.includes(parentId)) {
            connections.push(conn);
          }
        }
      } else if (level === 'board') {
        if (conn.source.includes('/board_') && !conn.source.includes('/chip_')) {
          if (!parentId || conn.source.includes(parentId)) {
            connections.push(conn);
          }
        }
      } else if (level === 'chip') {
        if (conn.source.includes('/chip_')) {
          if (!parentId || conn.source.includes(parentId)) {
            connections.push(conn);
          }
        }
      }
    }

    return connections;
  }

  // ============================================
  // Switch生成相关方法
  // ============================================

  /**
   * 通用Switch连接生成算法
   */
  private generateSwitchConnections(params: {
    switchLayers: SwitchLayerConfig[];
    switchTypes: SwitchTypeConfig[];
    devices: string[];
    redundancy: number;
    parentId: string;
    hierarchyLevel: 'inter_pod' | 'inter_rack' | 'inter_board' | 'inter_chip';
    connectionMode?: string;
    customConnections?: Array<{ device_id: string; switch_indices: number[] }>;
  }): [SwitchInstance[], ConnectionConfig[]] {
    const {
      switchLayers,
      switchTypes,
      devices,
      redundancy,
      parentId,
      hierarchyLevel,
      connectionMode = 'full_mesh',
      customConnections,
    } = params;

    const switches: SwitchInstance[] = [];
    const connections: ConnectionConfig[] = [];

    if (!switchLayers || switchLayers.length === 0 || devices.length === 0) {
      return [switches, connections];
    }

    // 获取Switch类型映射
    const typeMap: Record<string, SwitchTypeConfig> = {};
    for (const t of switchTypes) {
      typeMap[t.id] = t;
    }

    // 1. 创建Switch实例
    const layerSwitches: Record<string, SwitchInstance[]> = {};

    for (let layerIdx = 0; layerIdx < switchLayers.length; layerIdx++) {
      const layerConfig = switchLayers[layerIdx];
      const layerName = layerConfig.layer_name;
      const switchType = typeMap[layerConfig.switch_type_id];

      if (!switchType) {
        throw new Error(`未找到Switch类型: ${layerConfig.switch_type_id}`);
      }

      layerSwitches[layerName] = [];

      for (let i = 0; i < layerConfig.count; i++) {
        const switchId = parentId ? `${parentId}/${layerName}_${i}` : `${layerName}_${i}`;

        const sw: SwitchInstance = {
          id: switchId,
          type_id: layerConfig.switch_type_id,
          layer: layerName,
          hierarchy_level: hierarchyLevel,
          parent_id: parentId || undefined,
          label: `${switchType.name}-${i}`,
          uplink_ports_used: 0,
          downlink_ports_used: 0,
          inter_ports_used: 0,
          u_height: hierarchyLevel === 'inter_board' ? 1 : undefined,
        };

        switches.push(sw);
        layerSwitches[layerName].push(sw);
      }
    }

    // 2. 设备连接到最底层Switch
    const bottomLayer = switchLayers[0].layer_name;
    const bottomSwitches = layerSwitches[bottomLayer];

    if (connectionMode === 'custom') {
      const connCount = Math.min(redundancy, bottomSwitches.length);
      if (customConnections && customConnections.length > 0) {
        for (const customConn of customConnections) {
          const deviceId = customConn.device_id;
          const switchIndices = customConn.switch_indices;

          if (devices.includes(deviceId)) {
            for (const switchIdx of switchIndices) {
              if (switchIdx < bottomSwitches.length) {
                const sw = bottomSwitches[switchIdx];
                connections.push({
                  source: deviceId,
                  target: sw.id,
                  type: 'switch',
                  connection_role: 'downlink',
                  latency: 100.0,
                });
                sw.downlink_ports_used++;
              }
            }
          }
        }
      } else {
        for (let devIdx = 0; devIdx < devices.length; devIdx++) {
          const deviceId = devices[devIdx];
          for (let r = 0; r < connCount; r++) {
            const switchIdx = (devIdx + r) % bottomSwitches.length;
            const sw = bottomSwitches[switchIdx];
            connections.push({
              source: deviceId,
              target: sw.id,
              type: 'switch',
              connection_role: 'downlink',
              latency: 100.0,
            });
            sw.downlink_ports_used++;
          }
        }
      }
    } else {
      // full_mesh (默认)
      for (const deviceId of devices) {
        for (const sw of bottomSwitches) {
          connections.push({
            source: deviceId,
            target: sw.id,
            type: 'switch',
            connection_role: 'downlink',
            latency: 100.0,
          });
          sw.downlink_ports_used++;
        }
      }
    }

    // 3. 相邻层Switch全连接
    for (let i = 0; i < switchLayers.length - 1; i++) {
      const lowerLayer = switchLayers[i].layer_name;
      const upperLayer = switchLayers[i + 1].layer_name;

      for (const lowerSw of layerSwitches[lowerLayer]) {
        for (const upperSw of layerSwitches[upperLayer]) {
          connections.push({
            source: lowerSw.id,
            target: upperSw.id,
            type: 'switch',
            connection_role: 'uplink',
            latency: 100.0,
          });
          lowerSw.uplink_ports_used++;
          upperSw.downlink_ports_used++;
        }
      }
    }

    // 4. 同层Switch互联
    for (const layerConfig of switchLayers) {
      if (layerConfig.inter_connect) {
        const layerName = layerConfig.layer_name;
        const swList = layerSwitches[layerName];
        for (let i = 0; i < swList.length; i++) {
          for (let j = i + 1; j < swList.length; j++) {
            connections.push({
              source: swList[i].id,
              target: swList[j].id,
              type: 'switch',
              connection_role: 'inter',
              latency: 100.0,
            });
            swList[i].inter_ports_used++;
            swList[j].inter_ports_used++;
          }
        }
      }
    }

    return [switches, connections];
  }

  /**
   * 获取指定层级的顶层Switch
   */
  private getTopLayerSwitches(
    switches: SwitchInstance[],
    hierarchyLevel: string,
    parentId?: string
  ): SwitchInstance[] {
    const levelSwitches = switches.filter(
      s => s.hierarchy_level === hierarchyLevel &&
           (!parentId || s.parent_id === parentId)
    );

    if (levelSwitches.length === 0) {
      return [];
    }

    const layerOrder: Record<string, number> = { leaf: 0, spine: 1, core: 2 };
    const maxLayer = levelSwitches.reduce(
      (max, s) => Math.max(max, layerOrder[s.layer] ?? 0),
      0
    );
    const maxLayerName = Object.entries(layerOrder).find(([, v]) => v === maxLayer)?.[0] ?? 'leaf';

    return levelSwitches.filter(s => s.layer === maxLayerName);
  }
}

// 全局生成器实例
export const topologyGenerator = new HierarchicalTopologyGenerator();
