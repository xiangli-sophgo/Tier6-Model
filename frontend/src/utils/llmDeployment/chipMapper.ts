/**
 * 芯片自动放置模块
 *
 * 根据拓扑结构和并行策略，自动将物理芯片分配到并行组
 * 优化策略：TP组优先放同Board/Rack内，PP组可跨Rack，DP组可跨Pod
 */

import { HierarchicalTopology, PodConfig, RackConfig, BoardConfig, ChipConfig } from '../../types';
import {
  ParallelismStrategy,
  ChipMapping,
  CommunicationGroup,
  ParallelismType,
  CollectiveOp,
  CommunicationAnalysis,
} from './types';

// ============================================
// 辅助类型
// ============================================

/** 芯片物理信息 */
interface PhysicalChip {
  chipId: string;
  podId: string;
  rackId: string;
  boardId: string;
  podIndex: number;
  rackIndex: number;
  boardIndex: number;
  chipIndex: number;
}

// ============================================
// 辅助函数
// ============================================

/**
 * 从拓扑中收集所有芯片信息
 */
function collectChipsFromTopology(topology: HierarchicalTopology): PhysicalChip[] {
  const chips: PhysicalChip[] = [];

  topology.pods.forEach((pod: PodConfig, podIndex: number) => {
    pod.racks.forEach((rack: RackConfig, rackIndex: number) => {
      rack.boards.forEach((board: BoardConfig, boardIndex: number) => {
        board.chips.forEach((chip: ChipConfig, chipIndex: number) => {
          chips.push({
            chipId: chip.id,
            podId: pod.id,
            rackId: rack.id,
            boardId: board.id,
            podIndex,
            rackIndex,
            boardIndex,
            chipIndex,
          });
        });
      });
    });
  });

  return chips;
}

/**
 * 计算并行组的rank
 * 使用列主序: rank = dp * (TP * PP * EP * SP) + tp * (PP * EP * SP) + pp * (EP * SP) + ep * SP + sp
 */
function calculateParallelismRank(
  globalRank: number,
  parallelism: ParallelismStrategy
): { dp: number; tp: number; pp: number; ep: number; sp: number } {
  const { dp, tp, pp, ep, sp } = parallelism;

  // 从全局rank反推各维度rank
  // 顺序: SP -> EP -> PP -> TP -> DP (最内层到最外层)
  let remainder = globalRank;

  const spRank = remainder % sp;
  remainder = Math.floor(remainder / sp);

  const epRank = remainder % ep;
  remainder = Math.floor(remainder / ep);

  const ppRank = remainder % pp;
  remainder = Math.floor(remainder / pp);

  const tpRank = remainder % tp;
  remainder = Math.floor(remainder / tp);

  const dpRank = remainder % dp;

  return { dp: dpRank, tp: tpRank, pp: ppRank, ep: epRank, sp: spRank };
}

/**
 * 芯片放置优化：重排芯片顺序以优化通信
 * 策略：优先将同一TP组的芯片放在同Board/Rack内
 */
function optimizeChipPlacement(
  chips: PhysicalChip[],
  _parallelism: ParallelismStrategy
): PhysicalChip[] {

  // 按层级结构排序：Pod -> Rack -> Board -> Chip
  // 这样相邻的global rank会在物理上接近
  const sorted = [...chips].sort((a, b) => {
    // 首先按Pod
    if (a.podIndex !== b.podIndex) return a.podIndex - b.podIndex;
    // 然后按Rack
    if (a.rackIndex !== b.rackIndex) return a.rackIndex - b.rackIndex;
    // 然后按Board
    if (a.boardIndex !== b.boardIndex) return a.boardIndex - b.boardIndex;
    // 最后按Chip
    return a.chipIndex - b.chipIndex;
  });

  // 如果TP > 1，尝试让TP组内的芯片物理相邻
  // 当前简单实现：使用排序后的顺序，TP组自然在相邻位置
  // 更复杂的优化可以考虑：
  // 1. 统计每个Board的芯片数，确保TP组尽量在同Board
  // 2. 如果Board芯片数 < TP，则在同Rack内分配

  // TODO: 高级优化 - 根据Board芯片数调整分配策略

  return sorted;
}

// ============================================
// 主函数
// ============================================

/**
 * 自动将芯片映射到并行组
 */
export function autoMapChipsToParallelism(
  topology: HierarchicalTopology,
  parallelism: ParallelismStrategy
): { mapping: ChipMapping[]; totalChips: number; requiredChips: number } {
  // 收集所有芯片
  const chips = collectChipsFromTopology(topology);
  const totalChips = chips.length;

  // 计算所需芯片数
  const requiredChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep * parallelism.sp;

  if (totalChips < requiredChips) {
    throw new Error(`芯片数量不足: 需要 ${requiredChips} 个，实际 ${totalChips} 个`);
  }

  // 优化芯片放置顺序
  const optimizedChips = optimizeChipPlacement(chips, parallelism);

  // 创建映射
  const mapping: ChipMapping[] = optimizedChips.slice(0, requiredChips).map((chip, globalRank) => {
    const ranks = calculateParallelismRank(globalRank, parallelism);

    return {
      chipId: chip.chipId,
      physicalLocation: {
        pod: chip.podId,
        rack: chip.rackId,
        board: chip.boardId,
      },
      parallelismRank: ranks,
      globalRank,
    };
  });

  return { mapping, totalChips, requiredChips };
}

/**
 * 根据芯片映射生成通信组
 */
export function generateCommunicationGroups(
  mapping: ChipMapping[],
  parallelism: ParallelismStrategy,
  commAnalysis: CommunicationAnalysis
): CommunicationGroup[] {
  const groups: CommunicationGroup[] = [];
  const { dp, tp, pp, ep, sp } = parallelism;

  // 辅助函数：根据条件筛选芯片
  const filterChips = (
    fixedRanks: Partial<{ dp: number; tp: number; pp: number; ep: number; sp: number }>
  ): string[] => {
    return mapping
      .filter(m => {
        for (const [key, value] of Object.entries(fixedRanks)) {
          if (m.parallelismRank[key as keyof typeof m.parallelismRank] !== value) {
            return false;
          }
        }
        return true;
      })
      .sort((a, b) => {
        // 按照剩余维度排序，确保组内顺序一致
        return a.globalRank - b.globalRank;
      })
      .map(m => m.chipId);
  };

  // 生成 TP 通信组 (AllReduce)
  if (tp > 1) {
    for (let d = 0; d < dp; d++) {
      for (let p = 0; p < pp; p++) {
        for (let e = 0; e < ep; e++) {
          for (let s = 0; s < sp; s++) {
            const members = filterChips({ dp: d, pp: p, ep: e, sp: s });
            if (members.length > 1) {
              groups.push({
                id: `tp_d${d}_p${p}_e${e}_s${s}`,
                type: 'tp',
                members,
                collectiveOp: 'allreduce',
                messageSizeMb: commAnalysis.tp_comm_volume_gb * 1024 / (dp * pp * ep * sp), // 分摊到每个组
                frequency: 2, // 每层2次 (forward + backward的AllReduce)
              });
            }
          }
        }
      }
    }
  }

  // 生成 PP 通信组 (P2P)
  if (pp > 1) {
    for (let d = 0; d < dp; d++) {
      for (let t = 0; t < tp; t++) {
        for (let e = 0; e < ep; e++) {
          for (let s = 0; s < sp; s++) {
            const members = filterChips({ dp: d, tp: t, ep: e, sp: s });
            if (members.length > 1) {
              groups.push({
                id: `pp_d${d}_t${t}_e${e}_s${s}`,
                type: 'pp',
                members,
                collectiveOp: 'p2p',
                messageSizeMb: commAnalysis.pp_comm_volume_gb * 1024 / (dp * tp * ep * sp),
                frequency: 1, // 流水线传递
              });
            }
          }
        }
      }
    }
  }

  // 生成 DP 通信组 (AllReduce for gradients)
  if (dp > 1) {
    for (let t = 0; t < tp; t++) {
      for (let p = 0; p < pp; p++) {
        for (let e = 0; e < ep; e++) {
          for (let s = 0; s < sp; s++) {
            const members = filterChips({ tp: t, pp: p, ep: e, sp: s });
            if (members.length > 1) {
              groups.push({
                id: `dp_t${t}_p${p}_e${e}_s${s}`,
                type: 'dp',
                members,
                collectiveOp: 'allreduce',
                messageSizeMb: 0, // 推理时DP组通信量为0
                frequency: 0,
              });
            }
          }
        }
      }
    }
  }

  // 生成 EP 通信组 (AllToAll)
  if (ep > 1) {
    for (let d = 0; d < dp; d++) {
      for (let t = 0; t < tp; t++) {
        for (let p = 0; p < pp; p++) {
          for (let s = 0; s < sp; s++) {
            const members = filterChips({ dp: d, tp: t, pp: p, sp: s });
            if (members.length > 1) {
              groups.push({
                id: `ep_d${d}_t${t}_p${p}_s${s}`,
                type: 'ep',
                members,
                collectiveOp: 'alltoall',
                messageSizeMb: commAnalysis.ep_comm_volume_gb * 1024 / (dp * tp * pp * sp),
                frequency: 2, // MoE层的dispatch和combine
              });
            }
          }
        }
      }
    }
  }

  // 生成 SP 通信组 (AllGather/ReduceScatter)
  if (sp > 1) {
    for (let d = 0; d < dp; d++) {
      for (let t = 0; t < tp; t++) {
        for (let p = 0; p < pp; p++) {
          for (let e = 0; e < ep; e++) {
            const members = filterChips({ dp: d, tp: t, pp: p, ep: e });
            if (members.length > 1) {
              groups.push({
                id: `sp_d${d}_t${t}_p${p}_e${e}`,
                type: 'sp',
                members,
                collectiveOp: 'allgather',
                messageSizeMb: commAnalysis.sp_comm_volume_gb * 1024 / (dp * tp * pp * ep),
                frequency: 2,
              });
            }
          }
        }
      }
    }
  }

  return groups;
}

/**
 * 获取通信组的集合操作类型说明
 */
export function getCollectiveOpDescription(op: CollectiveOp): string {
  const descriptions: Record<CollectiveOp, string> = {
    allreduce: 'AllReduce (Ring)',
    p2p: 'Point-to-Point',
    alltoall: 'All-to-All',
    allgather: 'AllGather',
    reduce_scatter: 'ReduceScatter',
  };
  return descriptions[op];
}

/**
 * 获取并行类型的说明
 */
export function getParallelismTypeDescription(type: ParallelismType): string {
  const descriptions: Record<ParallelismType, string> = {
    tp: '张量并行 (Tensor Parallelism)',
    pp: '流水线并行 (Pipeline Parallelism)',
    dp: '数据并行 (Data Parallelism)',
    ep: '专家并行 (Expert Parallelism)',
    sp: '序列并行 (Sequence Parallelism)',
  };
  return descriptions[type];
}
