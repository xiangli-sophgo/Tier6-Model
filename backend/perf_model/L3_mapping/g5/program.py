"""CoreProgram / TIUCommand / DMACommand / SDMACommand / HAUCommand 数据结构 (G5 指令级仿真模式)

定义 L3.g5 指令生成层的核心数据结构:
    - TIUCommand: TIU 计算指令 (MatMul 等)
    - DMACommand: GDMA 搬运指令 (DDR <-> LMEM)
    - SDMACommand: SDMA 搬运指令 (核间通信, GMEM <-> GMEM)
    - HAUCommand: HAU 指令 (硬件排序/Top-K)
    - CoreInstructions: 单核指令序列
    - CoreProgram: 多核程序

参考设计: docs/plans/G5-instruction-level-simulator.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ========== 枚举 ==========

class TIUOpType(Enum):
    """TIU 操作类型"""
    MM2_NN = auto()    # MatMul, A(MxK) * B(KxN) -> C(MxN)
    # 后续扩展:
    # MM2_NT = auto()
    # CONV = auto()
    # SFU = auto()
    # AR = auto()


class DMADirection(Enum):
    """DMA 搬运方向"""
    DDR_TO_LMEM = auto()
    LMEM_TO_DDR = auto()


class HAUOpType(Enum):
    """HAU 操作类型"""
    SORT = auto()
    SORT_INDEX = auto()
    TOP_K = auto()
    UNIQUE = auto()


class SDMACommandType(Enum):
    """SDMA 命令类型"""
    TENSOR = auto()      # 常规张量传输
    GATHER = auto()      # MoE: 按索引收集 expert 输出
    SCATTER = auto()     # MoE: 按索引分发 token 到 expert


class HAUMsgAction(Enum):
    """HAU 与 SDMA 联动动作"""
    NONE = auto()        # 无联动
    SEND = auto()        # 完成后触发 SDMA 发送
    WAIT = auto()        # 等待 SDMA 消息


# ========== 指令数据结构 ==========

@dataclass
class TIUCommand:
    """TIU 计算指令

    Attributes:
        cmd_id: 指令 ID (单调递增)
        cmd_id_dep: 等待 tdma_sync_id >= 此值才可发射, 0=无依赖
        op_type: 操作类型
        result_addr: LMEM 结果地址
        operand_addrs: LMEM 操作数地址 [A_addr, B_addr]
        tile_m: M 维度 tile 大小
        tile_n: N 维度 tile 大小
        tile_k: K 维度 tile 大小
        precision: 精度 (BF16/INT8/FP16)
        source_op_id: 关联的 DistributedOp ID
    """
    cmd_id: int
    cmd_id_dep: int
    op_type: TIUOpType
    result_addr: int
    operand_addrs: list[int]
    tile_m: int
    tile_n: int
    tile_k: int
    precision: str
    source_op_id: str


@dataclass
class DMACommand:
    """GDMA 搬运指令

    Attributes:
        cmd_id: 指令 ID (单调递增)
        cmd_id_dep: 等待 tiu_sync_id >= 此值才可发射, 0=无依赖
        direction: 搬运方向
        src_addr: 源地址
        dst_addr: 目的地址
        data_bytes: 搬运数据量 (bytes)
        elem_bytes: 元素大小 (bytes)
        source_op_id: 关联的 DistributedOp ID
    """
    cmd_id: int
    cmd_id_dep: int
    direction: DMADirection
    src_addr: int
    dst_addr: int
    data_bytes: int
    elem_bytes: int
    source_op_id: str


@dataclass
class SDMACommand:
    """SDMA 搬运指令 (核间通信, GMEM <-> GMEM)

    Attributes:
        cmd_id: 指令 ID (单调递增)
        cmd_id_dep: 依赖的 sync_id (由 dep_engine 指定)
        dep_engine: 依赖哪个引擎的 sync_id ("sdma"/"tiu"/"hau"/"tdma")
        cmd_type: SDMA 命令类型
        src_addr: 源地址
        dst_addr: 目的地址
        data_bytes: 搬运数据量 (bytes)
        elem_bytes: 元素大小 (bytes)
        src_core_id: 源核心 ID
        dst_core_id: 目的核心 ID
        source_op_id: 关联的 DistributedOp ID
    """
    cmd_id: int
    cmd_id_dep: int
    dep_engine: str
    cmd_type: SDMACommandType
    src_addr: int
    dst_addr: int
    data_bytes: int
    elem_bytes: int
    src_core_id: int
    dst_core_id: int
    source_op_id: str


@dataclass
class HAUCommand:
    """HAU 指令 (硬件排序/Top-K)

    Attributes:
        cmd_id: 指令 ID (单调递增)
        cmd_id_dep: 依赖的 sync_id (由 dep_engine 指定)
        dep_engine: 依赖哪个引擎的 sync_id ("tiu"/"tdma")
        op_type: HAU 操作类型
        src_addr: 源地址
        dst_addr: 目的地址
        num_elements: 待排序/选择的元素数量
        top_k: Top-K 的 K 值 (仅 TOP_K 使用)
        data_format: 数据格式 (BF16/FP32)
        msg_action: 与 SDMA 联动动作
        msg_id: 消息 ID (联动用)
        source_op_id: 关联的 DistributedOp ID
    """
    cmd_id: int
    cmd_id_dep: int
    dep_engine: str
    op_type: HAUOpType
    src_addr: int
    dst_addr: int
    num_elements: int
    top_k: int
    data_format: str
    msg_action: HAUMsgAction
    msg_id: int
    source_op_id: str


# ========== 聚合结构 ==========

@dataclass
class CoreInstructions:
    """单核指令序列

    Attributes:
        core_id: 核心 ID
        tiu_cmds: TIU 指令列表
        dma_cmds: GDMA 指令列表
        sdma_cmds: SDMA 指令列表
        hau_cmds: HAU 指令列表
    """
    core_id: int
    tiu_cmds: list[TIUCommand] = field(default_factory=list)
    dma_cmds: list[DMACommand] = field(default_factory=list)
    sdma_cmds: list[SDMACommand] = field(default_factory=list)
    hau_cmds: list[HAUCommand] = field(default_factory=list)


@dataclass
class CommOp:
    """跨芯片通信操作 (空壳, 后续实现)"""
    op_id: str
    src_chip: int = 0
    dst_chip: int = 0
    data_bytes: int = 0


@dataclass
class CoreProgram:
    """多核程序

    L3.g5 的最终输出, 包含所有核心的指令序列及通信调度。

    Attributes:
        cores: 各核心的指令序列
        comm_schedule: 跨芯片通信调度 (空)
        metadata: 元信息 (tiling 参数, 原始 op 信息等)
    """
    cores: list[CoreInstructions] = field(default_factory=list)
    comm_schedule: list[CommOp] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def total_tiu_cmds(self) -> int:
        """TIU 指令总数"""
        return sum(len(c.tiu_cmds) for c in self.cores)

    def total_dma_cmds(self) -> int:
        """DMA 指令总数"""
        return sum(len(c.dma_cmds) for c in self.cores)

    def total_sdma_cmds(self) -> int:
        """SDMA 指令总数"""
        return sum(len(c.sdma_cmds) for c in self.cores)

    def total_hau_cmds(self) -> int:
        """HAU 指令总数"""
        return sum(len(c.hau_cmds) for c in self.cores)
