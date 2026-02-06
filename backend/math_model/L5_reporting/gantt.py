"""甘特图数据生成模块

将评估结果转换为前端可渲染的甘特图数据格式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from math_model.L4_evaluation.metrics import EngineResult, StepMetrics
    from math_model.L3_mapping.plan import ExecPlan


class GanttTaskType(str, Enum):
    """甘特图任务类型"""

    # 计算任务
    COMPUTE = "compute"
    EMBEDDING = "embedding"
    LAYERNORM = "layernorm"
    ATTENTION = "attention"
    ATTENTION_QKV = "attention_qkv"
    ATTENTION_SCORE = "attention_score"
    ATTENTION_SOFTMAX = "attention_softmax"
    ATTENTION_OUTPUT = "attention_output"
    FFN = "ffn"
    FFN_GATE = "ffn_gate"
    FFN_UP = "ffn_up"
    FFN_DOWN = "ffn_down"
    LM_HEAD = "lm_head"

    # MLA 细粒度
    MLA_Q_LORA = "mla_q_lora"
    MLA_KV_LORA = "mla_kv_lora"
    MLA_ATTN_FC = "mla_attn_fc"

    # MoE
    MOE_GATE = "moe_gate"
    MOE_EXPERT = "moe_expert"
    MOE_SHARED_EXPERT = "moe_shared_expert"

    # 通信
    TP_COMM = "tp_comm"
    PP_COMM = "pp_comm"
    EP_COMM = "ep_comm"
    DP_COMM = "dp_comm"
    ALLREDUCE = "allreduce"
    ALLGATHER = "allgather"
    ALLTOALL = "alltoall"
    P2P = "p2p"

    # 内存
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    WEIGHT_LOAD = "weight_load"
    KV_CACHE = "kv_cache"

    # 其他
    BUBBLE = "bubble"
    IDLE = "idle"


class InferencePhase(str, Enum):
    """推理阶段"""

    PREFILL = "prefill"
    DECODE = "decode"


# 任务类型颜色映射
TASK_COLORS = {
    # 计算 - 绿色系
    GanttTaskType.COMPUTE: "#52c41a",
    GanttTaskType.EMBEDDING: "#73d13d",
    GanttTaskType.LAYERNORM: "#95de64",
    GanttTaskType.ATTENTION: "#389e0d",
    GanttTaskType.ATTENTION_QKV: "#389e0d",
    GanttTaskType.ATTENTION_SCORE: "#52c41a",
    GanttTaskType.ATTENTION_SOFTMAX: "#73d13d",
    GanttTaskType.ATTENTION_OUTPUT: "#95de64",
    GanttTaskType.FFN: "#237804",
    GanttTaskType.FFN_GATE: "#237804",
    GanttTaskType.FFN_UP: "#389e0d",
    GanttTaskType.FFN_DOWN: "#52c41a",
    GanttTaskType.LM_HEAD: "#135200",
    # MLA - 青色系
    GanttTaskType.MLA_Q_LORA: "#13c2c2",
    GanttTaskType.MLA_KV_LORA: "#36cfc9",
    GanttTaskType.MLA_ATTN_FC: "#08979c",
    # MoE - 品红色系
    GanttTaskType.MOE_GATE: "#f759ab",
    GanttTaskType.MOE_EXPERT: "#eb2f96",
    GanttTaskType.MOE_SHARED_EXPERT: "#c41d7f",
    # 通信 - 蓝/紫色系
    GanttTaskType.TP_COMM: "#1890ff",
    GanttTaskType.PP_COMM: "#722ed1",
    GanttTaskType.EP_COMM: "#eb2f96",
    GanttTaskType.DP_COMM: "#531dab",
    GanttTaskType.ALLREDUCE: "#1890ff",
    GanttTaskType.ALLGATHER: "#2f54eb",
    GanttTaskType.ALLTOALL: "#9254de",
    GanttTaskType.P2P: "#722ed1",
    # 内存 - 橙色系
    GanttTaskType.MEMORY_READ: "#ffd666",
    GanttTaskType.MEMORY_WRITE: "#ffc53d",
    GanttTaskType.WEIGHT_LOAD: "#d48806",
    GanttTaskType.KV_CACHE: "#faad14",
    # 其他
    GanttTaskType.BUBBLE: "#ff4d4f",
    GanttTaskType.IDLE: "#d9d9d9",
}


@dataclass
class GanttTask:
    """甘特图任务

    Attributes:
        id: 任务 ID
        name: 任务名称
        resource: 资源行 ID
        start: 开始时间 (us)
        end: 结束时间 (us)
        task_type: 任务类型
        phase: 推理阶段
        device_id: 设备 ID
        pp_stage: PP 阶段
        layer_index: 层索引
        token_index: Token 索引
        color: 颜色
        attrs: 扩展属性
    """

    id: str
    name: str
    resource: str
    start: float
    end: float
    task_type: GanttTaskType
    phase: InferencePhase
    device_id: str = ""
    pp_stage: int = 0
    layer_index: int | None = None
    token_index: int | None = None
    color: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """持续时间 (us)"""
        return self.end - self.start


@dataclass
class GanttResource:
    """甘特图资源行

    Attributes:
        id: 资源 ID
        name: 资源名称
        pp_stage: PP 阶段
        resource_type: 资源类型 (compute/network)
    """

    id: str
    name: str
    pp_stage: int = 0
    resource_type: str = "compute"


@dataclass
class GanttChartData:
    """甘特图数据

    Attributes:
        resources: 资源行列表
        tasks: 任务列表
        time_range: 时间范围 (start, end)
        phase_transition: Prefill/Decode 分界点
    """

    resources: list[GanttResource] = field(default_factory=list)
    tasks: list[GanttTask] = field(default_factory=list)
    time_range: tuple[float, float] = (0.0, 0.0)
    phase_transition: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为前端格式"""
        return {
            "resources": [
                {
                    "id": r.id,
                    "name": r.name,
                    "ppStage": r.pp_stage,
                    "type": r.resource_type,
                }
                for r in self.resources
            ],
            "tasks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "resource": t.resource,
                    "start": t.start,
                    "end": t.end,
                    "type": t.task_type.value,
                    "phase": t.phase.value,
                    "chipId": t.device_id,
                    "ppStage": t.pp_stage,
                    "layer": t.layer_index,
                    "tokenIndex": t.token_index,
                    "color": t.color or TASK_COLORS.get(t.task_type, "#d9d9d9"),
                }
                for t in self.tasks
            ],
            "timeRange": {
                "start": self.time_range[0],
                "end": self.time_range[1],
            },
            "phaseTransition": self.phase_transition,
        }


class GanttChartBuilder:
    """甘特图构建器"""

    def __init__(self, pp_stages: int = 1) -> None:
        """初始化

        Args:
            pp_stages: PP 阶段数
        """
        self.pp_stages = pp_stages
        self.tasks: list[GanttTask] = []
        self.resources: list[GanttResource] = []
        self._task_counter = 0
        self._init_resources()

    def _init_resources(self) -> None:
        """初始化资源行"""
        for stage in range(self.pp_stages):
            self.resources.append(
                GanttResource(
                    id=f"stage{stage}_compute",
                    name=f"PP{stage} Compute",
                    pp_stage=stage,
                    resource_type="compute",
                )
            )
            self.resources.append(
                GanttResource(
                    id=f"stage{stage}_network",
                    name=f"PP{stage} Network",
                    pp_stage=stage,
                    resource_type="network",
                )
            )

    def _next_task_id(self) -> str:
        """生成唯一任务 ID"""
        self._task_counter += 1
        return f"task_{self._task_counter}"

    def add_task(
        self,
        name: str,
        start_us: float,
        end_us: float,
        task_type: GanttTaskType,
        phase: InferencePhase,
        device_id: str = "",
        pp_stage: int = 0,
        layer_index: int | None = None,
        token_index: int | None = None,
        **attrs: Any,
    ) -> GanttTask:
        """添加任务

        Args:
            name: 任务名称
            start_us: 开始时间 (us)
            end_us: 结束时间 (us)
            task_type: 任务类型
            phase: 推理阶段
            device_id: 设备 ID
            pp_stage: PP 阶段
            layer_index: 层索引
            token_index: Token 索引
            **attrs: 扩展属性

        Returns:
            GanttTask: 添加的任务
        """
        # 确定资源行
        if task_type in (
            GanttTaskType.TP_COMM,
            GanttTaskType.PP_COMM,
            GanttTaskType.EP_COMM,
            GanttTaskType.DP_COMM,
            GanttTaskType.ALLREDUCE,
            GanttTaskType.ALLGATHER,
            GanttTaskType.ALLTOALL,
            GanttTaskType.P2P,
        ):
            resource = f"stage{pp_stage}_network"
        else:
            resource = f"stage{pp_stage}_compute"

        task = GanttTask(
            id=self._next_task_id(),
            name=name,
            resource=resource,
            start=start_us,
            end=end_us,
            task_type=task_type,
            phase=phase,
            device_id=device_id,
            pp_stage=pp_stage,
            layer_index=layer_index,
            token_index=token_index,
            color=TASK_COLORS.get(task_type),
            attrs=attrs,
        )
        self.tasks.append(task)
        return task

    def build(self, phase_transition: float | None = None) -> GanttChartData:
        """构建甘特图数据

        Args:
            phase_transition: Prefill/Decode 分界点

        Returns:
            GanttChartData: 甘特图数据
        """
        if not self.tasks:
            return GanttChartData(
                resources=self.resources,
                tasks=[],
                time_range=(0.0, 0.0),
            )

        start_time = min(t.start for t in self.tasks)
        end_time = max(t.end for t in self.tasks)

        return GanttChartData(
            resources=self.resources,
            tasks=self.tasks,
            time_range=(start_time, end_time),
            phase_transition=phase_transition,
        )


def build_gantt_from_engine_result(
    result: "EngineResult",
    pp_stages: int = 1,
) -> GanttChartData:
    """从 EngineResult 构建甘特图

    Args:
        result: 评估结果
        pp_stages: PP 阶段数

    Returns:
        GanttChartData: 甘特图数据
    """
    builder = GanttChartBuilder(pp_stages=pp_stages)
    current_time = 0.0

    # Prefill 阶段
    for step in result.prefill_steps:
        task_type = _infer_task_type(step.name)
        duration_us = step.total_ns / 1000  # ns -> us

        builder.add_task(
            name=step.name,
            start_us=current_time,
            end_us=current_time + duration_us,
            task_type=task_type,
            phase=InferencePhase.PREFILL,
            flops=step.flops,
            bytes_accessed=step.bytes_accessed,
        )
        current_time += duration_us

    phase_transition = current_time

    # Decode 阶段
    for step in result.decode_steps:
        task_type = _infer_task_type(step.name)
        duration_us = step.total_ns / 1000

        builder.add_task(
            name=step.name,
            start_us=current_time,
            end_us=current_time + duration_us,
            task_type=task_type,
            phase=InferencePhase.DECODE,
            flops=step.flops,
            bytes_accessed=step.bytes_accessed,
        )
        current_time += duration_us

    return builder.build(phase_transition=phase_transition)


def _infer_task_type(name: str) -> GanttTaskType:
    """从任务名称推断任务类型

    Args:
        name: 任务名称

    Returns:
        GanttTaskType: 任务类型
    """
    name_lower = name.lower()

    if "allreduce" in name_lower:
        return GanttTaskType.ALLREDUCE
    if "allgather" in name_lower:
        return GanttTaskType.ALLGATHER
    if "alltoall" in name_lower:
        return GanttTaskType.ALLTOALL
    if "p2p" in name_lower:
        return GanttTaskType.P2P
    if "tp_comm" in name_lower or "tp comm" in name_lower:
        return GanttTaskType.TP_COMM
    if "pp_comm" in name_lower or "pp comm" in name_lower:
        return GanttTaskType.PP_COMM
    if "attention" in name_lower or "attn" in name_lower:
        return GanttTaskType.ATTENTION
    if "ffn" in name_lower or "mlp" in name_lower:
        return GanttTaskType.FFN
    if "embedding" in name_lower:
        return GanttTaskType.EMBEDDING
    if "layernorm" in name_lower or "rmsnorm" in name_lower:
        return GanttTaskType.LAYERNORM
    if "moe" in name_lower:
        if "gate" in name_lower:
            return GanttTaskType.MOE_GATE
        if "shared" in name_lower:
            return GanttTaskType.MOE_SHARED_EXPERT
        return GanttTaskType.MOE_EXPERT
    if "mla" in name_lower:
        if "q_lora" in name_lower:
            return GanttTaskType.MLA_Q_LORA
        if "kv_lora" in name_lower:
            return GanttTaskType.MLA_KV_LORA
        return GanttTaskType.MLA_ATTN_FC
    if "lm_head" in name_lower:
        return GanttTaskType.LM_HEAD

    return GanttTaskType.COMPUTE
