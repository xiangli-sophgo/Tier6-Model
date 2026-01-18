"""
甘特图数据生成模块

将模拟事件转换为前端可渲染的甘特图数据格式。
"""
from __future__ import annotations

from .types import (
    GanttTask, GanttResource, GanttChartData, GanttTaskType, InferencePhase,
    ParallelismStrategy,
)


# ============================================
# 任务类型颜色映射
# ============================================

TASK_COLORS = {
    # 计算任务 - 绿色系
    GanttTaskType.COMPUTE: "#52c41a",
    GanttTaskType.EMBEDDING: "#73d13d",
    GanttTaskType.LAYERNORM: "#95de64",
    GanttTaskType.ATTENTION_QKV: "#389e0d",
    GanttTaskType.ATTENTION_SCORE: "#52c41a",
    GanttTaskType.ATTENTION_SOFTMAX: "#73d13d",
    GanttTaskType.ATTENTION_OUTPUT: "#95de64",
    GanttTaskType.FFN_GATE: "#237804",
    GanttTaskType.FFN_UP: "#389e0d",
    GanttTaskType.FFN_DOWN: "#52c41a",
    GanttTaskType.LM_HEAD: "#135200",

    # 数据搬运 - 橙色系
    GanttTaskType.PCIE_H2D: "#fa8c16",
    GanttTaskType.PCIE_D2H: "#ffa940",
    GanttTaskType.HBM_WRITE: "#ffc53d",
    GanttTaskType.HBM_READ: "#ffd666",
    GanttTaskType.WEIGHT_LOAD: "#d48806",
    GanttTaskType.KV_CACHE_READ: "#faad14",
    GanttTaskType.KV_CACHE_WRITE: "#ffc53d",

    # 通信 - 蓝/紫色系
    GanttTaskType.TP_COMM: "#1890ff",
    GanttTaskType.PP_COMM: "#722ed1",
    GanttTaskType.EP_COMM: "#eb2f96",

    # SP 通信 - 蓝色系 (序列并行)
    GanttTaskType.SP_ALLGATHER: "#2f54eb",
    GanttTaskType.SP_REDUCE_SCATTER: "#1d39c4",

    # DP 通信 - 深紫色 (数据并行梯度同步)
    GanttTaskType.DP_GRADIENT_SYNC: "#531dab",

    # MLA细粒度 - 青色系 (DeepSeek特有)
    GanttTaskType.RMSNORM_Q_LORA: "#13c2c2",
    GanttTaskType.RMSNORM_KV_LORA: "#36cfc9",
    GanttTaskType.MM_Q_LORA_A: "#5cdbd3",
    GanttTaskType.MM_Q_LORA_B: "#87e8de",
    GanttTaskType.MM_KV_LORA_A: "#b5f5ec",
    GanttTaskType.ATTN_FC: "#08979c",
    GanttTaskType.BMM_QK: "#006d75",
    GanttTaskType.BMM_SV: "#00474f",

    # MoE - 品红/紫色系
    GanttTaskType.MOE_GATE: "#f759ab",
    GanttTaskType.MOE_EXPERT: "#eb2f96",
    GanttTaskType.MOE_SHARED_EXPERT: "#c41d7f",
    GanttTaskType.EP_DISPATCH: "#9254de",
    GanttTaskType.EP_COMBINE: "#722ed1",

    # 其他
    GanttTaskType.BUBBLE: "#ff4d4f",
    GanttTaskType.IDLE: "#d9d9d9",
}


# ============================================
# 任务类型标签映射
# ============================================

TASK_LABELS = {
    GanttTaskType.COMPUTE: "计算",
    GanttTaskType.EMBEDDING: "Embedding",
    GanttTaskType.LAYERNORM: "LayerNorm",
    GanttTaskType.ATTENTION_QKV: "Attn QKV",
    GanttTaskType.ATTENTION_SCORE: "Attn Score",
    GanttTaskType.ATTENTION_SOFTMAX: "Softmax",
    GanttTaskType.ATTENTION_OUTPUT: "Attn Out",
    GanttTaskType.FFN_GATE: "FFN Gate",
    GanttTaskType.FFN_UP: "FFN Up",
    GanttTaskType.FFN_DOWN: "FFN Down",
    GanttTaskType.LM_HEAD: "LM Head",

    GanttTaskType.PCIE_H2D: "PCIe H2D",
    GanttTaskType.PCIE_D2H: "PCIe D2H",
    GanttTaskType.HBM_WRITE: "HBM Write",
    GanttTaskType.HBM_READ: "HBM Read",
    GanttTaskType.WEIGHT_LOAD: "Weight Load",
    GanttTaskType.KV_CACHE_READ: "KV Read",
    GanttTaskType.KV_CACHE_WRITE: "KV Write",

    GanttTaskType.TP_COMM: "TP 通信",
    GanttTaskType.PP_COMM: "PP 通信",
    GanttTaskType.EP_COMM: "EP 通信",

    # SP 通信 (序列并行)
    GanttTaskType.SP_ALLGATHER: "SP AllGather",
    GanttTaskType.SP_REDUCE_SCATTER: "SP ReduceScatter",

    # DP 通信 (数据并行)
    GanttTaskType.DP_GRADIENT_SYNC: "DP 梯度同步",

    # MLA细粒度 (DeepSeek特有)
    GanttTaskType.RMSNORM_Q_LORA: "RMSNorm Q",
    GanttTaskType.RMSNORM_KV_LORA: "RMSNorm KV",
    GanttTaskType.MM_Q_LORA_A: "Q LoRA↓",
    GanttTaskType.MM_Q_LORA_B: "Q LoRA↑",
    GanttTaskType.MM_KV_LORA_A: "KV Compress",
    GanttTaskType.ATTN_FC: "Attn FC",
    GanttTaskType.BMM_QK: "BMM QK",
    GanttTaskType.BMM_SV: "BMM SV",

    # MoE
    GanttTaskType.MOE_GATE: "MoE Gate",
    GanttTaskType.MOE_EXPERT: "Expert FFN",
    GanttTaskType.MOE_SHARED_EXPERT: "Shared Expert",
    GanttTaskType.EP_DISPATCH: "EP Dispatch",
    GanttTaskType.EP_COMBINE: "EP Combine",

    GanttTaskType.BUBBLE: "气泡",
    GanttTaskType.IDLE: "空闲",
}


class GanttChartBuilder:
    """甘特图数据构建器"""

    def __init__(self, parallelism: ParallelismStrategy):
        """
        初始化构建器

        Args:
            parallelism: 并行策略
        """
        self.parallelism = parallelism
        self.tasks: list[GanttTask] = []
        self.resources: list[GanttResource] = []
        self._task_counter = 0

        # 初始化资源行
        self._init_resources()

    def _init_resources(self):
        """初始化资源行"""
        for pp_stage in range(self.parallelism.pp):
            # 计算资源行
            self.resources.append(GanttResource(
                id=f"stage{pp_stage}_compute",
                name=f"PP{pp_stage} 计算",
                pp_stage=pp_stage,
                type="compute",
            ))
            # 网络资源行
            self.resources.append(GanttResource(
                id=f"stage{pp_stage}_network",
                name=f"PP{pp_stage} 网络",
                pp_stage=pp_stage,
                type="network",
            ))

    def add_task(
        self,
        name: str,
        start: float,
        end: float,
        task_type: GanttTaskType,
        phase: InferencePhase,
        chip_id: str,
        pp_stage: int,
        layer_index: int | None = None,
        token_index: int | None = None,
    ) -> GanttTask:
        """
        添加任务

        Args:
            name: 任务名称
            start: 开始时间 (ms)
            end: 结束时间 (ms)
            task_type: 任务类型
            phase: 推理阶段
            chip_id: 芯片ID
            pp_stage: PP阶段
            layer_index: 层索引
            token_index: Token索引

        Returns:
            创建的任务
        """
        self._task_counter += 1

        # 确定资源行
        if task_type in (GanttTaskType.TP_COMM, GanttTaskType.PP_COMM, GanttTaskType.EP_COMM,
                         GanttTaskType.EP_DISPATCH, GanttTaskType.EP_COMBINE,
                         GanttTaskType.SP_ALLGATHER, GanttTaskType.SP_REDUCE_SCATTER,
                         GanttTaskType.DP_GRADIENT_SYNC):
            resource = f"stage{pp_stage}_network"
        else:
            resource = f"stage{pp_stage}_compute"

        task = GanttTask(
            id=f"task_{self._task_counter}",
            name=name,
            resource=resource,
            start=start,
            end=end,
            type=task_type,
            phase=phase,
            chip_id=chip_id,
            pp_stage=pp_stage,
            layer_index=layer_index,
            token_index=token_index,
            color=TASK_COLORS.get(task_type),
        )

        self.tasks.append(task)
        return task

    def add_compute_task(
        self,
        task_type: GanttTaskType,
        start: float,
        duration: float,
        phase: InferencePhase,
        chip_id: str,
        pp_stage: int,
        layer_index: int | None = None,
        token_index: int | None = None,
    ) -> GanttTask:
        """添加计算任务的便捷方法"""
        name = TASK_LABELS.get(task_type, str(task_type.value))
        if layer_index is not None:
            name = f"L{layer_index} {name}"
        if token_index is not None:
            name = f"T{token_index} {name}"

        return self.add_task(
            name=name,
            start=start,
            end=start + duration,
            task_type=task_type,
            phase=phase,
            chip_id=chip_id,
            pp_stage=pp_stage,
            layer_index=layer_index,
            token_index=token_index,
        )

    def add_comm_task(
        self,
        task_type: GanttTaskType,
        start: float,
        duration: float,
        phase: InferencePhase,
        chip_id: str,
        pp_stage: int,
        layer_index: int | None = None,
        token_index: int | None = None,
    ) -> GanttTask:
        """添加通信任务的便捷方法"""
        name = TASK_LABELS.get(task_type, str(task_type.value))
        if layer_index is not None:
            name = f"L{layer_index} {name}"

        return self.add_task(
            name=name,
            start=start,
            end=start + duration,
            task_type=task_type,
            phase=phase,
            chip_id=chip_id,
            pp_stage=pp_stage,
            layer_index=layer_index,
            token_index=token_index,
        )

    def add_bubble(
        self,
        start: float,
        duration: float,
        phase: InferencePhase,
        chip_id: str,
        pp_stage: int,
    ) -> GanttTask:
        """添加气泡任务"""
        return self.add_task(
            name="Bubble",
            start=start,
            end=start + duration,
            task_type=GanttTaskType.BUBBLE,
            phase=phase,
            chip_id=chip_id,
            pp_stage=pp_stage,
        )

    def build(self, phase_transition: float | None = None) -> GanttChartData:
        """
        构建甘特图数据

        Args:
            phase_transition: Prefill/Decode 分界时间点

        Returns:
            甘特图数据
        """
        if not self.tasks:
            return GanttChartData(
                resources=self.resources,
                tasks=[],
                time_range=(0.0, 0.0),
                phase_transition=None,
            )

        # 计算时间范围
        start_time = min(t.start for t in self.tasks)
        end_time = max(t.end for t in self.tasks)

        return GanttChartData(
            resources=self.resources,
            tasks=self.tasks,
            time_range=(start_time, end_time),
            phase_transition=phase_transition,
        )


def convert_to_frontend_format(gantt_data: GanttChartData) -> dict:
    """
    将甘特图数据转换为前端期望的格式

    Args:
        gantt_data: 甘特图数据

    Returns:
        前端格式的字典
    """
    resources = []
    for r in gantt_data.resources:
        resources.append({
            "id": r.id,
            "name": r.name,
            "ppStage": r.pp_stage,
            "type": r.type,
        })

    tasks = []
    for t in gantt_data.tasks:
        tasks.append({
            "id": t.id,
            "name": t.name,
            "resource": t.resource,
            "start": t.start,
            "end": t.end,
            "type": t.type.value,
            "phase": t.phase.value,
            "chipId": t.chip_id,
            "ppStage": t.pp_stage,
            "layerIndex": t.layer_index,
            "tokenIndex": t.token_index,
            "color": t.color,
        })

    return {
        "resources": resources,
        "tasks": tasks,
        "timeRange": {
            "start": gantt_data.time_range[0],
            "end": gantt_data.time_range[1],
        },
        "phaseTransition": gantt_data.phase_transition,
    }
