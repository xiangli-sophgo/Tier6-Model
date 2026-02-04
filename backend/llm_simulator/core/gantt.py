"""
甘特图数据生成模块

将模拟事件转换为前端可渲染的甘特图数据格式。
"""
from __future__ import annotations

from ..config import (
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

    def _generate_descriptive_name(
        self,
        task_type: GanttTaskType,
        layer_index: int | None,
        token_index: int | None,
        is_comm: bool = False
    ) -> str:
        """
        生成描述性的任务名称

        格式示例：
        - "Layer_0_Attention_Q_Proj"
        - "Layer_5_FFN_Gate"
        - "Layer_12_MLA_Q_A_Proj"
        - "Layer_3_Attention_TP_AllReduce"
        - "Embedding_Input"
        - "LM_Head_Output"

        Args:
            task_type: 任务类型
            layer_index: 层索引
            token_index: Token索引
            is_comm: 是否为通信任务

        Returns:
            描述性名称字符串
        """
        # 映射任务类型到描述性名称组件
        type_name_map = {
            # 计算任务
            GanttTaskType.COMPUTE: "Compute",
            GanttTaskType.EMBEDDING: "Embedding",
            GanttTaskType.LAYERNORM: "LayerNorm",
            GanttTaskType.ATTENTION_QKV: "Attention_QKV",
            GanttTaskType.ATTENTION_SCORE: "Attention_Score",
            GanttTaskType.ATTENTION_SOFTMAX: "Attention_Softmax",
            GanttTaskType.ATTENTION_OUTPUT: "Attention_Output",
            GanttTaskType.FFN_GATE: "FFN_Gate",
            GanttTaskType.FFN_UP: "FFN_Up",
            GanttTaskType.FFN_DOWN: "FFN_Down",
            GanttTaskType.LM_HEAD: "LM_Head",

            # MLA 细粒度
            GanttTaskType.RMSNORM_Q_LORA: "MLA_RMSNorm_Q",
            GanttTaskType.RMSNORM_KV_LORA: "MLA_RMSNorm_KV",
            GanttTaskType.MM_Q_LORA_A: "MLA_Q_LoRA_A",
            GanttTaskType.MM_Q_LORA_B: "MLA_Q_LoRA_B",
            GanttTaskType.MM_KV_LORA_A: "MLA_KV_Compress",
            GanttTaskType.ATTN_FC: "MLA_Attn_FC",
            GanttTaskType.BMM_QK: "MLA_BMM_QK",
            GanttTaskType.BMM_SV: "MLA_BMM_SV",

            # MoE
            GanttTaskType.MOE_GATE: "MoE_Gate",
            GanttTaskType.MOE_EXPERT: "MoE_Expert",
            GanttTaskType.MOE_SHARED_EXPERT: "MoE_Shared_Expert",

            # 通信任务
            GanttTaskType.TP_COMM: "TP_AllReduce",
            GanttTaskType.PP_COMM: "PP_P2P",
            GanttTaskType.EP_COMM: "EP_AllToAll",
            GanttTaskType.SP_ALLGATHER: "SP_AllGather",
            GanttTaskType.SP_REDUCE_SCATTER: "SP_ReduceScatter",
            GanttTaskType.DP_GRADIENT_SYNC: "DP_GradientSync",
            GanttTaskType.EP_DISPATCH: "EP_Dispatch",
            GanttTaskType.EP_COMBINE: "EP_Combine",

            # 数据搬运
            GanttTaskType.PCIE_H2D: "PCIe_H2D",
            GanttTaskType.PCIE_D2H: "PCIe_D2H",
            GanttTaskType.HBM_WRITE: "HBM_Write",
            GanttTaskType.HBM_READ: "HBM_Read",
            GanttTaskType.WEIGHT_LOAD: "Weight_Load",
            GanttTaskType.KV_CACHE_READ: "KV_Cache_Read",
            GanttTaskType.KV_CACHE_WRITE: "KV_Cache_Write",

            # 其他
            GanttTaskType.BUBBLE: "Bubble",
            GanttTaskType.IDLE: "Idle",
        }

        type_name = type_name_map.get(task_type, str(task_type.value))

        # 构建完整名称
        if layer_index is not None:
            # 带层索引的任务
            name = f"Layer_{layer_index}_{type_name}"
        elif task_type == GanttTaskType.EMBEDDING:
            # Embedding 任务
            name = "Embedding_Input"
        elif task_type == GanttTaskType.LM_HEAD:
            # LM Head 任务
            name = "LM_Head_Output"
        else:
            # 其他任务（不带层索引）
            name = type_name

        # 添加 token 索引（如果有）
        if token_index is not None:
            name = f"Token_{token_index}_{name}"

        return name

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
        **extra_fields,  # 新增：接收额外的详细信息字段
    ) -> GanttTask:
        """
        添加任务

        Args:
            name: 任务名称
            start: 开始时间 (us, 微秒)
            end: 结束时间 (us, 微秒)
            task_type: 任务类型
            phase: 推理阶段
            chip_id: 芯片ID
            pp_stage: PP阶段
            layer_index: 层索引
            token_index: Token索引
            **extra_fields: 额外的详细信息字段（如 flops, best_tile, arch_utilization 等）

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
            **extra_fields,  # 传入额外字段
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
        **extra_fields,  # 新增：接收额外的详细信息字段
    ) -> GanttTask:
        """添加计算任务的便捷方法

        Args:
            task_type: 任务类型
            start: 开始时间 (us)
            duration: 持续时间 (us)
            phase: 推理阶段
            chip_id: 芯片ID
            pp_stage: PP阶段
            layer_index: 层索引
            token_index: Token索引
            **extra_fields: 额外的详细信息字段（如 flops, best_tile 等）
        """
        # 生成描述性名称
        name = self._generate_descriptive_name(task_type, layer_index, token_index, is_comm=False)

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
            **extra_fields,  # 传递额外字段
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
        **extra_fields,  # 新增：接收额外的详细信息字段
    ) -> GanttTask:
        """添加通信任务的便捷方法

        Args:
            task_type: 任务类型
            start: 开始时间 (us)
            duration: 持续时间 (us)
            phase: 推理阶段
            chip_id: 芯片ID
            pp_stage: PP阶段
            layer_index: 层索引
            token_index: Token索引
            **extra_fields: 额外的详细信息字段（如 comm_size_bytes, comm_algorithm 等）
        """
        # 生成描述性名称
        name = self._generate_descriptive_name(task_type, layer_index, token_index, is_comm=True)

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
            **extra_fields,  # 传递额外字段
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
            "layer": t.layer_index,
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
