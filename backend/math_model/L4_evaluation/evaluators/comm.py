"""通信评估器."""

from __future__ import annotations

from typing import TYPE_CHECKING

from math_model.L4_evaluation.cost_models import (
    CommArchSpec,
    CommProtocolCostModel,
    CommProtocolParams,
)
from math_model.L4_evaluation.evaluators.base import BaseEvaluator
from math_model.L4_evaluation.metrics import BottleneckTag, StepMetrics

if TYPE_CHECKING:
    from math_model.L4_evaluation.registry import CostModel


class CommEvaluator(BaseEvaluator):
    """通信 Op 评估器

    支持 allreduce/allgather/all2all/p2p 等通信类 Op。
    """

    def evaluate(
        self,
        op_id: str,
        op_type: str,
        local_shape: dict[str, int],
        attrs: dict[str, str],
        hardware: dict[str, float],
        cost_model: "CostModel",
    ) -> StepMetrics:
        """评估通信 Op

        输入:
            - op_id, op_type, local_shape, attrs, hardware, cost_model
        输出:
            - StepMetrics
        关键步骤:
            - 从 attrs 获取 comm_bytes, path_key, participants
            - 调用 cost_model.estimate_comm
        """
        # 从 attrs 获取通信参数
        comm_bytes = int(attrs.get("comm_bytes", "0"))
        path_key = attrs.get("path_key", "inter_board")
        participants = int(attrs.get("participants", "2"))
        comm_protocol = int(attrs.get("comm_protocol", "1"))
        tp = int(attrs.get("tp", str(participants)))
        ep = int(attrs.get("ep", "1"))
        moe_tp = int(attrs.get("moe_tp", "1"))
        bs = int(attrs.get("bs", "0"))
        is_prefill = attrs.get("is_prefill", "False").lower() == "true"
        comm_type = attrs.get("comm_type", op_type).lower()
        reason = attrs.get("reason", "").lower()

        # 构建通信硬件口径（使用默认值是合理的，通信带宽通常有典型值）
        intra_bw = hardware.get("intra_board_bw_gbps", 400.0) * 1e9  # 默认 400 GB/s (NVLink)
        inter_bw = hardware.get("inter_board_bw_gbps", 200.0) * 1e9  # 默认 200 GB/s
        if path_key == "inter_node":
            inter_bw = hardware.get("inter_node_bw_gbps", 100.0) * 1e9  # 默认 100 GB/s
        arch = CommArchSpec(intra_bw=intra_bw, inter_bw=inter_bw)
        params = CommProtocolParams.from_mapping(hardware)
        model = CommProtocolCostModel(arch=arch, params=params)

        # 按 comm_type 路由（接近 DS_TPU 口径）
        if comm_type == "allreduce":
            latency_us, comm_size = model.allreduce(tp, comm_bytes, comm_protocol)
        elif comm_type == "allgather":
            latency_us, comm_size = model.allgather(tp, comm_bytes, comm_protocol)
        elif comm_type in {"reducescatter", "reduce_scatter"}:
            latency_us, comm_size = model.reducescatter(tp, comm_bytes, comm_protocol)
        elif comm_type == "all2all":
            if "dispatch" in reason:
                latency_us, comm_size = model.dispatch(
                    moe_tp, ep, comm_bytes, bs, comm_protocol, is_prefill
                )
            elif "combine" in reason:
                latency_us, comm_size = model.combine(
                    moe_tp, ep, comm_bytes, bs, comm_protocol, is_prefill
                )
            else:
                latency_us, comm_size = model.allgather(tp, comm_bytes, comm_protocol)
        else:
            latency_us, comm_size = model.allreduce(tp, comm_bytes, comm_protocol)

        # us -> ms
        t_comm = latency_us / 1000.0

        # 通信操作默认是带宽瓶颈或延迟瓶颈
        # 小数据量更可能是延迟瓶颈，大数据量更可能是带宽瓶颈
        latency_threshold_bytes = 1024 * 1024  # 1MB
        if comm_bytes < latency_threshold_bytes:
            bottleneck = BottleneckTag.LATENCY_BOUND
        else:
            bottleneck = BottleneckTag.BW_BOUND

        return StepMetrics(
            op_id=op_id,
            t_compute=0.0,
            t_comm=t_comm,
            t_wait=0.0,
            t_total=t_comm,
            bottleneck_tag=bottleneck,
            flops=0,
            bytes_read=comm_bytes,
            bytes_write=comm_bytes,
            meta={
                "evaluator": "comm",
                "op_type": op_type,
                "path_key": path_key,
                "participants": participants,
                "comm_protocol": comm_protocol,
                "tp": tp,
                "ep": ep,
                "moe_tp": moe_tp,
                "bs": bs,
                "reason": reason,
            },
        )
