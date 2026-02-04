"""计算评估器."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tier6.L4_evaluation.evaluators.base import BaseEvaluator
from tier6.L4_evaluation.metrics import BottleneckTag, StepMetrics

if TYPE_CHECKING:
    from tier6.L4_evaluation.registry import CostModel


class ComputeEvaluator(BaseEvaluator):
    """计算 Op 评估器

    支持 matmul/linear/gemm/softmax/layernorm 等计算类 Op。
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
        """评估计算 Op

        输入:
            - op_id, op_type, local_shape, attrs, hardware, cost_model
        输出:
            - StepMetrics
        关键步骤:
            - 估算 FLOPs 和访存量
            - 计算 compute/memory bound 时间
            - 确定瓶颈类型
        """
        # 估算 FLOPs 和访存量
        scale_key = f"flops_scale_{op_type.lower().replace('-', '_')}"
        flops_scale = float(hardware.get(scale_key, hardware.get("flops_scale", 1.0)))
        flops = int(cost_model.estimate_flops(op_type, local_shape) * flops_scale)
        compute_efficiency = float(hardware.get("compute_efficiency", 1.0))
        if compute_efficiency <= 0:
            compute_efficiency = 1.0
        tile_traffic = attrs.get("tile_traffic_bytes")
        if tile_traffic is not None:
            try:
                total_bytes = int(tile_traffic)
                bytes_read = total_bytes
                bytes_write = 0
            except (TypeError, ValueError):
                bytes_read, bytes_write = cost_model.estimate_bytes(op_type, local_shape)
                total_bytes = bytes_read + bytes_write
        else:
            bytes_read, bytes_write = cost_model.estimate_bytes(op_type, local_shape)
            total_bytes = bytes_read + bytes_write

        # 估算计算时间
        if tile_traffic is not None:
            compute_tflops = hardware.get("compute_tflops", 125.0)
            t_compute_bound = flops / (compute_tflops * 1e9) if compute_tflops > 0 else 0.0
            # L3 已完成 tile 选择，这里以计算核时延为主口径；
            # 访存压力通过 bytes 指标进入聚合分析（MBU）。
            t_compute = t_compute_bound / compute_efficiency
        else:
            t_compute = (
                cost_model.estimate_compute(op_type, local_shape, hardware)
                / compute_efficiency
            )

        # 确定瓶颈类型
        compute_tflops = hardware.get("compute_tflops", 125.0)
        memory_bw_gbps = hardware.get("memory_bandwidth_gbps", 2000.0)

        # 计算算术强度 (FLOPs / Bytes)
        arithmetic_intensity = flops / total_bytes if total_bytes > 0 else float("inf")

        # 平衡点 = Peak FLOPS / Peak BW = TFLOPS * 1e12 / (GB/s * 1e9) = TFLOPS * 1e3 / GB/s
        balance_point = (
            compute_tflops * 1e3 / memory_bw_gbps if memory_bw_gbps > 0 else 0
        )

        if arithmetic_intensity > balance_point:
            bottleneck = BottleneckTag.COMPUTE_BOUND
        else:
            bottleneck = BottleneckTag.BW_BOUND

        return StepMetrics(
            op_id=op_id,
            t_compute=t_compute,
            t_comm=0.0,
            t_wait=0.0,
            t_total=t_compute,
            bottleneck_tag=bottleneck,
            flops=flops,
            bytes_read=bytes_read,
            bytes_write=bytes_write,
            meta={
                "evaluator": "compute",
                "op_type": op_type,
                "arithmetic_intensity": arithmetic_intensity,
                "tile_traffic_bytes": attrs.get("tile_traffic_bytes"),
                "tile_lmem_bytes": attrs.get("tile_lmem_bytes"),
                "local_weight_bytes": attrs.get("local_weight_bytes"),
            },
        )
