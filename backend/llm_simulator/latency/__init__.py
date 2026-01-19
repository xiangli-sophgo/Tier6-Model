"""
延迟计算模块

统一导出所有延迟计算函数
重构自原 latency.py，使用精确 GEMM 评估器
"""

# 核心
from .core import (
    init_evaluators,
    get_arch,
    get_evaluator,
    get_fa2_evaluator,
    calc_gemm_latency,
    calc_gemm_with_details,
    calc_memory_latency,
    calc_fa2_latency,
    calc_fa2_with_details,
)

# Attention
from .attention import (
    calc_attention_qkv_latency,
    calc_attention_score_latency,
    calc_attention_softmax_latency,
    calc_attention_sv_latency,
    calc_attention_output_proj_latency,
    calc_attention_output_latency,
)

# FFN
from .ffn import (
    calc_ffn_gate_latency,
    calc_ffn_up_latency,
    calc_ffn_down_latency,
    calc_ffn_activation_latency,
    calc_layernorm_latency,
    calc_residual_add_latency,
)

# MoE
from .moe import (
    get_max_expert,
    get_max_expert_float,
    calc_moe_gate_latency,
    calc_moe_expert_ffn_latency,
    calc_moe_shared_expert_latency,
    calc_moe_alltoall_latency,
    calc_ep_dispatch_latency,
    calc_ep_combine_latency,
    is_moe_layer,
)

# MLA
from .mla import (
    calc_mla_q_projection_latency,
    calc_mla_kv_compression_latency,
    calc_mla_kv_decompression_latency,
    calc_mla_attention_score_latency,
    calc_mla_decode_attention_latency,
    calc_mla_output_projection_latency,
    calc_mla_kv_cache_read_latency,
    calc_mla_kv_cache_write_latency,
    calc_mla_output_latency,
    # MLA 细粒度
    calc_rmsnorm_q_lora_latency,
    calc_rmsnorm_kv_lora_latency,
    calc_mm_q_lora_a_latency,
    calc_mm_q_lora_b_latency,
    calc_mm_kv_lora_a_latency,
    calc_bmm_qk_latency,
    calc_bmm_sv_latency,
    calc_attn_fc_latency,
)

# Memory
from .memory import (
    calc_hbm_read_latency,
    calc_hbm_write_latency,
    calc_weight_load_latency,
    calc_kv_cache_read_latency,
    calc_kv_cache_write_latency,
    calc_embedding_latency,
    calc_lm_head_latency,
    calc_pcie_h2d_latency,
    calc_pcie_d2h_latency,
    calc_activation_memory_latency,
)

# Communication
from .communication import (
    calc_tp_allreduce_latency,
    calc_pp_p2p_latency,
    calc_ep_alltoall_latency,
    calc_sp_allgather_latency,
    calc_sp_reduce_scatter_latency,
    calc_dp_allreduce_latency,
    calc_attention_allreduce_latency,
    calc_ffn_allreduce_latency,
    calc_sp_comm_volume_gb,
    calc_ep_tp_combined_latency,
    calc_dp_gradient_sync_latency,
)

# Fusion
from .fusion import (
    calc_fused_layernorm_qkv_latency,
    calc_fused_ffn_gate_up_latency,
    calc_single_layer_latency_fused,
    OVERLAP_COEFFICIENTS,
)

__all__ = [
    # Core
    'init_evaluators',
    'get_arch',
    'get_evaluator',
    'get_fa2_evaluator',
    'calc_gemm_latency',
    'calc_gemm_with_details',
    'calc_memory_latency',
    'calc_fa2_latency',
    'calc_fa2_with_details',
    # Attention
    'calc_attention_qkv_latency',
    'calc_attention_score_latency',
    'calc_attention_softmax_latency',
    'calc_attention_sv_latency',
    'calc_attention_output_proj_latency',
    'calc_attention_output_latency',
    # FFN
    'calc_ffn_gate_latency',
    'calc_ffn_up_latency',
    'calc_ffn_down_latency',
    'calc_ffn_activation_latency',
    'calc_layernorm_latency',
    'calc_residual_add_latency',
    # MoE
    'get_max_expert',
    'get_max_expert_float',
    'calc_moe_gate_latency',
    'calc_moe_expert_ffn_latency',
    'calc_moe_shared_expert_latency',
    'calc_moe_alltoall_latency',
    'calc_ep_dispatch_latency',
    'calc_ep_combine_latency',
    'is_moe_layer',
    # MLA
    'calc_mla_q_projection_latency',
    'calc_mla_kv_compression_latency',
    'calc_mla_kv_decompression_latency',
    'calc_mla_attention_score_latency',
    'calc_mla_decode_attention_latency',
    'calc_mla_output_projection_latency',
    'calc_mla_kv_cache_read_latency',
    'calc_mla_kv_cache_write_latency',
    'calc_mla_output_latency',
    'calc_rmsnorm_q_lora_latency',
    'calc_rmsnorm_kv_lora_latency',
    'calc_mm_q_lora_a_latency',
    'calc_mm_q_lora_b_latency',
    'calc_mm_kv_lora_a_latency',
    'calc_bmm_qk_latency',
    'calc_bmm_sv_latency',
    'calc_attn_fc_latency',
    # Memory
    'calc_hbm_read_latency',
    'calc_hbm_write_latency',
    'calc_weight_load_latency',
    'calc_kv_cache_read_latency',
    'calc_kv_cache_write_latency',
    'calc_embedding_latency',
    'calc_lm_head_latency',
    'calc_pcie_h2d_latency',
    'calc_pcie_d2h_latency',
    'calc_activation_memory_latency',
    # Communication
    'calc_tp_allreduce_latency',
    'calc_pp_p2p_latency',
    'calc_ep_alltoall_latency',
    'calc_sp_allgather_latency',
    'calc_sp_reduce_scatter_latency',
    'calc_dp_allreduce_latency',
    'calc_attention_allreduce_latency',
    'calc_ffn_allreduce_latency',
    'calc_sp_comm_volume_gb',
    'calc_ep_tp_combined_latency',
    'calc_dp_gradient_sync_latency',
    # Fusion
    'calc_fused_layernorm_qkv_latency',
    'calc_fused_ffn_gate_up_latency',
    'calc_single_layer_latency_fused',
    'OVERLAP_COEFFICIENTS',
]
