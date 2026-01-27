"""
模型工具函数

提供模型参数量计算等辅助功能
"""

from typing import Optional
from ..config.types import LLMModelConfig, MoEConfig, MLAConfig


def calculate_model_params(model: LLMModelConfig) -> int:
    """
    计算模型总参数量

    参考模型参数量:
    - LLaMA-7B: 6.74B (tie_word_embeddings=true)
    - LLaMA-70B: 68.98B (tie_word_embeddings=true)
    - Qwen-7B: 7.72B (tie_word_embeddings=false)
    - DeepSeek-V3: 671B

    Args:
        model: 模型配置

    Returns:
        总参数量
    """
    H = model.hidden_size
    L = model.num_layers
    V = model.vocab_size
    I = model.intermediate_size
    num_heads = model.num_attention_heads
    num_kv_heads = model.num_kv_heads

    # 每个头的维度
    head_dim = H // num_heads

    # Embedding 层: token embedding
    embedding_params = V * H

    # LM Head: 如果共享 embedding 权重则不额外计算
    # 大多数现代模型 (LLaMA, Mistral, DeepSeek) 共享权重
    tie_word_embeddings = True  # 默认共享
    lm_head_params = 0 if tie_word_embeddings else H * V

    # Attention 参数量计算
    if model.attention_type == "mla" and model.mla_config:
        # MLA (Multi-head Latent Attention): 5 个投影矩阵
        mla = model.mla_config
        Nh = model.num_attention_heads

        # q_a_proj: H → q_lora_rank
        W_q_a = H * mla.q_lora_rank
        # q_b_proj: q_lora_rank → Nh × (qk_nope_head_dim + qk_rope_head_dim)
        W_q_b = mla.q_lora_rank * Nh * (mla.qk_nope_head_dim + mla.qk_rope_head_dim)
        # kv_a_proj: H → (kv_lora_rank + qk_rope_head_dim)
        W_kv_a = H * (mla.kv_lora_rank + mla.qk_rope_head_dim)
        # kv_b_proj: kv_lora_rank → Nh × (qk_nope_head_dim + v_head_dim)
        W_kv_b = mla.kv_lora_rank * Nh * (mla.qk_nope_head_dim + mla.v_head_dim)
        # o_proj: Nh × v_head_dim → H
        W_o = Nh * mla.v_head_dim * H

        attention_params = W_q_a + W_q_b + W_kv_a + W_kv_b + W_o
    else:
        # 标准 GQA/MHA
        #   - Q: H * H
        #   - K: H * headDim * numKVHeads
        #   - V: H * headDim * numKVHeads
        #   - O: H * H
        q_params = H * H
        k_params = H * head_dim * num_kv_heads
        v_params = H * head_dim * num_kv_heads
        o_params = H * H
        attention_params = q_params + k_params + v_params + o_params

    # FFN (SwiGLU):
    #   - gate: H * I
    #   - up: H * I
    #   - down: I * H
    ffn_params = 3 * H * I

    # MoE: 区分 Dense 层和 MoE 层
    if model.model_type == "moe" and model.moe_config:
        moe = model.moe_config
        num_experts = moe.num_experts
        num_shared_experts = moe.num_shared_experts
        expert_I = moe.expert_intermediate_size if moe.expert_intermediate_size > 0 else I
        first_k_dense = moe.first_k_dense_replace

        # Dense 层 (前 firstKDense 层): 使用标准 FFN
        num_dense_layers = min(first_k_dense, L)
        dense_ffn_params = 3 * H * I * num_dense_layers

        # MoE 层参数
        num_moe_layers = L - num_dense_layers
        # Router: H * num_experts
        # Experts: 3 * H * expert_I * (num_experts + num_shared_experts)
        moe_ffn_params = (3 * H * expert_I * (num_experts + num_shared_experts) + H * num_experts) * num_moe_layers

        # 覆盖 ffn_params（计算平均每层）
        ffn_params = (dense_ffn_params + moe_ffn_params) // L if L > 0 else moe_ffn_params

    # LayerNorm/RMSNorm: 每层 2 个 (attention前 + FFN前)
    # RMSNorm: H (仅 gamma), LayerNorm: 2H (gamma + beta)
    norm_type = model.norm_type if model.norm_type else "rmsnorm"
    layer_norm_params = 2 * H if norm_type == "rmsnorm" else 4 * H

    # 每层总参数
    params_per_layer = attention_params + ffn_params + layer_norm_params

    # Final LayerNorm (模型输出前的最后一个 RMSNorm/LayerNorm)
    final_layer_norm_params = H if norm_type == "rmsnorm" else 2 * H

    # 总参数
    total_params = embedding_params + L * params_per_layer + lm_head_params + final_layer_norm_params

    return total_params


def calculate_params_from_dict(model_dict: dict) -> int:
    """
    从字典配置计算模型参数量

    Args:
        model_dict: 模型配置字典

    Returns:
        总参数量
    """
    # 解析 MLA 配置
    mla_config = None
    if model_dict.get("mla_config"):
        mla_dict = model_dict["mla_config"]
        mla_config = MLAConfig(
            kv_lora_rank=mla_dict.get("kv_lora_rank", 0),
            q_lora_rank=mla_dict.get("q_lora_rank", 0),
            qk_nope_head_dim=mla_dict.get("qk_nope_head_dim", 0),
            qk_rope_head_dim=mla_dict.get("qk_rope_head_dim", 0),
            v_head_dim=mla_dict.get("v_head_dim", 0),
        )

    # 解析 MoE 配置
    moe_config = None
    if model_dict.get("moe_config"):
        moe_dict = model_dict["moe_config"]
        moe_config = MoEConfig(
            num_experts=moe_dict.get("num_experts", 1),
            num_experts_per_tok=moe_dict.get("num_experts_per_tok", 1),
            num_shared_experts=moe_dict.get("num_shared_experts", 0),
            expert_intermediate_size=moe_dict.get("expert_intermediate_size", 0),
            first_k_dense_replace=moe_dict.get("first_k_dense_replace", 0),
        )

    # 构建模型配置
    model = LLMModelConfig(
        model_name=model_dict.get("model_name", "unknown"),
        model_type=model_dict.get("model_type", "dense"),
        hidden_size=model_dict.get("hidden_size", 0),
        num_layers=model_dict.get("num_layers", 0),
        num_attention_heads=model_dict.get("num_attention_heads", 0),
        num_kv_heads=model_dict.get("num_kv_heads", model_dict.get("num_attention_heads", 0)),
        intermediate_size=model_dict.get("intermediate_size", 0),
        vocab_size=model_dict.get("vocab_size", 0),
        dtype=model_dict.get("dtype", "bf16"),
        max_seq_length=model_dict.get("max_seq_length", 4096),
        attention_type=model_dict.get("attention_type", "gqa"),
        norm_type=model_dict.get("norm_type", "rmsnorm"),
        mla_config=mla_config,
        moe_config=moe_config,
    )

    return calculate_model_params(model)


def format_params(params: int) -> str:
    """
    格式化参数量显示

    Args:
        params: 参数量

    Returns:
        格式化字符串，如 "7.0B", "671B", "1.5T"
    """
    if params >= 1e12:
        return f"{params / 1e12:.1f}T"
    elif params >= 1e9:
        return f"{params / 1e9:.1f}B"
    elif params >= 1e6:
        return f"{params / 1e6:.1f}M"
    elif params >= 1e3:
        return f"{params / 1e3:.1f}K"
    else:
        return str(params)
