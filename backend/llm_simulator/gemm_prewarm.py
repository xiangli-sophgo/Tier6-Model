"""
GEMM 评估器离线预调优

在模拟器初始化时预先评估常见的 GEMM 形状，避免运行时重复搜索。
"""

import logging
from typing import List, Tuple, Optional
from .evaluators import GEMMEvaluator

logger = logging.getLogger(__name__)


def generate_transformer_gemm_shapes(
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    batch_sizes: List[int],
    seq_lengths: List[int],
    mla_config: Optional[dict] = None,
    moe_config: Optional[dict] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    生成 Transformer 中所有可能的 GEMM 形状

    Args:
        hidden_size: 隐藏层大小
        intermediate_size: FFN 中间层大小
        num_attention_heads: 注意力头数量
        num_kv_heads: KV 头数量（GQA）
        batch_sizes: 批次大小列表（通常 [1, 2, 4, 8, ...]）
        seq_lengths: 序列长度列表（Prefill: [128, 256, 512, 1024, 2048], Decode: [1]）
        mla_config: MLA 配置（可选）
        moe_config: MoE 配置（可选）

    Returns:
        List of (G, M, K, N) 元组
    """
    shapes = []
    head_dim = hidden_size // num_attention_heads

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            M = batch_size * seq_len

            # ========== 标准 Attention ==========
            if not mla_config:
                # QKV 投影: [M, hidden] × [hidden, 3*hidden] (for Q, K, V together)
                # 或者分开: Q=[M, hidden]×[hidden, hidden], K/V=[M, hidden]×[hidden, kv_hidden]
                shapes.append((1, M, hidden_size, hidden_size))  # Q projection
                shapes.append((1, M, hidden_size, num_kv_heads * head_dim))  # K projection (GQA)
                shapes.append((1, M, hidden_size, num_kv_heads * head_dim))  # V projection (GQA)

                # Attention Score: [batch, num_heads, seq_len, head_dim] × [batch, num_heads, head_dim, seq_len]
                # → Batched GEMM: G=batch*num_heads, M=seq_len, K=head_dim, N=seq_len
                # 这个在 FlashAttention 中通常融合，不单独计算

                # Attention Output: [M, hidden] × [hidden, hidden]
                shapes.append((1, M, hidden_size, hidden_size))

            # ========== MLA (DeepSeek V3) ==========
            else:
                kv_lora_rank = mla_config.get("kv_lora_rank", 512)
                q_lora_rank = mla_config.get("q_lora_rank", 1536)
                qk_nope_head_dim = mla_config.get("qk_nope_head_dim", 128)
                qk_rope_head_dim = mla_config.get("qk_rope_head_dim", 64)
                v_head_dim = mla_config.get("v_head_dim", 128)

                # Q path: W_DQ, W_UQ, W_QR
                shapes.append((1, M, hidden_size, q_lora_rank))  # W_DQ
                shapes.append((1, M, q_lora_rank, num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim)))  # W_UQ

                # KV path: W_DKV, W_UK, W_UV
                shapes.append((1, M, hidden_size, kv_lora_rank))  # W_DKV
                shapes.append((1, M, kv_lora_rank, num_attention_heads * qk_nope_head_dim))  # W_UK
                shapes.append((1, M, kv_lora_rank, num_attention_heads * v_head_dim))  # W_UV

                # Output: W_O
                shapes.append((1, M, num_attention_heads * v_head_dim, hidden_size))

            # ========== FFN ==========
            if not moe_config:
                # Standard FFN: gate, up, down
                shapes.append((1, M, hidden_size, intermediate_size))  # gate
                shapes.append((1, M, hidden_size, intermediate_size))  # up
                shapes.append((1, M, intermediate_size, hidden_size))  # down
            else:
                # MoE FFN
                num_experts = moe_config.get("num_experts", 64)
                expert_intermediate = moe_config.get("expert_intermediate_size", intermediate_size)

                # Router: [M, hidden] × [hidden, num_experts]
                shapes.append((1, M, hidden_size, num_experts))

                # Expert FFN (每个 expert 的形状相同，只计算一次)
                shapes.append((1, M, hidden_size, expert_intermediate))  # expert gate
                shapes.append((1, M, hidden_size, expert_intermediate))  # expert up
                shapes.append((1, M, expert_intermediate, hidden_size))  # expert down

                # Shared expert (如果有)
                if moe_config.get("num_shared_experts", 0) > 0:
                    shapes.append((1, M, hidden_size, intermediate_size))
                    shapes.append((1, M, intermediate_size, hidden_size))

    # 去重
    shapes = list(set(shapes))
    return shapes


def prewarm_gemm_evaluator(
    evaluator: GEMMEvaluator,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    batch_size: int,
    input_seq_length: int,
    output_seq_length: int,
    mla_config: Optional[dict] = None,
    moe_config: Optional[dict] = None,
) -> int:
    """
    预热 GEMM 评估器，预先评估常见的 GEMM 形状

    Args:
        evaluator: GEMM 评估器实例
        ... 其他参数与 generate_transformer_gemm_shapes 相同

    Returns:
        预热的 GEMM 形状数量
    """
    import time

    logger.info("🔥 开始 GEMM 评估器预热...")
    start = time.time()

    # 生成常见的批次大小和序列长度组合
    batch_sizes = [batch_size]  # 只预热当前批次大小

    # Prefill 和 Decode 的序列长度
    seq_lengths = [
        input_seq_length,  # Prefill
        1,                 # Decode (每次生成 1 个 token)
    ]

    # 生成所有 GEMM 形状
    shapes = generate_transformer_gemm_shapes(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        batch_sizes=batch_sizes,
        seq_lengths=seq_lengths,
        mla_config=mla_config,
        moe_config=moe_config,
    )

    logger.info(f"   生成 {len(shapes)} 个 GEMM 形状待预热")

    # 预热评估
    dtype = "bf16"  # 默认使用 bf16
    for i, (G, M, K, N) in enumerate(shapes):
        try:
            # 调用 evaluate 会自动缓存结果
            evaluator.evaluate(
                G=G, M=M, K=K, N=N,
                input_dtype=dtype,
                output_dtype=dtype,
                use_multiprocess=False,  # 预热时禁用多进程（避免启动开销）
            )

            if (i + 1) % 10 == 0:
                logger.info(f"   预热进度: {i+1}/{len(shapes)}")

        except Exception as e:
            logger.warning(f"   预热 GEMM ({G}, {M}, {K}, {N}) 失败: {e}")

    elapsed = time.time() - start
    logger.info(f"✅ GEMM 预热完成，耗时 {elapsed:.2f}s，缓存 {len(shapes)} 个配置")

    return len(shapes)


def get_cache_stats(evaluator: GEMMEvaluator) -> dict:
    """
    获取缓存统计信息

    Returns:
        缓存统计字典
    """
    return {
        "cached_configs": len(evaluator._cache),
        "cache_keys": list(evaluator._cache.keys())[:5],  # 显示前 5 个
    }
