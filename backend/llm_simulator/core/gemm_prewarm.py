"""
GEMM 评估器离线预调优

在模拟器初始化时预先评估常见的 GEMM 形状，避免运行时重复搜索。
"""

import logging
from typing import List, Tuple, Optional
from ..evaluators import GEMMEvaluator

logger = logging.getLogger(__name__)


def generate_transformer_gemm_shapes(
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    batch_sizes: List[int],
    seq_lengths: List[int],
    tp: int = 1,  # ⭐ 新增: 张量并行度
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
        tp: 张量并行度（默认1，无并行）
        mla_config: MLA 配置（可选）
        moe_config: MoE 配置（可选）

    Returns:
        List of (G, M, K, N) 元组
    """
    shapes = []
    head_dim = hidden_size // num_attention_heads

    # ⭐ TP分片后的维度
    heads_per_tp = num_attention_heads // tp
    kv_heads_per_tp = num_kv_heads // tp
    hidden_per_tp = hidden_size // tp
    intermediate_per_tp = intermediate_size // tp

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            M = batch_size * seq_len

            # ========== 标准 Attention ==========
            if not mla_config:
                # ⭐ TP后的QKV投影形状
                # QKV合并投影: qkv_dim = (heads_per_tp + 2 * kv_heads_per_tp) * head_dim
                qkv_dim = (heads_per_tp + 2 * kv_heads_per_tp) * head_dim
                shapes.append((1, M, hidden_size, qkv_dim))  # QKV projection (TP分片)

                # ⭐ TP后的Output投影形状
                # Input: heads_per_tp * head_dim, Output: hidden_size (全量，后续AllReduce)
                shapes.append((1, M, heads_per_tp * head_dim, hidden_size))

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
                # ⭐ TP后的FFN形状
                # Gate/Up投影: hidden_size -> intermediate_per_tp
                shapes.append((1, M, hidden_size, intermediate_per_tp))  # gate
                shapes.append((1, M, hidden_size, intermediate_per_tp))  # up
                # Down投影: intermediate_per_tp -> hidden_size
                shapes.append((1, M, intermediate_per_tp, hidden_size))  # down
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
    tp: int = 1,  # ⭐ 新增: 张量并行度
    mla_config: Optional[dict] = None,
    moe_config: Optional[dict] = None,
    progress_callback: Optional[callable] = None,  # 进度回调
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

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("🔥 GEMM 评估器预热")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    start = time.time()

    # 生成常见的批次大小和序列长度组合
    batch_sizes = [batch_size]  # 只预热当前批次大小

    # 🚀 优化策略：只预热 Decode 阶段（最频繁使用）
    # Prefill 形状会在首次运行时按需评估并缓存
    # 这样可以将预热形状数量减少 50%，加速启动
    seq_lengths = [
        1,  # Decode (每次生成 1 个 token，最频繁使用)
    ]

    # 如果 input_seq_length 很小（<= 512），也预热 Prefill
    # 因为小序列长度搜索很快
    if input_seq_length <= 512:
        seq_lengths.append(input_seq_length)

    # 生成所有 GEMM 形状
    shapes = generate_transformer_gemm_shapes(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        batch_sizes=batch_sizes,
        seq_lengths=seq_lengths,
        tp=tp,  # ⭐ 传递TP参数
        mla_config=mla_config,
        moe_config=moe_config,
    )

    logger.info(f"   模型配置: hidden={hidden_size}, intermediate={intermediate_size}")
    logger.info(f"   批次大小: {batch_size}, 预热序列长度: {seq_lengths}")
    logger.info(f"   并行策略: TP={tp}")  # ⭐ 显示TP配置
    logger.info(f"   生成 {len(shapes)} 个 GEMM 形状待预热")
    logger.info(f"   搜索模式: 多进程并行搜索 (加速 15-20x)")

    # 预热评估
    dtype = "bf16"  # 默认使用 bf16
    prewarm_times = []

    for i, (G, M, K, N) in enumerate(shapes):
        try:
            shape_start = time.time()

            # 调用 evaluate 会自动缓存结果
            # 🚀 启用多进程加速搜索（24核 vs 1核，提速 15-20x）
            evaluator.evaluate(
                G=G, M=M, K=K, N=N,
                input_dtype=dtype,
                output_dtype=dtype,
                use_multiprocess=True,  # ✅ 启用多进程并行搜索
            )

            shape_time = (time.time() - shape_start) * 1000
            prewarm_times.append(shape_time)

            # 每5个或在最后打印进度
            if (i + 1) % 5 == 0 or (i + 1) == len(shapes):
                avg_time = sum(prewarm_times[-5:]) / min(5, len(prewarm_times[-5:]))
                logger.info(f"   进度: {i+1}/{len(shapes)} (平均 {avg_time:.1f}ms/形状)")

                # 调用进度回调（预热占 10% 的总进度）
                if progress_callback:
                    prewarm_progress = (i + 1) / len(shapes) * 10  # 0-10%
                    progress_callback(prewarm_progress, f"GEMM 预热 {i+1}/{len(shapes)}")

        except Exception as e:
            logger.warning(f"   ⚠️  预热失败 GEMM({G},{M},{K},{N}): {e}")

    elapsed = time.time() - start
    avg_prewarm_time = sum(prewarm_times) / len(prewarm_times) if prewarm_times else 0

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"✅ 预热完成")
    logger.info(f"   总耗时: {elapsed:.2f}s")
    logger.info(f"   已缓存: {len(shapes)} 个配置")
    logger.info(f"   平均耗时: {avg_prewarm_time:.1f}ms/形状")
    if prewarm_times:
        logger.info(f"   最慢形状: {max(prewarm_times):.1f}ms")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

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
