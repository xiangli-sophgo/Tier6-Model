"""诊断脚本: 逐算子追踪 A 管线中间结果，与 B (CHIPMathica) 对比

用法: cd backend && python3.12 -m tests.diagnostic_pipeline_trace

配置: DeepSeek-V3 671B, TP=1, DP=32, EP=32, batch=2048, decode (seq=1, kv=4097)
对齐 B 的芯片参数: SG2262 (core=64, freq=1.5GHz, BW=8601.6 GB/s, SRAM=128MB)
"""

import sys
import os
import math

# 加入项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from math_model.L0_entry.config_loader import load_chip, load_model, load_topology
from math_model.L0_entry.eval_config import (
    EvalConfig,
    build_eval_config,
    _extract_first_chip_config,
)


def build_config():
    """构建评估配置 (对齐 B 的 DeepSeek-V3 decode 场景)"""
    model_cfg = load_model("DeepSeek-V3-671B-A37B")
    topo_cfg = load_topology("P1-R1-B4-C32")
    chip_cfg = _extract_first_chip_config(topo_cfg["chips"])

    inference_cfg = {
        "batch_size": 2048,
        "input_seq_length": 4096,
        "output_seq_length": 1,
        "weight_dtype": "fp8",
        "activation_dtype": "fp8",
    }

    manual_parallelism = {
        "tp": 1,
        "pp": 1,
        "dp": 32,
        "ep": 32,
        "moe_tp": 1,
        "seq_len": 4096,
        "enable_tp_sp": False,
        "enable_ring_attention": False,
        "enable_zigzag": False,
        "embed_tp": 1,
        "lmhead_tp": 1,
        "comm_protocol": 0,
        "kv_cache_rate": 1.0,
        "is_prefill": False,
    }

    return build_eval_config(chip_cfg, model_cfg, topo_cfg, manual_parallelism, inference_cfg)


def trace_l1(eval_config: EvalConfig):
    """L1: 追踪 WorkloadIR 的层和算子"""
    from math_model.L1_workload.models.llm.deepseek import DeepSeekV3Model

    model = DeepSeekV3Model.from_model_config(eval_config.model)
    ir = model.to_ir()
    layers = ir.get_layers()

    print("=" * 80)
    print("L1: WorkloadIR")
    print("=" * 80)
    print(f"Total layers: {len(layers)}")

    # 只看前几个关键层
    for i, layer in enumerate(layers[:8]):
        ops = layer.ops or []
        print(f"\n--- Layer {i}: {layer.name} (type={layer.op_type}) ---")
        print(f"  attrs: layer_type={layer.attrs.get('layer_type', 'N/A')}")
        for j, op in enumerate(ops):
            if hasattr(op, '_m'):
                print(f"  Op[{j}] {op.name} (type={op.op_type}): M={op._m}, K={op._k}, N={op._n}, G={getattr(op, '_g', 1)}")
            elif hasattr(op, '_b'):
                print(f"  Op[{j}] {op.name} (type={op.op_type}): B={op._b}, QS={op._qs}, KS={op._ks}, QD={op._qd}, VD={op._vd}")
            else:
                print(f"  Op[{j}] {op.name} (type={op.op_type})")
            # 打印输入输出 shape
            for inp in op.inputs:
                print(f"    input: {inp.name} shape={inp.shape}")
            for out in op.outputs:
                print(f"    output: {out.name} shape={out.shape}")

    return ir


def trace_l3(eval_config: EvalConfig, ir):
    """L3: 追踪并行切分和 Tiling"""
    from math_model.L2_arch.chip import ChipSpecImpl
    from math_model.L3_mapping.parallelism.planner import DeploymentSpec, ParallelismPlanner, BoardSpec
    from math_model.L0_entry.eval_config import _require

    # L2: ChipSpec
    chip_config = eval_config.chip_config
    chip_name = _require(chip_config, "name", "chip config")
    chip = ChipSpecImpl.from_config(chip_name, chip_config)

    print("\n" + "=" * 80)
    print("L2: ChipSpec")
    print("=" * 80)
    print(f"  Name: {chip.name}")
    print(f"  Cores: {chip.core_count}")
    print(f"  Peak FP8 FLOPS: {chip.get_peak_flops('FP8', 'cube') / 1e12:.3f} TFLOPS")
    print(f"  Peak BF16 FLOPS: {chip.get_peak_flops('BF16', 'cube') / 1e12:.3f} TFLOPS")
    print(f"  GMEM BW: {chip.get_gmem_bandwidth():.1f} GB/s")
    print(f"  Total SRAM: {chip.get_total_sram() / 1024 / 1024:.1f} MB")

    # L3: Parallelism
    dep = eval_config.deployment
    deployment = DeploymentSpec(
        tp=dep.tp, pp=dep.pp, ep=dep.ep, dp=dep.dp,
        moe_tp=dep.moe_tp, seq_len=dep.seq_len,
        batch_size=dep.batch_size,
        enable_tp_sp=dep.enable_tp_sp,
        enable_ring_attention=dep.enable_ring_attention,
        embed_tp=dep.embed_tp, lmhead_tp=dep.lmhead_tp,
        comm_protocol=dep.comm_protocol,
        kv_cache_rate=dep.kv_cache_rate,
        is_prefill=dep.is_prefill,
    )

    brd = eval_config.board
    board = BoardSpec(
        num_chips=brd.num_chips,
        chip_memory_gb=brd.chip_memory_gb,
        inter_chip_bw_gbps=brd.inter_chip_bw_gbps,
    )

    print("\n" + "=" * 80)
    print("L3: Parallelism Planning")
    print("=" * 80)
    print(f"  TP={dep.tp}, DP={dep.dp}, EP={dep.ep}, PP={dep.pp}, moe_tp={dep.moe_tp}")
    print(f"  batch_size={dep.batch_size}, seq_len={dep.seq_len}")
    print(f"  q_seq_len={dep.q_seq_len}, kv_seq_len={dep.kv_seq_len}")
    print(f"  is_prefill={dep.is_prefill}")
    print(f"  batch_local (B expected) = 2048 // 32 = 64")
    print(f"  batch_local (A ceil) = ceil(2048/32) = {math.ceil(2048/32)}")

    planner = ParallelismPlanner(deployment, board)
    dist_model = planner.plan(ir)

    # 打印所有 distributed op 的 local_shape
    compute_ops = dist_model.get_compute_ops()
    comm_ops = dist_model.get_comm_ops()

    print(f"\n  Total distributed ops: {len(dist_model.ops)}")
    print(f"  Compute ops: {len(compute_ops)}, Comm ops: {len(comm_ops)}")

    # 只看前20个 compute op 的 local_shape
    print(f"\n--- First 20 Compute Ops (local_shape after TP/DP/MoE/SP shard) ---")
    for i, op in enumerate(compute_ops[:20]):
        layer_type = op.attrs.get("layer_type", "?")
        op_role = op.attrs.get("op_role", "?")
        print(f"  [{i:3d}] {op.op_id:<40s}  type={op.op_type:<8s}  layer={layer_type:<12s}  role={op_role:<15s}  shape={op.local_shape}")

    # 特别打印第一个 MoE 层的所有 ops
    print(f"\n--- First MoE Layer Ops (compute + comm) ---")
    first_moe_layer = None
    for op in dist_model.ops:
        layer_name = op.attrs.get("layer_name", "")
        if "moe" in layer_name.lower() and first_moe_layer is None:
            first_moe_layer = layer_name
    if first_moe_layer:
        for op in dist_model.ops:
            if op.attrs.get("layer_name", "") == first_moe_layer:
                role = "COMPUTE" if op.role.name == "COMPUTE" else "COMM"
                reason = op.attrs.get("reason", "") or (op.reason or "")
                print(f"  [{role:7s}] {op.op_id:<50s}  shape={op.local_shape}  reason={reason}")
                if op.comm_bytes:
                    print(f"           comm_bytes={op.comm_bytes}")

    # comm ops
    print(f"\n--- First 10 Comm Ops ---")
    for i, op in enumerate(comm_ops[:10]):
        reason = op.attrs.get("reason", "") or (op.reason or "")
        print(f"  [{i:3d}] {op.op_id:<50s}  comm_type={op.comm_type}  bytes={op.comm_bytes}  reason={reason}")

    return dist_model, chip


def trace_l3_tiling(eval_config: EvalConfig, dist_model, chip):
    """L3: 追踪 Tiling"""
    from math_model.L3_mapping.tiling.planner import TilingPlanner
    from math_model.L4_evaluation.evaluators.precise import PreciseTileEvaluator

    dep = eval_config.deployment
    cube_dtype = eval_config.inference.weight_dtype.upper()
    if cube_dtype not in ("FP8", "INT8"):
        cube_dtype = "BF16"
    compute_tflops = chip.get_peak_flops(cube_dtype, "cube") / 1e12
    memory_bw_gbps = chip.get_gmem_bandwidth()

    print("\n" + "=" * 80)
    print("L3: Tiling Planning")
    print("=" * 80)
    print(f"  cube_dtype={cube_dtype}, compute_tflops={compute_tflops:.3f}, memory_bw_gbps={memory_bw_gbps:.1f}")

    precise_evaluator = PreciseTileEvaluator(
        compute_tflops=compute_tflops,
        memory_bandwidth_gbps=memory_bw_gbps,
        is_prefill=dep.is_prefill,
        enable_zigzag=dep.enable_zigzag,
    )

    tile_plan = TilingPlanner(chip, l4_evaluator=precise_evaluator).plan(dist_model)

    # 打印前20个 compute op 的 tile config 和精评估结果
    print(f"\n--- Tile Configs & Precise Evaluation (first 20 compute ops) ---")
    compute_ops = dist_model.get_compute_ops()

    for i, op in enumerate(compute_ops[:20]):
        tc = tile_plan.tile_configs.get(op.op_id)
        kc = tile_plan.kernel_configs.get(op.op_id, {})
        layer_type = op.attrs.get("layer_type", "?")
        op_role = op.attrs.get("op_role", "?")

        if tc:
            print(f"\n  [{i:3d}] {op.op_id}")
            print(f"        layer={layer_type}, role={op_role}")
            print(f"        local_shape={op.local_shape}")
            print(f"        tile: m={tc.tile_m}, k={tc.tile_k}, n={tc.tile_n}")
            if kc:
                traffic = kc.get("traffic", "N/A")
                t_comp = kc.get("t_compute_ms", "N/A")
                t_mem = kc.get("t_memory_ms", "N/A")
                cores = kc.get("active_cores", "N/A")
                urate = kc.get("arch_urate", "N/A")
                overlap = kc.get("overlap_rate", "N/A")
                print(f"        kernel: traffic={traffic}, t_compute_ms={t_comp}, t_memory_ms={t_mem}")
                print(f"                active_cores={cores}, arch_urate={urate}, overlap_rate={overlap}")

    return tile_plan


def trace_l4(eval_config: EvalConfig, dist_model, chip, tile_plan):
    """L4: 追踪评估引擎"""
    from math_model.L3_mapping.scheduling.scheduler import Scheduler
    from math_model.L4_evaluation.engine import EvaluationEngine
    from math_model.L4_evaluation.metrics import Granularity
    from math_model.L0_entry.engine import _build_hardware_spec

    dep = eval_config.deployment

    # Scheduling
    exec_plan = Scheduler().plan(dist_model, tile_plan)

    # Build hardware
    hardware = _build_hardware_spec(chip, eval_config)

    deployment_dict = {
        "enable_ring_attention": dep.enable_ring_attention,
        "tp": dep.tp,
    }

    engine = EvaluationEngine()
    engine_result = engine.evaluate(
        exec_plan=exec_plan,
        distributed_model=dist_model,
        hardware=hardware,
        granularity=Granularity.CHIP,
        output_tokens=dep.batch_size,
        deployment_config=deployment_dict,
        is_prefill=dep.is_prefill,
    )

    print("\n" + "=" * 80)
    print("L4: Evaluation Engine Results")
    print("=" * 80)
    print(f"  Hardware: compute_tflops={hardware.get('compute_tflops', 'N/A')}, memory_bw={hardware.get('memory_bandwidth_gbps', 'N/A')}")

    steps = engine_result.step_metrics
    print(f"\n  Total steps: {len(steps)}")

    # 汇总
    total_compute = sum(s.t_compute for s in steps)
    total_comm = sum(s.t_comm for s in steps)
    total_wait = sum(s.t_wait for s in steps)
    total_time = sum(s.t_total for s in steps)
    total_flops = sum(s.flops for s in steps)
    total_bytes_r = sum(s.bytes_read for s in steps)
    total_bytes_w = sum(s.bytes_write for s in steps)
    total_bytes = total_bytes_r + total_bytes_w

    print(f"\n  Totals:")
    print(f"    total_time = {total_time:.4f} ms")
    print(f"    compute    = {total_compute:.4f} ms")
    print(f"    comm       = {total_comm:.4f} ms")
    print(f"    wait       = {total_wait:.4f} ms")
    print(f"    flops      = {total_flops:.0f} ({total_flops/1e9:.3f} GFLOPS)")
    print(f"    bytes_read = {total_bytes_r:.0f} ({total_bytes_r/1e9:.3f} GB)")
    print(f"    bytes_write= {total_bytes_w:.0f} ({total_bytes_w/1e9:.3f} GB)")
    print(f"    bytes_total= {total_bytes:.0f} ({total_bytes/1e9:.3f} GB)")

    # 前30个 step 详细
    print(f"\n--- First 30 Steps ---")
    for i, step in enumerate(steps[:30]):
        reason = step.meta.get("reason", "")
        bottleneck = step.meta.get("bottleneck", "")
        total_b = step.bytes_read + step.bytes_write
        print(
            f"  [{i:3d}] {step.op_id:<50s}  "
            f"t_comp={step.t_compute:10.4f}  t_comm={step.t_comm:10.4f}  t_total={step.t_total:10.4f}  "
            f"flops={step.flops:>15.0f}  bytes={total_b:>15.0f}  "
            f"bneck={bottleneck:<8s}  reason={reason}"
        )

    # MFU / MBU 计算
    agg = engine_result.aggregates
    print(f"\n  Aggregates:")
    print(f"    total_time = {agg.total_time:.4f} ms")
    print(f"    compute_time = {agg.total_compute_time:.4f} ms")
    print(f"    comm_time = {agg.total_comm_time:.4f} ms")
    print(f"    mfu = {agg.mfu:.6f}")
    print(f"    mbu = {agg.mbu:.6f}")
    print(f"    tps = {agg.tps:.2f}")
    print(f"    total_flops = {agg.total_flops:.0f} ({agg.total_flops/1e9:.3f} GFLOPS)")
    print(f"    total_bytes = {agg.total_bytes:.0f} ({agg.total_bytes/1e9:.3f} GB)")

    return engine_result


def compare_with_b():
    """打印 B (CHIPMathica) 的已知参考值"""
    print("\n" + "=" * 80)
    print("B (CHIPMathica) Reference Values")
    print("=" * 80)
    print("  Config: DeepSeek-V3 671B, TP=1, DP=32, EP=32, batch=2048, decode")
    print("  Chip: SG2262 (core=64, freq=1.5GHz)")
    print("")
    print("  B's key formulas:")
    print("    batch_local = batch_size // dp = 2048 // 32 = 64")
    print("    tokens_local = batch_local * q_seq_len = 64 * 1 = 64")
    print("")
    print("  B's MLA decode ops (per chip, batch_local=64):")
    print("    q_a:    M=64, K=7168, N=1536  -> flops = 2*64*7168*1536 = 1,409,286,144")
    print("    q_b:    M=64, K=1536, N=24576 -> flops = 2*64*1536*24576 = 4,831,838,208")
    print("    kv_a:   M=64, K=7168, N=576   -> flops = 2*64*7168*576   = 528,482,304")
    print("    kv_b:   M=64, K=512, N=32768  -> flops = 2*64*512*32768  = 2,147,483,648")
    print("    fa2:    B=64*128=8192, QS=1, KS=4097, QD=192, VD=128")
    print("            flops = 2*8192*1*4097*(192+128) = 21,462,507,520")
    print("    o_proj: M=64, K=16384, N=7168 -> flops = 2*64*16384*7168 = 15,032,385,536")
    print("  MLA total flops per layer = ~45.4 GFLOPS")
    print("")
    print("  A's expected local_shape (TP=1, DP=32):")
    print(f"    batch_local = ceil(2048/32) = {math.ceil(2048/32)}")
    print(f"    tokens_local = 64 * 1 = 64 (decode)")


def main():
    print("Diagnostic Pipeline Trace: A (Tier6+Model) vs B (CHIPMathica)")
    print("Config: DeepSeek-V3 671B, TP=1, DP=32, EP=32, batch=2048, decode")
    print("Chip: SG2262 (aligned to B: core=64, freq=1.5GHz, BW=8601.6 GB/s)")
    print()

    # 构建配置
    eval_config = build_config()

    # B 参考值
    compare_with_b()

    # L1: WorkloadIR
    ir = trace_l1(eval_config)

    # L3: Parallelism
    dist_model, chip = trace_l3(eval_config, ir)

    # L3: Tiling
    tile_plan = trace_l3_tiling(eval_config, dist_model, chip)

    # L4: Evaluation
    engine_result = trace_l4(eval_config, dist_model, chip, tile_plan)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
