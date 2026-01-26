"""
DRAM 流量差异分析

分析 Tier6 和 DS_TPU 的流量计算差异
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_traffic():
    """分析流量差异"""

    # GEMM 参数
    G, M, K, N = 8, 48, 7168, 2048
    dtype_bytes = 2  # bf16

    # Tier6 分块方案
    P_G, P_M, P_N, P_K = 8, 1, 2, 4
    num_cores = P_G * P_M * P_N * P_K  # 64

    # 每核分配
    g_nom = G // P_G  # 1
    m_nom = M // P_M  # 48
    n_nom = N // P_N  # 1024
    k_nom = K // P_K  # 1792

    # Tile 大小
    m_t, n_t, k_t = 48, 1024, 384

    print("=" * 60)
    print("GEMM 配置")
    print("=" * 60)
    print(f"GEMM: G={G}, M={M}, K={K}, N={N}")
    print(f"分块: P_G={P_G}, P_M={P_M}, P_N={P_N}, P_K={P_K} (共 {num_cores} 核)")
    print(f"每核: g={g_nom}, m={m_nom}, n={n_nom}, k={k_nom}")
    print(f"Tile: m_t={m_t}, n_t={n_t}, k_t={k_t}")

    # 理论张量大小
    A_total = G * M * K * dtype_bytes / 1e6  # MB
    B_total = G * K * N * dtype_bytes / 1e6
    C_total = G * M * N * dtype_bytes / 1e6

    print(f"\n理论张量大小:")
    print(f"  A (input):  {A_total:.2f} MB")
    print(f"  B (weight): {B_total:.2f} MB")
    print(f"  C (output): {C_total:.2f} MB")
    print(f"  总计 (A+B+C): {A_total + B_total + C_total:.2f} MB")

    # ==================== 方案 1: Tier6 当前实现 ====================
    print("\n" + "=" * 60)
    print("方案 1: Tier6 当前实现 (累加所有核心流量)")
    print("=" * 60)

    # 每核流量计算 (mnk 循环)
    tile_num_m = (m_nom + m_t - 1) // m_t  # 1
    tile_num_n = (n_nom + n_t - 1) // n_t  # 1
    tile_num_k = (k_nom + k_t - 1) // k_t  # 5

    a_blk = m_nom * k_nom * dtype_bytes  # 每核 A 大小
    b_blk = n_nom * k_nom * dtype_bytes  # 每核 B 大小
    c_blk = m_nom * n_nom * dtype_bytes  # 每核 C 大小

    # mnk 循环: A 重复 tile_num_n 次, B 重复 tile_num_m 次
    per_core_traffic = a_blk * tile_num_n + b_blk * tile_num_m + c_blk

    # 乘以 g_nom (每核负责的 G)
    per_core_total = g_nom * per_core_traffic

    # 累加所有核心
    total_traffic_v1 = per_core_total * num_cores

    print(f"每核流量计算:")
    print(f"  tile_num: m={tile_num_m}, n={tile_num_n}, k={tile_num_k}")
    print(f"  a_blk={a_blk/1e6:.2f} MB, b_blk={b_blk/1e6:.2f} MB, c_blk={c_blk/1e6:.2f} MB")
    print(f"  per_core_traffic = {per_core_traffic/1e6:.2f} MB")
    print(f"  per_core_total (x g_nom={g_nom}) = {per_core_total/1e6:.2f} MB")
    print(f"  总流量 (x {num_cores} cores) = {total_traffic_v1/1e6:.2f} MB")

    # ==================== 方案 2: 只计算最慢核心 ====================
    print("\n" + "=" * 60)
    print("方案 2: 只计算最慢核心流量 (不累加)")
    print("=" * 60)

    total_traffic_v2 = per_core_total
    print(f"总流量 = {total_traffic_v2/1e6:.2f} MB")

    # ==================== 方案 3: 考虑权重 B 复用 ====================
    print("\n" + "=" * 60)
    print("方案 3: 假设权重 B 在 G 维度复用")
    print("=" * 60)

    # 如果 B 在 G 维度共享,只需要加载一次
    b_shared = K * N * dtype_bytes  # 不乘 G
    a_total_load = G * M * K * dtype_bytes
    c_total_write = G * M * N * dtype_bytes

    total_traffic_v3 = a_total_load + b_shared + c_total_write
    print(f"A (全部): {a_total_load/1e6:.2f} MB")
    print(f"B (共享): {b_shared/1e6:.2f} MB")
    print(f"C (全部): {c_total_write/1e6:.2f} MB")
    print(f"总流量 = {total_traffic_v3/1e6:.2f} MB")

    # ==================== 方案 4: 只计算 B+C (假设 A 在片上) ====================
    print("\n" + "=" * 60)
    print("方案 4: 假设 A (激活) 在片上,只计算 B+C")
    print("=" * 60)

    total_traffic_v4 = B_total * 1e6 + C_total * 1e6
    print(f"B: {B_total:.2f} MB")
    print(f"C: {C_total:.2f} MB")
    print(f"总流量 = {total_traffic_v4/1e6:.2f} MB")

    # ==================== 方案 5: G 维度不重复计算 B ====================
    print("\n" + "=" * 60)
    print("方案 5: B 在 G 维度不重复 (每 G 独立的权重,但每核只算一次)")
    print("=" * 60)

    # 实际 MoE: 每个专家有独立权重,但每个核只负责一个 G
    # 关键: K 分到 P_K=4 核, 每核只需要 1/4 的 B
    b_per_k_core = K // P_K * N * dtype_bytes  # 每 K 核的 B

    # G 分到 P_G=8 核, 每核负责一个 G 的 A 和 C
    # N 分到 P_N=2 核, 每核负责一半的 B 和 C

    # 每核完整流量:
    # A: m_nom * k_nom (本地 K 部分)
    # B: k_nom * n_nom (本地 K 和 N 部分)
    # C: m_nom * n_nom (本地 N 部分)

    # 但 K 维度累加需要 AllReduce,这里只算 DRAM

    # 换一个思路: 计算实际需要的数据量
    # A: 每个 (g, m) 需要完整的 K
    # B: 每个 (k, n) 被所有 (g, m) 使用
    # C: 每个 (g, m, n) 独立

    # 如果 B 可以在 G 维度广播:
    b_per_g = K * N * dtype_bytes / 1e6  # 不含 G
    total_traffic_v5 = A_total + b_per_g + C_total
    print(f"A: {A_total:.2f} MB")
    print(f"B (per G, no repeat): {b_per_g:.2f} MB")
    print(f"C: {C_total:.2f} MB")
    print(f"总流量 = {total_traffic_v5:.2f} MB")

    # ==================== 方案 6: 考虑 L2 Cache ====================
    print("\n" + "=" * 60)
    print("方案 6: 假设 50% L2 Cache 命中率")
    print("=" * 60)

    l2_hit_rate = 0.5
    total_traffic_v6 = total_traffic_v1 / 1e6 * (1 - l2_hit_rate)
    print(f"原始流量: {total_traffic_v1/1e6:.2f} MB")
    print(f"L2 命中率: {l2_hit_rate*100:.0f}%")
    print(f"实际 DRAM 流量 = {total_traffic_v6:.2f} MB")

    # ==================== 对比总结 ====================
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)

    ds_tpu_ref = 129.24  # DS_TPU 参考值

    results = [
        ("Tier6 当前 (累加所有核心)", total_traffic_v1 / 1e6),
        ("只计算最慢核心", total_traffic_v2 / 1e6),
        ("B 在 G 维度复用", total_traffic_v3 / 1e6),
        ("只计算 B+C", total_traffic_v4 / 1e6),
        ("B 不重复 (per G)", total_traffic_v5),
        ("50% L2 Cache", total_traffic_v6),
    ]

    print(f"\nDS_TPU 参考值: {ds_tpu_ref:.2f} MB\n")
    print(f"{'方案':<30} {'流量 (MB)':<12} {'vs DS_TPU':<12}")
    print("-" * 54)

    for name, traffic in results:
        ratio = traffic / ds_tpu_ref
        marker = "***" if 0.9 <= ratio <= 1.1 else ""
        print(f"{name:<30} {traffic:>10.2f}  {ratio:>10.2f}x  {marker}")

    # ==================== 结论 ====================
    print("\n" + "=" * 60)
    print("结论与建议")
    print("=" * 60)

    print("""
关键发现:
1. 计算时间已对齐 (14.68 vs 14.34 us, ~2% 误差)
2. 流量差异是性能差距的根本原因

可能的解释:
- 方案 6 (50% L2 Cache): 126.09 MB, 最接近 DS_TPU 的 129.24 MB

建议下一步:
1. 确认 DS_TPU 是否使用了 L2 Cache 建模
2. 或者确认 DS_TPU 的流量计算公式
3. 如果是 L2 Cache, 需要在 Tier6 中添加 Cache 层次模型
""")


if __name__ == "__main__":
    analyze_traffic()
