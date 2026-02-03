
  GPU 显存中的数据结构
  GPU 显存 (24GB on RTX 4090)：

  ┌─────────────────────────────────────┐
  │ Embedding 表 (已加载)               │
  │ Shape: [32000, 4096]                │
  │ 大小: 512 MB                        │
  ├─────────────────────────────────────┤
  │ token_id 0   → [0.1, 0.2, ...]      │
  │ token_id 1   → [0.3, 0.4, ...]      │
  │ ...                                 │
  │ token_id 2769 → [0.2, -0.5, ...]  ← 要读
  │ ...                                 │
  │ token_id 32000 → [-0.1, 0.6, ...]  │
  └─────────────────────────────────────┘

---

  GPU 执行 Embedding Lookup 的过程

  步骤 1：发起请求

# GPU 端的 CUDA 核函数（伪代码）

  __global__ void embedding_lookup(
      int* token_ids,           # 输入: [2769, 3614, 1140, 4507]
      float* embedding_table,   # embedding 表首地址
      float* output,            # 输出位置
      int embedding_dim         # 4096
  ) {
      // 一个 GPU 线程块处理一个 token
      int token_id = token_ids[blockIdx.x];

    // 计算该 token 向量在显存中的地址
      float* vector = embedding_table + token_id * embedding_dim;

    // 多个线程并行读取该向量的元素
      for(int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
          output[blockIdx.x * embedding_dim + i] = vector[i];
      }
  }

---

  步骤 2：并行读取

  假设输入 4 个 token：[2769, 3614, 1140, 4507]

  GPU 并行执行 4 个线程块：

  Block 0 (256 threads)     Block 1 (256 threads)
    │                          │
    ▼ (同时读)                 ▼ (同时读)
  embedding_table[2769]    embedding_table[3614]
    16 KB ────┐                 │── 16 KB
              │                 │
              ▼                 ▼
         GPU L2 缓存 (12 MB 共享)

  Block 2 (256 threads)     Block 3 (256 threads)
    │                          │
    ▼ (同时读)                 ▼ (同时读)
  embedding_table[1140]    embedding_table[4507]
    16 KB ────┐                 │── 16 KB
              │                 │
              └────────┬────────┘
                       ▼
             GPU 显存控制器处理

---

  步骤 3：数据路径

  GPU 显存访问延迟（按快慢顺序）：

1. L1 缓存（每 SM 128 KB）
   延迟：~32 cycles
2. L2 缓存（共享，12 MB）
   延迟：~200 cycles
3. 主显存（GDDR6X）
   延迟：~400-800 cycles

  显存带宽：

- RTX 4090：1000 GB/s
- H100：3400 GB/s

---

  关键问题：读多少数据？

  答案：只读需要的行，不读整个表！

  Embedding 表总大小：512 MB

  但一次查表只读：
  4 个 token × 4096 维 × 4 bytes = 64 KB
                                 （只有表的 0.01%）

  100 个 token 也只读：
  100 × 4096 × 4 = 1.6 MB
                 （只有表的 0.3%）

  所以：完全不需要加载整个表到缓存！

---

  性能数据

  单个 Token 的查表时间

  RTX 4090：

  1 个 token 查表：
    数据量：4096 × 4 bytes = 16 KB
    显存带宽：1000 GB/s
    耗时：16 KB / 1000 GB/s = 0.016 ms

  100 个 token 查表：
    数据量：1.6 MB
    耗时：0.0016 ms × 100 ≈ 0.16 ms

    （这些都极快，不是推理瓶颈！）

### 完整推理流程的时间分布

```
LLaMA-70B 推理 4 个输入 token，生成 100 个 token：

Prefill 阶段（处理输入）：
├─ Embedding lookup: 0.064 ms
├─ Transformer 计算: ~10 ms
└─ 小计: ~10 ms

Decode 阶段（生成 100 个 token）：
  100 次循环：
  ├─ 每步 Embedding lookup: 0.016 ms
  ├─ 每步 Transformer 计算: ~100 ms ← 瓶颈！
  └─ 小计: ~10000 ms

────────────────────────────
总耗时: ~10 秒

结论：Embedding lookup 只占 0.064 ms，
     完全不是瓶颈！
     真正的瓶颈是 Transformer 中的 KV Cache 读取。
```

---

## GPU 访问模式优化

```
坏的情况（随机访问）：
token_ids = [2769, 15000, 500, 29999, ...]
           │    │     │   │
           │    └─────┴───┴─ 在表中跳来跳去

L2 缓存命中率低
显存延迟大
性能：~300 GB/s

好的情况（顺序或聚集访问）：
token_ids = [1000, 1001, 1002, 1003, ...]
           │    │    │    │
           └────┴────┴────┘ 顺序访问，利用预取

L2 缓存高命中率
显存延迟小
性能：~800 GB/s

GPU 会自动优化（排序 + 重排）。
```

---

## PyTorch 代码实现

```python
import torch
import time

# GPU 初始化
embedding_table = torch.randn(32000, 4096).cuda()
token_ids = torch.tensor([2769, 3614, 1140, 4507]).cuda()

# ===== 查表操作 =====
start = time.time()
embeddings = embedding_table[token_ids]  # GPU 上执行
torch.cuda.synchronize()
elapsed = (time.time() - start) * 1000

print(f"查表耗时: {elapsed:.4f} ms")  # 输出: ~0.2 ms
print(f"输出形状: {embeddings.shape}")  # [4, 4096]

# 实际传输的数据：
bytes_read = 4 * 4096 * 4  # 64 KB
print(f"读取数据: {bytes_read / 1024:.1f} KB")  # 64 KB
```

---

## 总结

```
GPU Embedding Lookup：

1. 发起请求：Token IDs 在 GPU 显存中
2. 并行读取：256+ 线程同时读不同 token 的向量
3. 显存访问：通过 L1/L2 缓存 → 主显存
4. 数据量：按需读取，只读所需行（几 KB-MB）
5. 性能：极快，0.016 ms/token
6. 是否瓶颈：NO，完全不是推理的瓶颈

关键：不需要读整个表！
      只读所需的部分，极其高效。
```
