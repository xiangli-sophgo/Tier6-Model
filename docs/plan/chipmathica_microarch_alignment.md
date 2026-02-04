# CHIPMathica 微架构建模对齐方案

**文档创建时间**: 2026-02-03
**目标**: 分析 CHIPMathica 和 Tier6-Model 在芯片微架构建模上的差异，制定对齐方案

---

## 1. 背景

在对比 CHIPMathica 和 Tier6-Model 的 DeepSeek-V3 仿真结果时，发现两边使用的芯片参数完全不同：

| 指标 | CHIPMathica SG2262 | Tier6 SG2262 | 差异 |
|------|-------------------|--------------|------|
| **核心数** | 4 核 | 64 核 | 16倍 |
| **Lane数/核** | 64 Lane | ? | 未明确 |
| **BF16算力** | 256 TFLOPS | 384 TFLOPS | 1.5倍 |
| **内存带宽** | 273 GB/s | 12000 GB/s | 44倍 |

这导致两边的仿真结果无法对比。根本原因是**建模粒度不同**：
- **CHIPMathica**: 细粒度微架构建模（Lane, MAC, DMA, 多级内存）
- **Tier6**: 粗粒度性能建模（总算力、总带宽）

---

## 2. CHIPMathica 微架构建模分析

### 2.1 计算单元建模

#### 2.1.1 Cube（矩阵计算单元）

**关键参数**:
```python
class CubeSpec:
    mac_per_lane: dict[str, int]  # 每Lane的MAC单元数 (按数据类型)
    lane_count: int               # Lane数量（SIMD并行度）
    core_count: int               # 核心数量
    frequency_ghz: float          # 工作频率
```

**算力计算公式**:
```
峰值算力 (FLOPS) = mac_per_lane × lane_count × core_count × frequency × 2

其中 ×2 是因为 MAC (Multiply-Accumulate) = 2个浮点操作
```

**SG2262 实例**:
```yaml
cores: 4
lanes_per_core: 64
mac_per_lane (BF16): 500
frequency: 1.0 GHz

→ 总算力 = 500 × 64 × 4 × 1.0 × 2 = 256 TFLOPS
→ 总Lane数 = 4 × 64 = 256 Lanes
→ 总MAC单元 = 500 × 256 = 128,000 个
```

#### 2.1.2 Vector（向量计算单元）

**关键参数**:
```python
class VectorSpec:
    eu_per_lane: dict[str, int]   # 每Lane的执行单元数
    lane_count: int
    core_count: int
    frequency_ghz: float
```

**算力计算**:
```
峰值算力 (FLOPS) = eu_per_lane × lane_count × core_count × frequency
```

**作用**: 用于标量运算、LayerNorm、Softmax等非矩阵运算

### 2.2 内存层级建模

CHIPMathica 建模了 **4级内存层级**:

```
GMEM (Global Memory)
  ↓ GDMA (64-273 GB/s, 100ns启动延迟)
L2M (L2 Memory)
  ↓ SDMA (1000 GB/s, 50ns启动延迟)
LMEM (Local Memory)
  ↓ 片内总线
SMEM (Shared Memory/SRAM)
```

#### 2.2.1 存储层级参数

| 层级 | 容量 | 带宽 | 延迟 | 作用域 |
|------|------|------|------|--------|
| **GMEM** | 64 GB | 273 GB/s | 100 ns | Chip |
| **L2M** | 64 MB | 1000 GB/s | 10 ns | Chip |
| **LMEM** | 64 MB (16MB/核) | 2000 GB/s | 1 ns | Core |
| **SMEM** | 256 KB (64KB/核) | 2000 GB/s | 1 ns | Core |

#### 2.2.2 关键特性

1. **分级延迟模型**: 每级有独立的访问延迟
2. **分级带宽**: 越靠近计算单元带宽越高
3. **作用域控制**: 区分芯片级、核心级共享

### 2.3 DMA 引擎建模

CHIPMathica 建模了 **3类DMA引擎**:

#### 2.3.1 GDMA (Global DMA)

```python
bandwidth: 273 / 4 = 68 GB/s (每核)
startup_latency: 100 ns
efficiency: 0.9
paths: GMEM ↔ LMEM
```

**传输时间计算**:
```
transfer_time = startup_latency + data_size / (bandwidth × efficiency)
```

#### 2.3.2 SDMA (Shared DMA)

```python
bandwidth: 1000 GB/s
startup_latency: 50 ns
efficiency: 0.95
paths: L2M ↔ LMEM
```

#### 2.3.3 CDMA (C2C DMA)

用于芯片间数据传输（多芯片场景）

### 2.4 互联网络建模

#### 2.4.1 NoC (Network-on-Chip)

```python
topology: "mesh"
bandwidth: 1000 GB/s
latency: 10 ns
```

**作用**: 核心间通信（Core-to-Core）

#### 2.4.2 C2C (Chip-to-Chip)

```python
# SG2262: 单芯片，无C2C
links: 0
bandwidth: 0

# SG2260E: 10 links
links: 10
bandwidth_per_link: 112 GB/s
total_bandwidth: 1120 GB/s
```

### 2.5 通信原语建模

CHIPMathica 实现了详细的通信模型:

#### 2.5.1 Ring AllReduce

```python
time = 2 * (n-1) / n * data_size / bandwidth + 2 * (n-1) * latency
```

#### 2.5.2 Tree AllReduce

```python
time = 2 * log2(n) * data_size / bandwidth + 2 * log2(n) * latency
```

---

## 3. Tier6-Model 建模现状

### 3.1 当前建模参数

**芯片配置** (`chip_presets/SG2262.yaml`):
```yaml
name: SG2262
num_cores: 64              # ❌ 与CHIPMathica不一致（4核）
compute_tflops_fp8: 768
compute_tflops_bf16: 384   # ❌ 与CHIPMathica不一致（256 TFLOPS）
memory_capacity_gb: 64
memory_bandwidth_gbps: 12000  # ❌ 与CHIPMathica不一致（273 GB/s）
memory_bandwidth_utilization: 0.85
lmem_capacity_mb: 2
lmem_bandwidth_gbps: 512
cube_m: 16
cube_k: 32
cube_n: 8
sram_size_kb: 2048
sram_utilization: 0.45
lane_num: 16               # ⚠️ 含义不明确（不是总Lane数）
align_bytes: 32
compute_dma_overlap_rate: 0.8
```

### 3.2 缺失的建模要素

#### 3.2.1 计算单元层面

| 参数 | CHIPMathica | Tier6 | 说明 |
|------|-------------|-------|------|
| `lanes_per_core` | ✅ 64 | ❌ 缺失 | 每核Lane数（并行度） |
| `mac_per_lane` | ✅ 500 (BF16) | ❌ 缺失 | 每Lane的MAC单元数 |
| `eu_per_lane` | ✅ 32 (BF16) | ❌ 缺失 | 每Lane的向量执行单元数 |
| `frequency_ghz` | ✅ 1.0 | ❌ 缺失 | 工作频率 |

**影响**: 无法从微架构参数推导算力，只能使用预设的总算力值

#### 3.2.2 内存层级

| 层级 | CHIPMathica | Tier6 | 说明 |
|------|-------------|-------|------|
| GMEM 容量 | ✅ 64 GB | ✅ 64 GB | 已对齐 |
| GMEM 带宽 | ✅ 273 GB/s | ❌ 12000 GB/s | **差44倍** |
| GMEM 延迟 | ✅ 100 ns | ❌ 缺失 | 无延迟建模 |
| L2M 层级 | ✅ 64 MB, 1000 GB/s | ❌ 缺失 | 无L2建模 |
| LMEM 容量 | ✅ 64 MB | ⚠️ 2 MB | **差32倍** |
| LMEM 带宽 | ✅ 2000 GB/s | ⚠️ 512 GB/s | 差4倍 |
| SMEM 容量 | ✅ 256 KB | ⚠️ 2048 KB | 含义不同？ |

**影响**: 内存瓶颈分析不准确，访存时间计算偏差大

#### 3.2.3 DMA 引擎

| DMA类型 | CHIPMathica | Tier6 | 说明 |
|---------|-------------|-------|------|
| GDMA | ✅ 68 GB/s, 100ns | ❌ 缺失 | 无独立DMA建模 |
| SDMA | ✅ 1000 GB/s, 50ns | ❌ 缺失 | 无独立DMA建模 |
| 启动延迟 | ✅ 50-100 ns | ❌ 0 | DMA开销被忽略 |
| 传输效率 | ✅ 0.9-0.95 | ❌ 1.0 | 假设100%效率 |

**影响**: 数据搬运时间计算不准确，尤其是小数据传输

#### 3.2.4 互联网络

| 参数 | CHIPMathica | Tier6 | 说明 |
|------|-------------|-------|------|
| NoC拓扑 | ✅ Mesh | ❌ 缺失 | 无NoC建模 |
| NoC带宽 | ✅ 1000 GB/s | ❌ 缺失 | - |
| NoC延迟 | ✅ 10 ns | ❌ 缺失 | - |
| C2C链路数 | ✅ 0 (SG2262) | ❌ 缺失 | - |

**影响**: 核心间通信建模缺失

---

## 4. 关键差异分析

### 4.1 建模粒度对比

| 层面 | CHIPMathica | Tier6 | 粒度差异 |
|------|-------------|-------|----------|
| **计算** | Lane → MAC → Core → Chip | Chip总算力 | 3级细化 vs 0级 |
| **内存** | GMEM → L2M → LMEM → SMEM (4级) | GMEM + LMEM (2级) | 4级 vs 2级 |
| **传输** | GDMA/SDMA/CDMA (3类) | 隐式（带宽模型） | 显式 vs 隐式 |
| **互联** | NoC + C2C (独立建模) | 互联带宽延迟（统一） | 分层 vs 扁平 |

### 4.2 算力推导方式对比

**CHIPMathica（自底向上）**:
```
微架构参数 → 算力
mac_per_lane × lanes × cores × freq × 2 = 256 TFLOPS
```

**Tier6（自顶向下）**:
```
直接给定总算力
compute_tflops_bf16 = 384 TFLOPS
```

**问题**:
1. Tier6 无法验证算力是否与微架构匹配
2. 调整微架构参数不会影响仿真性能（算力是固定值）

### 4.3 内存带宽差异分析

**疑问**: Tier6 的 12000 GB/s 是哪里来的？

**假设1**: 可能是**理论聚合带宽**
```
假设每核有独立LMEM，64核 × 187.5 GB/s = 12000 GB/s
```

**假设2**: 可能是**错误配置**
```
参考其他高端芯片（如H100: 3.35 TB/s HBM3）
但SG2262使用LPDDR5，273 GB/s是合理值
```

**结论**: Tier6 的 12000 GB/s 可能是配置错误或含义不同

---

## 5. 对齐方案

### 5.1 短期方案（快速对齐）

**目标**: 使 Tier6 能与 CHIPMathica 结果对比

**步骤**:

1. **创建 CHIPMathica 对齐的芯片配置**

创建 `chip_presets/SG2262_CHIPMathica.yaml`:
```yaml
name: SG2262_CHIPMathica
# 核心参数（与CHIPMathica对齐）
num_cores: 4                      # ✅ 对齐
lanes_per_core: 64                # ✅ 新增
mac_per_lane_bf16: 500            # ✅ 新增
frequency_ghz: 1.0                # ✅ 新增

# 推导的总算力
compute_tflops_bf16: 256.0        # ✅ 对齐 (500×64×4×1.0×2/1000)
compute_tflops_fp8: 512.0         # ✅ 推导 (1000×64×4×1.0×2/1000)

# 内存（与CHIPMathica对齐）
memory_capacity_gb: 64
memory_bandwidth_gbps: 273.0      # ✅ 对齐（LPDDR5）
memory_bandwidth_utilization: 0.85

# 片内存储（对齐）
lmem_capacity_mb: 64              # ✅ 对齐 (4核×16MB)
lmem_bandwidth_gbps: 2000.0       # ✅ 对齐
sram_size_kb: 256                 # ✅ 对齐 (4核×64KB)
sram_utilization: 0.45

# 微架构（保持现有）
cube_m: 16
cube_k: 32
cube_n: 8
lane_num: 16
align_bytes: 32
compute_dma_overlap_rate: 0.8
```

2. **更新对比脚本使用新配置**

修改 `backend/tests/comparison/chipmathica_tier6_comparison.py`:
```python
# 使用对齐的芯片参数
chip_params = {
    "name": "SG2262_CHIPMathica",
    "num_cores": 4,              # CHIPMathica对齐
    "compute_tflops_fp8": 512.0,
    "compute_tflops_bf16": 256.0,  # CHIPMathica对齐
    "memory_capacity_gb": 64.0,
    "memory_bandwidth_gbps": 273.0,  # CHIPMathica对齐
    # ... 其他参数
}
```

3. **验证对比结果**

重新运行对比，预期：
- ✅ 算力匹配：256 TFLOPS vs 256 TFLOPS
- ✅ 内存带宽匹配：273 GB/s vs 273 GB/s
- ⚠️ 细节差异仍会存在（DMA、NoC建模不同）

### 5.2 中期方案（微架构扩展）

**目标**: 在 Tier6 中增加微架构参数支持

**新增参数**:

```yaml
# chip_presets 扩展
microarchitecture:
  compute:
    lanes_per_core: 64
    mac_per_lane:
      bf16: 500
      fp8: 1000
      int8: 1000
    eu_per_lane:
      bf16: 32
      fp32: 16
    frequency_ghz: 1.0

  memory_hierarchy:
    gmem:
      capacity_gb: 64
      bandwidth_gbps: 273
      latency_ns: 100
    l2m:
      capacity_mb: 64
      bandwidth_gbps: 1000
      latency_ns: 10
    lmem:
      capacity_mb_per_core: 16
      bandwidth_gbps: 2000
      latency_ns: 1
    smem:
      capacity_kb_per_core: 64
      bandwidth_gbps: 2000
      latency_ns: 1

  dma_engines:
    gdma:
      bandwidth_gbps_per_core: 68
      startup_latency_ns: 100
      efficiency: 0.9
    sdma:
      bandwidth_gbps: 1000
      startup_latency_ns: 50
      efficiency: 0.95

  interconnect:
    noc:
      topology: mesh
      bandwidth_gbps: 1000
      latency_ns: 10
    c2c:
      links: 0
      bandwidth_per_link_gbps: 0
```

**实现步骤**:

1. 扩展 `ChipHardwareConfig` 数据类
2. 修改验证逻辑支持可选的微架构参数
3. 更新仿真器使用微架构参数（如果提供）
4. 保持向后兼容（微架构参数可选）

### 5.3 长期方案（架构升级）

**目标**: 全面对齐 CHIPMathica 的微架构建模

**核心改造**:

1. **计算建模升级**
   - 从 Lane 级别建模算力
   - 支持 Cube 和 Vector 双计算单元
   - 频率感知的性能模型

2. **内存建模升级**
   - 4级内存层级（GMEM/L2M/LMEM/SMEM）
   - 每级独立的延迟模型
   - 作用域控制（Chip/Core）

3. **DMA建模升级**
   - 显式的 GDMA/SDMA 建模
   - 启动延迟 + 传输延迟
   - 效率因子

4. **互联建模升级**
   - NoC 拓扑建模（Mesh/Ring）
   - C2C 独立建模
   - 延迟 + 带宽双因子

5. **评估器升级**
   - 从微架构参数计算操作时间
   - 支持 Roofline 分析
   - 支持瓶颈归因

---

## 6. 实施优先级

### P0（立即执行）

- [x] ✅ **对比分析完成**：本文档
- [ ] 🔄 **创建对齐配置**：`SG2262_CHIPMathica.yaml`
- [ ] 🔄 **更新对比脚本**：使用对齐参数
- [ ] 🔄 **验证结果差异**：重新运行对比

**预期时间**: 2小时
**预期收益**: 能够进行有意义的对比

### P1（本周完成）

- [ ] **微架构参数扩展**：扩展芯片配置schema
- [ ] **向后兼容性测试**：确保现有配置仍可用
- [ ] **文档更新**：更新芯片配置文档

**预期时间**: 1天
**预期收益**: 支持更精细的芯片建模

### P2（中长期）

- [ ] **评估器重构**：支持微架构级评估
- [ ] **Roofline 分析**：基于微架构的性能上界
- [ ] **自动推导**：从微架构参数自动推导总性能

**预期时间**: 1-2周
**预期收益**: 达到 CHIPMathica 级别的建模精度

---

## 7. 风险与挑战

### 7.1 技术风险

**R1: 微架构参数获取困难**
- **问题**: 真实芯片的 Lane、MAC 参数可能是机密
- **缓解**: 使用公开芯片（H100、SG2262）的已知参数作为参考

**R2: 建模复杂度增加**
- **问题**: 4级内存 + 3类DMA 会大幅增加代码复杂度
- **缓解**: 逐步引入，保持可选（向后兼容）

**R3: 性能开销**
- **问题**: 细粒度建模可能降低仿真速度
- **缓解**: 提供"快速模式"（使用总算力）和"精确模式"（使用微架构）

### 7.2 数据风险

**R4: CHIPMathica 参数可能不准确**
- **问题**: CHIPMathica 的配置可能也是估计值
- **缓解**: 对比真实硬件测试结果（如有）

**R5: 配置不一致**
- **问题**: 存在多个 SG2262 配置版本
- **缓解**: 明确版本和来源，建立配置溯源

---

## 8. 附录

### 8.1 术语表

| 术语 | 含义 | 示例 |
|------|------|------|
| **Lane** | SIMD并行通道，类似GPU的CUDA Core | 64 Lanes/核 |
| **MAC** | 乘加单元 (Multiply-Accumulate) | 500 MAC/Lane |
| **EU** | 执行单元 (Execution Unit)，用于向量运算 | 32 EU/Lane |
| **GMEM** | 全局内存（HBM/DDR） | 64GB, 273GB/s |
| **L2M** | L2缓存（芯片级共享） | 64MB, 1000GB/s |
| **LMEM** | 本地内存（核心级） | 16MB/核, 2000GB/s |
| **SMEM** | 共享内存（SRAM） | 64KB/核 |
| **GDMA** | 全局DMA (GMEM↔LMEM) | 68GB/s/核 |
| **SDMA** | 共享DMA (L2M↔LMEM) | 1000GB/s |
| **NoC** | 片内网络 (Network-on-Chip) | Mesh, 1000GB/s |
| **C2C** | 芯片间互联 (Chip-to-Chip) | 10×112GB/s |

### 8.2 参考资料

1. CHIPMathica 芯片配置: `CHIPMathica/configs/chips/sg2262.yaml`
2. Tier6 芯片配置: `Tier6-Model/backend/configs/chip_presets/SG2262.yaml`
3. SG2262 芯片实现: `CHIPMathica/chipmathica/arch/chips/sg2262.py`
4. 算力计算: `CHIPMathica/chipmathica/arch/compute.py`

### 8.3 示例：算力验证

**CHIPMathica SG2262 算力验证**:
```python
# 参数
cores = 4
lanes_per_core = 64
mac_per_lane_bf16 = 500
frequency_ghz = 1.0

# 计算
total_lanes = cores × lanes_per_core = 4 × 64 = 256
total_macs = total_lanes × mac_per_lane = 256 × 500 = 128,000
peak_flops = total_macs × frequency × 2 (MAC)
           = 128,000 × 1.0 GHz × 2
           = 256,000 GFLOPS
           = 256 TFLOPS ✅
```

**Tier6 当前配置**:
```yaml
compute_tflops_bf16: 384  # ❌ 无法从微架构推导
```

---

**文档状态**: 待审阅
**下一步**: 执行 P0 任务，创建对齐配置并验证
