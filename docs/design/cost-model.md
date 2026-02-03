# LLM 推理集群成本模型

> **文档版本**: v1.0
> **创建时间**: 2026-01-28
> **适用范围**: 大规模 LLM 推理集群的成本估算与优化

---

## 目录

1. [概述](#概述)
2. [成本模型架构](#成本模型架构)
3. [单服务器成本计算](#单服务器成本计算)
4. [互联成本分层模型](#互联成本分层模型)
5. [单 CP 总成本公式](#单-cp-总成本公式)
6. [成本参数参考表](#成本参数参考表)
7. [成本计算示例](#成本计算示例)
8. [集成到 Tier6+Model 系统](#集成到-tier6model-系统)
9. [成本优化建议](#成本优化建议)

---

## 概述

LLM 推理集群的成本由两大部分构成：

1. **服务器成本**（芯片 + 主板 + 基础设施）
2. **互联成本**（网络交换机 + 线缆 + 光模块）

随着集群规模扩大，**互联成本呈阶梯式增长**，是成本优化的关键因素。

### 核心概念

- **CP (Compute Pool)**: 计算池，指一组协同工作的服务器集群
- **QAM 模组**: 芯片模组单元
- **Lane**: 网络通道单位（112Gbps/lane）
- **互联带宽**: 芯片间通信所需的网络带宽

---

## 成本模型架构

```
┌─────────────────────────────────────────────────────┐
│              LLM 推理集群总成本                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────┐      ┌────────────────────┐ │
│  │  单服务器成本    │  +   │   互联成本         │ │
│  ├──────────────────┤      ├────────────────────┤ │
│  │ • 芯片成本       │      │ • 交换机           │ │
│  │ • 服务器基础成本 │      │ • 线缆/光纤        │ │
│  │ • RDMA 网卡      │      │ • 光模块           │ │
│  └──────────────────┘      └────────────────────┘ │
│           ↓                         ↓              │
│      线性增长               阶梯式增长（分层定价）  │
└─────────────────────────────────────────────────────┘
```

---

## 单服务器成本计算

### 公式

```
单服务器成本 = (A × n + 750) × m + 12000 + 7500
```

### 参数说明

| 参数 | 说明 | 典型值 | 备注 |
|------|------|--------|------|
| `A` | 单芯片成本 | $6303 (B200) | 根据芯片型号变化 |
| `n` | 每个 QAM 模组的芯片数 | 1 | 默认为 1 |
| `m` | 每服务器的 QAM 模组数 | 8 | 一般为 8 个模组/服务器 |
| `12000` | 服务器基础成本 | $12000 | 主板、电源、机箱等 |
| `7500` | RDMA 网络成本 | $7500 | <200Gbps 网络设备 |

### 单服务器成本分解

以 **B200 芯片** 为例：

```
单芯片成本    = $6303
QAM 模组成本  = (6303 × 1 + 750) × 8 = $62,424
服务器基础    = $12,000
RDMA 网络     = $7,500
─────────────────────────────────────
单服务器总成本 = $81,924
```

---

## 互联成本分层模型

互联成本根据 **芯片数量 (CP_num)** 采用阶梯式定价：

### 分层定价表

| 规模 (芯片数) | 互联方案 | 成本 ($/112Gbps lane) | 说明 |
|---------------|----------|----------------------|------|
| **1-2** | PCIe 直连 | **$1** | 直接通过 PCIe 总线 Full-mesh 互联 |
| **8** | Eth Switch | **$55** | 引入以太网交换机（E1.2T 交换机，200万并置，分摊 200000/512/7） |
| **16-32** | Switch + DAC 线缆 | **$70** | 添加 DAC 铜缆（$80/400Gbps，分摊 $55+$15） |
| **32-64** | Switch + AEC 光缆 | **$105** | 添加 AEC 有源光缆（$60/100Gbps，分摊 $55+$50） |
| **64+** | Switch + 光模块 + 光纤 | **$247** | 完整光模块方案（¥4950/800Gbps + ¥800Gbps 光纤） |

### 成本拐点分析

```
  互联成本 ($/lane)
  250 │                                  ┌─ 64+ 芯片 ($247)
      │                                  │
  200 │                                  │
      │                                  │
  150 │                                  │
      │                          ┌───────┘
  100 │                  ┌───────┘ 32-64 芯片 ($105)
      │          ┌───────┘ 16-32 芯片 ($70)
   50 │  ┌───────┘ 8 芯片 ($55)
      │──┘ 1-2 芯片 ($1)
    0 └──────────────────────────────────────────> 芯片数 (CP_num)
        1   8    16   32         64
```

### 关键拐点

1. **8 芯片**: 引入交换机，成本从 $1 → $55（**55x 增长**）
2. **16-32 芯片**: 需要 DAC 扩展，$55 → $70（+27%）
3. **32-64 芯片**: 升级 AEC 光缆，$70 → $105（+50%）
4. **64+ 芯片**: 全光模块方案，$105 → $247（**135% 增长**）

---

## 单 CP 总成本公式

### 完整公式

```
单 CP 成本 = CP_num × 单服务器成本 + (CP_num × m × n) × 单芯互联带宽需求/112Gbps × lane 成本
```

### 简化形式

假设 `m=8`, `n=1`（标准配置）：

```
单 CP 成本 = CP_num × 单服务器成本 + CP_num × 8 × (互联带宽需求/112Gbps) × lane_cost(CP_num)
```

### lane_cost(CP_num) 函数

```python
def lane_cost(cp_num):
    if cp_num < 8:
        return 0        # 小规模无需交换机
    elif cp_num == 8:
        return 55       # 标准交换机
    elif 16 <= cp_num <= 32:
        return 70       # Switch + DAC
    elif cp_num == 64:
        return 105.247  # Switch + AEC
    else:  # cp_num > 64
        return 247      # 全光方案
```

### 互联带宽需求计算

互联带宽取决于 **并行策略**：

- **Tensor Parallelism (TP)**: 需要 All-Reduce 通信，带宽需求高
- **Pipeline Parallelism (PP)**: 仅 P2P 通信，带宽需求中等
- **Data Parallelism (DP)**: 梯度同步，带宽需求低（推理时为 0）

**经验公式**（推理场景）：

```
互联带宽需求 (Gbps) ≈ 2 × 模型大小 (GB) × TP 并行度 / TPOT (ms) × 1000
```

示例：
- 模型大小：671B 参数 × 2 字节 = 1342 GB
- TP = 8
- TPOT = 0.07 ms

```
带宽需求 = 2 × 1342 × 8 / 0.07 × 1000 ≈ 307,200 Gbps
lanes 数量 = 307,200 / 112 ≈ 2,743 lanes
```

---

## 成本参数参考表

### 芯片成本 (A)

| 芯片型号 | 单价 ($) | 算力 (TFLOPS FP16) | 显存 (GB) | 备注 |
|----------|---------|-------------------|----------|------|
| B200 | 6,303 | 2,000 | 80 | NVIDIA Blackwell |
| H100 SXM | 4,500 | 1,979 | 80 | NVIDIA Hopper |
| H800 | 3,800 | 1,979 | 80 | 中国版 H100 |
| SG2262 | 2,500 | 2,000 | 80 | 国产芯片 |

### 固定成本

| 项目 | 成本 ($) | 说明 |
|------|---------|------|
| 服务器基础成本 | 12,000 | 主板、电源、机箱、风扇等 |
| RDMA 网络 (<200Gbps) | 7,500 | RDMA 网卡 + 线缆 |
| QAM 模组附加成本 | 750 | 每个模组的额外成本 |

### 互联设备成本

| 设备 | 单价 ($) | 规格 | 备注 |
|------|---------|------|------|
| E1.2T 交换机 | 2,000,000 | 512端口 | 分摊到 7 年 → $200,000/512/7 ≈ $55/端口 |
| DAC 线缆 | 80 | 400Gbps | 铜缆，短距离 |
| AEC 有源光缆 | 60 | 100Gbps | 中距离 |
| 光模块 | ¥4,950 | 800Gbps | 长距离（约 $700） |
| 光纤 | ¥800 | 800Gbps | 按距离计费（约 $115） |

---

## 成本计算示例

### 示例 1: 8 芯片集群（单机配置）

**配置**：
- 芯片：B200
- 数量：8 芯片（1 台服务器）
- 并行策略：TP=8

**计算**：

```
单服务器成本 = (6303 × 1 + 750) × 8 + 12000 + 7500 = $81,924

互联成本:
  lanes = 假设 2,743 lanes
  lane_cost = $55 (8 芯片档位)
  互联成本 = 8 × 8 × 2,743 × 55 = $9,664,640

总成本 = 81,924 + 9,664,640 ≈ $9.75M
```

### 示例 2: 64 芯片集群（中型集群）

**配置**：
- 芯片：H100
- 数量：64 芯片（8 台服务器）
- 并行策略：TP=8, PP=8

**计算**：

```
单服务器成本 = (4500 × 1 + 750) × 8 + 12000 + 7500 = $61,500
服务器总成本 = 8 × 61,500 = $492,000

互联成本:
  lanes = 假设 2,743 lanes (TP=8 部分)
  lane_cost = $105.247 (64 芯片档位)
  互联成本 = 64 × 8 × 2,743 × 105.247 ≈ $148M

总成本 = 492,000 + 148,000,000 ≈ $148.5M
```

> **注意**: 实际互联成本需根据具体拓扑和并行策略精确计算。

---

## 集成到 Tier6+Model 系统

### 集成方案

在现有的部署分析系统中添加 **成本评估模块**：

```python
# backend/llm_simulator/evaluators/cost_evaluator.py

class CostEvaluator:
    """LLM 推理集群成本评估器"""

    def __init__(self, chip_prices: dict):
        self.chip_prices = chip_prices  # 芯片价格表

    def calculate_server_cost(self, chip_type: str, m: int = 8, n: int = 1) -> float:
        """计算单服务器成本"""
        A = self.chip_prices.get(chip_type, 6303)  # 默认 B200
        return (A * n + 750) * m + 12000 + 7500

    def get_lane_cost(self, cp_num: int) -> float:
        """获取互联 lane 成本（分层定价）"""
        if cp_num < 8:
            return 0
        elif cp_num == 8:
            return 55
        elif 16 <= cp_num <= 32:
            return 70
        elif cp_num == 64:
            return 105.247
        else:
            return 247

    def estimate_interconnect_bandwidth(
        self,
        model_size_gb: float,
        tp: int,
        tpot_ms: float
    ) -> float:
        """估算互联带宽需求 (Gbps)"""
        return 2 * model_size_gb * tp / tpot_ms * 1000

    def calculate_total_cost(
        self,
        cp_num: int,
        chip_type: str,
        model_size_gb: float,
        tp: int,
        tpot_ms: float,
        m: int = 8,
        n: int = 1
    ) -> dict:
        """计算 CP 总成本"""
        # 服务器成本
        server_cost = self.calculate_server_cost(chip_type, m, n)
        total_server_cost = cp_num * server_cost

        # 互联成本
        bandwidth_gbps = self.estimate_interconnect_bandwidth(
            model_size_gb, tp, tpot_ms
        )
        lanes = bandwidth_gbps / 112
        lane_cost = self.get_lane_cost(cp_num)
        interconnect_cost = cp_num * m * n * lanes * lane_cost

        total_cost = total_server_cost + interconnect_cost

        return {
            "server_cost": total_server_cost,
            "interconnect_cost": interconnect_cost,
            "total_cost": total_cost,
            "bandwidth_gbps": bandwidth_gbps,
            "lanes": lanes,
            "lane_cost": lane_cost,
        }
```

### 在评估结果中添加成本指标

修改 `backend/llm_simulator/tasks/deployment.py`：

```python
# 在 _convert_to_dstpu_format 中添加
cost_evaluator = CostEvaluator(chip_prices={"B200": 6303, "H100": 4500})

cost_result = cost_evaluator.calculate_total_cost(
    cp_num=chips,
    chip_type=chip_hw.get("chip_type", "B200"),
    model_size_gb=model_config.get("num_parameters") * 2 / 1e9,  # 参数量转 GB
    tp=parallelism["tp"],
    tpot_ms=avg_tpot / 1000.0,
)

return {
    ...
    "cost": {
        "total_cost_usd": cost_result["total_cost"],
        "server_cost_usd": cost_result["server_cost"],
        "interconnect_cost_usd": cost_result["interconnect_cost"],
        "cost_per_chip_usd": cost_result["total_cost"] / chips,
        "cost_per_million_tokens_usd": cost_result["total_cost"] / (tps * 1e6),  # $/M tokens
    },
    ...
}
```

### 前端展示

在 `AnalysisResultDisplay` 中添加成本卡片：

```tsx
{/* 推理成本 */}
<div className="mb-3">
  <span className="text-[13px] font-medium block mb-2">推理成本</span>
  <div className="grid grid-cols-2 gap-2">
    <div style={metricCardStyle()}>
      <span className="text-[13px]">总成本</span>
      <div className="text-lg font-semibold mt-1">
        ${(cost?.total_cost_usd / 1e6).toFixed(2)}M
      </div>
    </div>
    <div style={metricCardStyle()}>
      <span className="text-[13px]">$/M tokens</span>
      <div className="text-lg font-semibold mt-1">
        ${cost?.cost_per_million_tokens_usd.toFixed(3)}
      </div>
    </div>
  </div>
</div>
```

---

## 成本优化建议

### 1. 避开成本拐点

**策略**: 尽量选择拐点前的规模
- ❌ **不推荐**: 9-15 芯片（已引入交换机但规模不足）
- ✅ **推荐**: 8 芯片（单机），32 芯片（小集群），64 芯片（中集群）

### 2. 优化并行策略

**TP vs PP 的成本权衡**：

- **TP（Tensor Parallelism）**: 互联带宽需求高（All-Reduce），成本高
- **PP（Pipeline Parallelism）**: 互联带宽需求低（P2P），成本低

**建议**：
```
TP ≤ 8 (单机)，PP 用于跨机扩展
```

### 3. 选择合适的芯片型号

**性价比排序**（推理场景）：

```
SG2262 > H800 > H100 > B200
(成本低)              (性能高)
```

根据 **预算** 和 **性能要求** 选择：
- **成本敏感**: SG2262 / H800
- **性能优先**: B200 / H100

### 4. 批量采购降低 lane 成本

**策略**: 与多个项目共享交换机设备
- 交换机分摊成本假设 7 年折旧
- 提高设备利用率可进一步降低单 lane 成本

### 5. 动态扩展策略

**分阶段部署**：
```
Phase 1: 8 芯片（验证）      → 成本低
Phase 2: 32 芯片（小规模）   → 成本可控
Phase 3: 64+ 芯片（大规模）  → 评估 ROI
```

---

## 附录

### 参考文献

1. **NVIDIA Blackwell 架构白皮书** - GPU 规格与定价
2. **以太网交换机市场报告 2025** - 网络设备成本分析
3. **LLM 推理优化最佳实践** - 并行策略与成本权衡

### 更新日志

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2026-01-28 | 初始版本，包含完整成本模型 |

### 联系方式

- **项目**: Tier6+Model
- **GitHub**: https://github.com/your-repo/tier6-model
- **问题反馈**: 请在 GitHub Issues 中提交

---

**📌 重要提示**: 本文档中的价格仅供参考，实际成本会因供应商、地区、采购量等因素波动。使用前请根据最新市场价格更新参数。
