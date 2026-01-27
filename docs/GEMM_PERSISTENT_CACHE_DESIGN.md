# GEMM 持久化缓存设计方案

**文档版本**: v1.0
**创建日期**: 2025-01-27
**状态**: 设计提案

## 背景

### 当前问题

GEMM 评估是模拟器最耗时的操作：
- **单个 GEMM 形状评估时间**：100-220 秒（启用分区搜索）/ <1 秒（禁用分区搜索）
- **DeepSeek V3 模型**：约 60 层，每层 10-20 个 GEMM → 总共 600-1200 个唯一形状
- **首次运行总耗时**：启用完整搜索时可达 **16+ 小时**
- **缓存生命周期**：仅存在于单次运行，进程结束即丢失

### 核心痛点

1. **重复计算浪费**：同一硬件配置、同一模型，每次启动都需要重新搜索
2. **无法积累优化**：用户测试多个配置时，已搜索的结果无法复用
3. **开发调试缓慢**：代码修改后重启，所有搜索结果丢失

---

## 设计目标

### 核心目标

- ✅ 将 GEMM 搜索结果持久化到本地文件
- ✅ 支持跨运行复用缓存
- ✅ 每次运行可增量丰富缓存内容
- ✅ 确保缓存正确性（架构匹配、版本控制）

### 性能目标

- **首次运行**：正常搜索并建立缓存（16 小时）
- **第二次运行**：100% 缓存命中，耗时 <1 秒
- **收益比**：60000 倍提升 ✅

---

## 架构设计

### 1. 缓存键设计（关键）

#### 为什么缓存键设计至关重要？

同一个 GEMM 形状，在不同条件下结果会**完全不同**，必须精确区分：

```python
缓存键组成 = {
    # ========== GEMM 形状参数 ==========
    "G": 1,          # Batch/Group 维度
    "M": 1,          # 输出行数
    "K": 4096,       # 累加维度
    "N": 7168,       # 输出列数
    "input_dtype": "bf16",   # 输入数据类型
    "output_dtype": "bf16",  # 输出数据类型

    # ========== 硬件架构参数（必须！）==========
    "chip_type": "SG2262",
    "num_cores": 64,           # 核心数影响分区策略
    "sram_kb": 8192,           # SRAM 大小影响 tile 大小
    "cube_m": 16,              # 计算单元参数
    "cube_n": 16,
    "cube_k": 32,
    "tflops_int8": 256,        # 算力影响计算时间

    # ========== 搜索模式参数（影响结果！）==========
    "enable_tile_search": False,      # tile 搜索开关
    "enable_partition_search": False  # 分区搜索开关
}
```

#### 为什么硬件架构必须包含？

| 参数变化 | 影响 | 示例 |
|---------|------|------|
| `num_cores: 32 → 64` | 分区策略完全不同 | (1,4,8,1) vs (1,8,8,1) |
| `sram_kb: 4096 → 8192` | 可用的 tile 大小范围 | (16,16,32) vs (32,32,64) |
| `tflops_int8: 128 → 256` | 计算时间相差 2 倍 | 100μs vs 50μs |

❌ **错误示例**：使用 `(G, M, K, N)` 作为缓存键
- 64 核和 32 核芯片会使用相同缓存 → **结果错误！**

✅ **正确示例**：包含架构指纹
```python
cache_key = (G, M, K, N, input_dtype, output_dtype,
             arch_fingerprint, enable_tile_search, enable_partition_search)
```

#### 为什么搜索模式必须包含？

| 模式 | Tile 结果 | 分区结果 | 延迟 |
|------|----------|---------|------|
| `tile=F, partition=F` | 固定 (16,16,32) | 固定 (1,8,8,1) | 150μs |
| `tile=T, partition=F` | 搜索最优 (32,64,64) | 固定 (1,8,8,1) | 120μs |
| `tile=T, partition=T` | 搜索最优 (32,64,64) | 搜索最优 (1,4,16,1) | 100μs |

不同搜索模式的结果**不能混用**！

---

### 2. 文件格式选择

#### 格式对比

| 格式 | 优点 | 缺点 | 适用规模 | 推荐度 |
|------|------|------|---------|--------|
| **JSON** | 易读、跨平台、可手动编辑、无依赖 | 性能一般、文件较大 | <1000 条记录 | ✅ 推荐 |
| **Pickle** | Python 原生、快速 | 不跨版本、不可读、安全风险 | 任意 | ❌ 不推荐 |
| **SQLite** | 结构化、支持查询、并发安全 | 需要 SQL、略重 | >10000 条 | ⚠️ 可选 |
| **MessagePack** | 紧凑、快速 | 需要额外依赖、不可读 | 任意 | ⚠️ 可选 |

#### 推荐方案：JSON

**理由**：
- DeepSeek V3：约 1200 个唯一 GEMM 形状
- 每条记录约 200 字节
- 总文件大小：<500KB
- 加载时间：5-10ms（完全可接受）

**文件路径规范**：
```
backend/.cache/gemm/gemm_cache_{arch_fingerprint}.json
```

示例：
```
backend/.cache/gemm/gemm_cache_a3f5b9c2.json  # SG2262_64cores_8192kb
backend/.cache/gemm/gemm_cache_d1e8c4a7.json  # SG2260E_32cores_4096kb
```

---

### 3. 缓存文件结构

```json
{
  "version": "1.0.0",
  "cache_format_version": "2025.01",
  "arch_fingerprint": "a3f5b9c2",

  "architecture": {
    "chip_type": "SG2262",
    "num_cores": 64,
    "sram_kb": 8192,
    "cube_m": 16,
    "cube_n": 16,
    "cube_k": 32,
    "tflops_int8": 256
  },

  "cache_entries": {
    "hash_12345abc": {
      "shape": {
        "G": 1,
        "M": 1,
        "K": 4096,
        "N": 7168
      },
      "dtypes": {
        "input": "bf16",
        "output": "bf16"
      },
      "search_mode": {
        "tile_search": false,
        "partition_search": false
      },

      "result": {
        "latency_us": 123.45,
        "compute_time_us": 100.0,
        "memory_time_us": 23.45,
        "flops": 116391936,
        "dram_traffic_bytes": 123456,
        "arch_utilization": 0.85,
        "effective_utilization": 0.75,
        "best_tile": [16, 16, 32],
        "best_loop_order": "mnk",
        "best_partition": [1, 8, 8, 1]
      },

      "metadata": {
        "timestamp": "2025-01-27T10:30:00Z",
        "search_time_ms": 115000,
        "num_searched_partitions": 84,
        "num_searched_tiles": 12
      }
    }
  },

  "statistics": {
    "total_entries": 1234,
    "created_at": "2025-01-20T08:00:00Z",
    "last_updated": "2025-01-27T10:30:00Z",
    "total_search_time_hours": 16.7,
    "cache_hits": 5678,
    "cache_misses": 1234
  }
}
```

#### 字段说明

**顶层字段**：
- `version`: 代码版本，用于缓存失效判断
- `cache_format_version`: 缓存文件格式版本
- `arch_fingerprint`: 架构指纹（MD5 hash）
- `architecture`: 完整架构参数（用于调试和验证）

**缓存条目**：
- `hash_12345abc`: 缓存键的 hash 值（用于快速索引）
- `shape`: GEMM 形状参数
- `dtypes`: 数据类型
- `search_mode`: 搜索模式配置
- `result`: 评估结果（核心数据）
- `metadata`: 元数据（搜索耗时、时间戳等）

---

## 实现方案

### 核心实现逻辑

```python
# backend/llm_simulator/evaluators/gemm_cache.py

import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict

CACHE_VERSION = "1.0.0"
CACHE_FORMAT_VERSION = "2025.01"

class GEMMPersistentCache:
    """GEMM 持久化缓存管理器"""

    def __init__(self, arch: AcceleratorMicroArch):
        self.arch = arch
        self.arch_fingerprint = self._compute_arch_fingerprint()

        # 缓存文件路径
        cache_dir = Path("backend/.cache/gemm")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / f"gemm_cache_{self.arch_fingerprint}.json"

        # 内存缓存
        self._cache: Dict[Tuple, GEMMResult] = {}

        # 加载持久化缓存
        self._load_from_disk()

    def _compute_arch_fingerprint(self) -> str:
        """
        计算硬件架构指纹

        包含所有影响 GEMM 性能的架构参数
        """
        key_params = {
            "chip_type": self.arch.chip_type,
            "num_cores": self.arch.num_cores,
            "sram_kb": self.arch.sram_kb,
            "cube_m": self.arch.cube_m,
            "cube_n": self.arch.cube_n,
            "cube_k": self.arch.cube_k,
            "tflops_int8": self.arch.tflops_int8,
        }

        # 生成确定性的 hash
        key_str = json.dumps(key_params, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:8]

    def _make_cache_key(
        self,
        G: int, M: int, K: int, N: int,
        input_dtype: str, output_dtype: str,
        enable_tile_search: bool,
        enable_partition_search: bool
    ) -> Tuple:
        """
        构造完整缓存键

        包含：形状、数据类型、架构指纹、搜索模式
        """
        return (
            G, M, K, N,
            input_dtype, output_dtype,
            self.arch_fingerprint,
            enable_tile_search,
            enable_partition_search
        )

    def get(self, cache_key: Tuple) -> Optional[GEMMResult]:
        """从内存缓存中获取结果"""
        return self._cache.get(cache_key)

    def put(self, cache_key: Tuple, result: GEMMResult, search_time_ms: float):
        """
        保存结果到内存缓存，并异步写入磁盘

        Args:
            cache_key: 缓存键
            result: GEMM 评估结果
            search_time_ms: 搜索耗时（毫秒）
        """
        # 1. 保存到内存
        self._cache[cache_key] = result

        # 2. 保存到磁盘（异步或批量）
        self._save_to_disk(cache_key, result, search_time_ms)

    def _load_from_disk(self):
        """从磁盘加载缓存"""
        if not self.cache_file.exists():
            logger.info(f"缓存文件不存在，将创建新缓存: {self.cache_file}")
            return

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 版本检查
            if data.get("version") != CACHE_VERSION:
                logger.warning(
                    f"缓存版本不匹配 (文件: {data.get('version')}, "
                    f"当前: {CACHE_VERSION})，忽略旧缓存"
                )
                return

            # 架构指纹检查
            if data.get("arch_fingerprint") != self.arch_fingerprint:
                logger.warning(
                    f"架构指纹不匹配 (文件: {data.get('arch_fingerprint')}, "
                    f"当前: {self.arch_fingerprint})，忽略缓存"
                )
                return

            # 加载缓存条目
            loaded_count = 0
            for entry_data in data["cache_entries"].values():
                cache_key = self._reconstruct_cache_key(entry_data)
                result = self._reconstruct_result(entry_data["result"])
                self._cache[cache_key] = result
                loaded_count += 1

            logger.info(f"✅ 成功加载 {loaded_count} 条 GEMM 缓存记录")

        except Exception as e:
            logger.error(f"加载缓存失败: {e}，将使用空缓存")

    def _save_to_disk(self, cache_key: Tuple, result: GEMMResult, search_time_ms: float):
        """保存单条记录到磁盘（增量更新）"""
        # 读取现有数据
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = self._create_empty_cache_file()

        # 添加新条目
        entry_hash = hashlib.md5(str(cache_key).encode()).hexdigest()[:12]

        G, M, K, N, in_dtype, out_dtype, arch_fp, tile_search, part_search = cache_key

        data["cache_entries"][entry_hash] = {
            "shape": {"G": G, "M": M, "K": K, "N": N},
            "dtypes": {"input": in_dtype, "output": out_dtype},
            "search_mode": {
                "tile_search": tile_search,
                "partition_search": part_search
            },
            "result": {
                "latency_us": result.latency_us,
                "compute_time_us": result.compute_time_us,
                "memory_time_us": result.memory_time_us,
                "flops": result.flops,
                "dram_traffic_bytes": result.dram_traffic_bytes,
                "arch_utilization": result.arch_utilization,
                "effective_utilization": result.effective_utilization,
                "best_tile": list(result.best_tile),
                "best_loop_order": result.best_loop_order,
                "best_partition": list(result.best_partition)
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "search_time_ms": search_time_ms
            }
        }

        # 更新统计信息
        data["statistics"]["total_entries"] = len(data["cache_entries"])
        data["statistics"]["last_updated"] = datetime.now().isoformat()

        # 写入文件（原子操作）
        tmp_file = self.cache_file.with_suffix('.tmp')
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_file.replace(self.cache_file)

    def _create_empty_cache_file(self) -> dict:
        """创建空缓存文件结构"""
        return {
            "version": CACHE_VERSION,
            "cache_format_version": CACHE_FORMAT_VERSION,
            "arch_fingerprint": self.arch_fingerprint,
            "architecture": {
                "chip_type": self.arch.chip_type,
                "num_cores": self.arch.num_cores,
                "sram_kb": self.arch.sram_kb,
                "cube_m": self.arch.cube_m,
                "cube_n": self.arch.cube_n,
                "cube_k": self.arch.cube_k,
                "tflops_int8": self.arch.tflops_int8
            },
            "cache_entries": {},
            "statistics": {
                "total_entries": 0,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "cache_hits": 0,
                "cache_misses": 0
            }
        }
```

### 集成到 GEMMEvaluator

```python
# backend/llm_simulator/evaluators/gemm_eval.py

class GEMMEvaluator:
    def __init__(self, arch: AcceleratorMicroArch, enable_partition_search: bool = True):
        self.arch = arch
        self.enable_partition_search = enable_partition_search

        # 持久化缓存管理器
        self.persistent_cache = GEMMPersistentCache(arch)

        # 内存缓存（快速访问）
        self._cache = self.persistent_cache._cache

    def evaluate(
        self,
        G: int, M: int, K: int, N: int,
        input_dtype: str = "bf16",
        output_dtype: str = "bf16",
        use_multiprocess: bool = True,
    ) -> GEMMResult:
        # 1. 构造缓存键（包含搜索模式）
        cache_key = self.persistent_cache._make_cache_key(
            G, M, K, N, input_dtype, output_dtype,
            not hasattr(self, 'fast_mode') or not self.fast_mode,  # tile_search
            self.enable_partition_search
        )

        # 2. 查缓存
        cached_result = self.persistent_cache.get(cache_key)
        if cached_result is not None:
            self._cache_hits += 1
            return cached_result

        # 3. 缓存未命中，执行搜索
        import time
        search_start = time.time()

        result = self._perform_search(
            G, M, K, N, input_dtype, output_dtype, use_multiprocess
        )

        search_time_ms = (time.time() - search_start) * 1000

        # 4. 保存到缓存
        self._cache_misses += 1
        self.persistent_cache.put(cache_key, result, search_time_ms)

        return result
```

---

## 关键问题和解决方案

### 1. 并发写入冲突

**问题**：多个任务同时运行，同时写缓存文件

**解决方案**：

| 方案 | 实现 | 优点 | 缺点 | 推荐度 |
|------|------|------|------|--------|
| **文件锁** | `fcntl.flock` (Linux) 或 `portalocker` (跨平台) | 简单、可靠 | Windows 需要额外库 | ✅ 推荐 |
| **进程独立缓存** | 每个进程写 `cache_{pid}.json`，定期合并 | 无锁开销 | 需要合并逻辑 | ⚠️ 备选 |
| **SQLite** | 使用 SQLite 的内置并发控制 | 并发安全、支持查询 | 略重、需要迁移 | ⚠️ 未来可选 |

**推荐实现**：文件锁

```python
import portalocker  # pip install portalocker

def _save_to_disk(self, ...):
    # 加锁写入
    with portalocker.Lock(self.cache_file, 'a+', timeout=10) as f:
        f.seek(0)
        data = json.load(f) if f.read() else self._create_empty_cache_file()

        # 更新数据
        data["cache_entries"][entry_hash] = ...

        # 原子写入
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2)
```

---

### 2. 缓存失效策略

**问题**：代码逻辑升级，旧缓存结果不准确

**解决方案**：版本控制 + 自动清理

```python
# 在代码中定义当前版本
CACHE_VERSION = "1.0.0"  # 评估逻辑变化时递增

def _load_from_disk(self):
    data = json.load(f)

    if data.get("version") != CACHE_VERSION:
        logger.warning(
            f"⚠️  缓存版本不匹配 ({data.get('version')} vs {CACHE_VERSION})，"
            f"旧缓存将被忽略"
        )

        # 可选：自动清理旧缓存
        # self.cache_file.unlink()

        return
```

**版本升级规则**：
- **主版本**（1.x → 2.x）：评估算法重大变化 → 清空所有缓存
- **次版本**（1.0 → 1.1）：优化改进 → 保留旧缓存，但标记为"待验证"
- **补丁版本**（1.0.0 → 1.0.1）：Bug 修复 → 兼容旧缓存

---

### 3. 缓存文件膨胀

**问题**：用户测试很多不同配置，缓存文件变大

**解决方案**：

1. **按架构分文件**（已实现）
   ```
   gemm_cache_a3f5b9c2.json  # SG2262_64cores
   gemm_cache_d1e8c4a7.json  # SG2260E_32cores
   ```

2. **限制单文件大小**
   ```python
   MAX_CACHE_ENTRIES = 10000  # 约 5MB

   if len(data["cache_entries"]) > MAX_CACHE_ENTRIES:
       # 删除最旧的 10% 条目
       self._prune_old_entries(data, ratio=0.1)
   ```

3. **定期清理**
   ```python
   def _prune_old_entries(self, data: dict, max_age_days: int = 90):
       """删除超过 N 天未使用的条目"""
       cutoff = datetime.now() - timedelta(days=max_age_days)

       data["cache_entries"] = {
           k: v for k, v in data["cache_entries"].items()
           if datetime.fromisoformat(v["metadata"]["timestamp"]) > cutoff
       }
   ```

---

### 4. 搜索模式不一致

**问题**：用户先用 `partition_search=True` 跑一次，再用 `False` 跑，结果混淆

**解决方案**：缓存键必须包含搜索模式

```python
# ✅ 正确：搜索模式是缓存键的一部分
cache_key = (G, M, K, N, input_dtype, output_dtype,
             arch_fingerprint,
             enable_tile_search,      # ← 必须包含
             enable_partition_search) # ← 必须包含

# ❌ 错误：缺少搜索模式
cache_key = (G, M, K, N, input_dtype, output_dtype)
```

---

## 性能评估

### 加载性能

| 缓存大小 | 条目数 | 文件大小 | 加载时间 | 评估 |
|---------|-------|---------|---------|------|
| 小型 | 500 | 250 KB | 2-5 ms | ✅ 优秀 |
| 中型 | 2000 | 1 MB | 8-15 ms | ✅ 良好 |
| 大型 | 10000 | 5 MB | 40-80 ms | ⚠️ 可接受 |
| 超大 | 50000 | 25 MB | 200-400 ms | ❌ 需优化 |

**建议**：单文件限制在 10000 条记录内（约 5MB）

### 保存性能

**同步保存**（每次新增都写文件）：
- 单次保存时间：5-10 ms
- 影响：可接受（搜索本身耗时 100+ 秒）

**批量保存**（积累 N 条后一次性写入）：
- 批量大小：10-50 条
- 单次保存时间：10-20 ms
- 风险：进程崩溃时丢失未保存的记录

**推荐**：同步保存（可靠性优先）

### 收益分析

**场景 1：启用完整搜索**
- 首次运行：600 个形状 × 120 秒/形状 = **72000 秒（20 小时）**
- 第二次运行：600 个形状 × 0.001 秒/形状 = **0.6 秒**
- 收益：**120000 倍提升**

**场景 2：禁用分区搜索**
- 首次运行：600 个形状 × 0.5 秒/形状 = **300 秒（5 分钟）**
- 第二次运行：600 个形状 × 0.001 秒/形状 = **0.6 秒**
- 收益：**500 倍提升**

---

## 实施计划

### Phase 1：基础功能（优先级：高）

- [ ] 实现 `GEMMPersistentCache` 类
- [ ] 架构指纹计算
- [ ] 缓存文件的加载和保存
- [ ] 搜索模式包含在缓存键中
- [ ] 版本控制和兼容性检查

**预计工作量**：4-6 小时
**预期效果**：实现跨运行缓存复用

---

### Phase 2：稳定性优化（优先级：中）

- [ ] 文件锁（并发安全）
- [ ] 原子写入（防止损坏）
- [ ] 异常处理和降级
- [ ] 缓存统计和日志
- [ ] 单元测试

**预计工作量**：2-3 小时
**预期效果**：提升鲁棒性

---

### Phase 3：性能优化（优先级：低）

- [ ] 异步保存（避免阻塞主流程）
- [ ] 批量写入优化
- [ ] 缓存预热（预先加载常用形状）
- [ ] 缓存压缩（减小文件大小）

**预计工作量**：2-4 小时
**预期效果**：进一步提升性能

---

### Phase 4：高级特性（可选）

- [ ] 迁移到 SQLite（支持 >10000 条记录）
- [ ] Web 界面查看缓存统计
- [ ] 缓存共享（团队协作）
- [ ] 自动清理和优化

**预计工作量**：8-12 小时
**预期效果**：企业级特性

---

## 测试计划

### 单元测试

```python
def test_arch_fingerprint():
    """测试架构指纹计算"""
    arch1 = AcceleratorMicroArch(chip_type="SG2262", num_cores=64, ...)
    arch2 = AcceleratorMicroArch(chip_type="SG2262", num_cores=64, ...)
    arch3 = AcceleratorMicroArch(chip_type="SG2262", num_cores=32, ...)

    cache1 = GEMMPersistentCache(arch1)
    cache2 = GEMMPersistentCache(arch2)
    cache3 = GEMMPersistentCache(arch3)

    # 相同架构应该有相同指纹
    assert cache1.arch_fingerprint == cache2.arch_fingerprint

    # 不同架构应该有不同指纹
    assert cache1.arch_fingerprint != cache3.arch_fingerprint

def test_cache_persistence():
    """测试缓存持久化"""
    # 1. 创建缓存并保存
    cache = GEMMPersistentCache(arch)
    result = GEMMResult(latency_us=100, ...)
    cache.put(cache_key, result, search_time_ms=50000)

    # 2. 销毁对象，模拟进程重启
    del cache

    # 3. 重新加载，验证数据存在
    cache2 = GEMMPersistentCache(arch)
    loaded_result = cache2.get(cache_key)

    assert loaded_result is not None
    assert loaded_result.latency_us == 100

def test_version_mismatch():
    """测试版本不匹配时的处理"""
    # 创建旧版本缓存文件
    old_cache = {"version": "0.9.0", ...}
    with open(cache_file, 'w') as f:
        json.dump(old_cache, f)

    # 加载缓存
    cache = GEMMPersistentCache(arch)

    # 应该忽略旧缓存，使用空缓存
    assert len(cache._cache) == 0
```

### 集成测试

```python
def test_end_to_end():
    """端到端测试"""
    # 1. 首次运行：无缓存
    evaluator1 = GEMMEvaluator(arch, enable_partition_search=True)
    result1 = evaluator1.evaluate(1, 1, 4096, 7168)
    assert evaluator1._cache_misses == 1

    # 2. 第二次运行：命中缓存
    evaluator2 = GEMMEvaluator(arch, enable_partition_search=True)
    result2 = evaluator2.evaluate(1, 1, 4096, 7168)
    assert evaluator2._cache_hits == 1

    # 3. 结果应该完全相同
    assert result1.latency_us == result2.latency_us
```

---

## 风险评估

| 风险 | 严重性 | 概率 | 缓解措施 |
|------|--------|------|---------|
| 缓存键设计不完整导致错误结果 | 🔴 严重 | 中 | 充分测试、代码审查 |
| 并发写入导致文件损坏 | 🟡 中等 | 低 | 文件锁 + 原子写入 |
| 缓存文件过大影响性能 | 🟢 轻微 | 中 | 限制大小、定期清理 |
| 版本升级后缓存不兼容 | 🟢 轻微 | 高 | 版本控制、自动清理 |

---

## 总结

### 核心价值

✅ **极大提升开发效率**：从 20 小时 → 1 秒
✅ **积累优化成果**：每次运行丰富缓存库
✅ **支持团队协作**：共享缓存文件

### 关键设计

🔑 **缓存键设计**：必须包含架构指纹和搜索模式
🔑 **版本控制**：确保缓存与代码版本匹配
🔑 **并发安全**：文件锁保证多进程安全

### 实施建议

1. **优先实现 Phase 1**：基础功能即可获得 90% 收益
2. **充分测试缓存键设计**：错误的缓存键会导致严重问题
3. **渐进式部署**：先在本地测试，稳定后推广

---

**文档状态**: ✅ 设计完成，待评审
**下一步**: 用户确认设计方案 → 开始 Phase 1 实现
