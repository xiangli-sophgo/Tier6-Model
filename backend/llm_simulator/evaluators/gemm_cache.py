"""
GEMM 持久化缓存管理器

功能：
1. 将 GEMM 搜索结果持久化到本地 JSON 文件
2. 支持跨运行复用缓存
3. 架构指纹匹配（确保缓存正确性）
4. 版本控制和兼容性检查
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING
from datetime import datetime

from .arch_config import AcceleratorMicroArch

if TYPE_CHECKING:
    from .gemm_eval import GEMMResult

logger = logging.getLogger(__name__)

# 缓存版本（评估逻辑变化时递增）
CACHE_VERSION = "1.0.0"
CACHE_FORMAT_VERSION = "2025.01"


class GEMMPersistentCache:
    """GEMM 持久化缓存管理器

    设计原则：
    - 缓存键必须包含：形状、数据类型、架构指纹、搜索模式
    - 使用 JSON 格式（易读、跨平台、无依赖）
    - 按架构指纹分文件存储
    - 自动版本检查和兼容性验证
    """

    def __init__(self, arch: AcceleratorMicroArch):
        """
        初始化持久化缓存

        Args:
            arch: 硬件微架构配置
        """
        self.arch = arch
        self.arch_fingerprint = self._compute_arch_fingerprint()

        # 缓存文件路径
        cache_dir = Path(__file__).parent.parent.parent / ".cache" / "gemm"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / f"gemm_cache_{self.arch_fingerprint}.json"

        # 内存缓存（快速访问）
        self._cache: Dict[str, "GEMMResult"] = {}

        # 统计信息
        self._cache_hits = 0
        self._cache_misses = 0

        # 加载持久化缓存
        self._load_from_disk()

    def _compute_arch_fingerprint(self) -> str:
        """
        计算硬件架构指纹

        包含所有影响 GEMM 性能的架构参数

        Returns:
            8 字符的 MD5 hash（用于文件名）
        """
        key_params = {
            "name": self.arch.name,
            "num_cores": self.arch.num_cores,
            "cube_m": self.arch.cube_m,
            "cube_n": self.arch.cube_n,
            "cube_k": self.arch.cube_k,
            "sram_kb": self.arch.sram_size_bytes // 1024,
            "freq_ghz": self.arch.freq_ghz,
            "dram_bw_gbps": self.arch.dram_bandwidth_bytes / 1e9,
            "lane_num": self.arch.lane_num,
            "align_bytes": self.arch.align_bytes,
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
    ) -> str:
        """
        构造完整缓存键

        包含：形状、数据类型、架构指纹、搜索模式

        Returns:
            缓存键的 hash 字符串
        """
        cache_key_tuple = (
            G, M, K, N,
            input_dtype, output_dtype,
            self.arch_fingerprint,
            enable_tile_search,
            enable_partition_search
        )

        # 生成 hash（用于快速索引和文件存储）
        key_str = json.dumps(cache_key_tuple, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def get(
        self,
        G: int, M: int, K: int, N: int,
        input_dtype: str, output_dtype: str,
        enable_tile_search: bool,
        enable_partition_search: bool
    ) -> Optional["GEMMResult"]:
        """
        从缓存中获取 GEMM 评估结果

        Args:
            G, M, K, N: GEMM 形状参数
            input_dtype: 输入数据类型
            output_dtype: 输出数据类型
            enable_tile_search: 是否启用 tile 搜索
            enable_partition_search: 是否启用分区搜索

        Returns:
            缓存的评估结果，如果未命中则返回 None
        """
        cache_key = self._make_cache_key(
            G, M, K, N, input_dtype, output_dtype,
            enable_tile_search, enable_partition_search
        )

        result = self._cache.get(cache_key)
        if result is not None:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        return result

    def put(
        self,
        G: int, M: int, K: int, N: int,
        input_dtype: str, output_dtype: str,
        enable_tile_search: bool,
        enable_partition_search: bool,
        result: "GEMMResult",
        search_time_ms: float
    ):
        """
        保存 GEMM 评估结果到缓存

        Args:
            G, M, K, N: GEMM 形状参数
            input_dtype: 输入数据类型
            output_dtype: 输出数据类型
            enable_tile_search: 是否启用 tile 搜索
            enable_partition_search: 是否启用分区搜索
            result: GEMM 评估结果
            search_time_ms: 搜索耗时（毫秒）
        """
        cache_key = self._make_cache_key(
            G, M, K, N, input_dtype, output_dtype,
            enable_tile_search, enable_partition_search
        )

        # 保存到内存
        self._cache[cache_key] = result

        # 保存到磁盘
        self._save_to_disk(
            cache_key, G, M, K, N, input_dtype, output_dtype,
            enable_tile_search, enable_partition_search,
            result, search_time_ms
        )

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
            for entry_hash, entry_data in data.get("cache_entries", {}).items():
                try:
                    result = self._reconstruct_result(entry_data["result"])
                    self._cache[entry_hash] = result
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"加载缓存条目失败 {entry_hash}: {e}")

            logger.info(f"✅ 成功加载 {loaded_count} 条 GEMM 缓存记录 (架构: {self.arch.name})")

        except Exception as e:
            logger.error(f"加载缓存失败: {e}，将使用空缓存")

    def _save_to_disk(
        self,
        cache_key: str,
        G: int, M: int, K: int, N: int,
        input_dtype: str, output_dtype: str,
        enable_tile_search: bool,
        enable_partition_search: bool,
        result: "GEMMResult",
        search_time_ms: float
    ):
        """保存单条记录到磁盘（增量更新）"""
        try:
            # 读取现有数据
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = self._create_empty_cache_file()

            # 添加新条目
            data["cache_entries"][cache_key] = {
                "shape": {"G": G, "M": M, "K": K, "N": N},
                "dtypes": {"input": input_dtype, "output": output_dtype},
                "search_mode": {
                    "tile_search": enable_tile_search,
                    "partition_search": enable_partition_search
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
            data["statistics"]["total_search_time_hours"] = (
                data["statistics"].get("total_search_time_hours", 0) +
                search_time_ms / 1000 / 3600
            )

            # 原子写入（先写临时文件，再替换）
            tmp_file = self.cache_file.with_suffix('.tmp')
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp_file.replace(self.cache_file)

        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def _create_empty_cache_file(self) -> dict:
        """创建空缓存文件结构"""
        return {
            "version": CACHE_VERSION,
            "cache_format_version": CACHE_FORMAT_VERSION,
            "arch_fingerprint": self.arch_fingerprint,
            "architecture": {
                "name": self.arch.name,
                "num_cores": self.arch.num_cores,
                "sram_kb": self.arch.sram_size_bytes // 1024,
                "cube_m": self.arch.cube_m,
                "cube_n": self.arch.cube_n,
                "cube_k": self.arch.cube_k,
                "freq_ghz": self.arch.freq_ghz,
                "dram_bw_gbps": self.arch.dram_bandwidth_bytes / 1e9,
            },
            "cache_entries": {},
            "statistics": {
                "total_entries": 0,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_search_time_hours": 0.0,
                "cache_hits": 0,
                "cache_misses": 0
            }
        }

    def _reconstruct_result(self, result_data: dict) -> "GEMMResult":
        """从 JSON 数据重建 GEMMResult 对象"""
        from .gemm_eval import GEMMResult

        return GEMMResult(
            latency_us=result_data["latency_us"],
            compute_time_us=result_data["compute_time_us"],
            memory_time_us=result_data["memory_time_us"],
            flops=result_data["flops"],
            dram_traffic_bytes=result_data["dram_traffic_bytes"],
            arch_utilization=result_data["arch_utilization"],
            effective_utilization=result_data["effective_utilization"],
            best_tile=tuple(result_data["best_tile"]),
            best_loop_order=result_data["best_loop_order"],
            best_partition=tuple(result_data["best_partition"])
        )

    def print_cache_stats(self):
        """打印缓存统计信息"""
        total_queries = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_queries if total_queries > 0 else 0

        print(f"\n📊 GEMM 持久化缓存统计:")
        print(f"  缓存文件: {self.cache_file}")
        print(f"  架构指纹: {self.arch_fingerprint} ({self.arch.name})")
        print(f"  总条目数: {len(self._cache)}")
        print(f"  缓存命中: {self._cache_hits}")
        print(f"  缓存未命中: {self._cache_misses}")
        print(f"  命中率: {hit_rate*100:.1f}%")
