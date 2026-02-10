"""
Tile 评估持久化缓存

功能：
1. 将 tile 搜索结果持久化到本地 JSON 文件
2. 支持跨运行复用缓存
3. 芯片架构指纹匹配（确保缓存正确性）
4. 版本控制和兼容性检查
5. 命中率统计

迁移自 llm_simulator/evaluators/gemm_cache.py，适配 math_model 架构
"""

import json
import hashlib
import logging
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

# 跨平台文件锁定支持
if sys.platform == 'win32':
    import msvcrt
    _LOCK_AVAILABLE = True
else:
    try:
        import fcntl
        _LOCK_AVAILABLE = True
    except ImportError:
        _LOCK_AVAILABLE = False

logger = logging.getLogger(__name__)

# 缓存版本（评估逻辑变化时递增）
CACHE_VERSION = "2.0.0"


class TilePersistentCache:
    """Tile 评估持久化缓存

    缓存键包含：op_type, shape, dtypes, 芯片指纹, 搜索模式
    按芯片指纹分文件存储（JSON 格式）
    """

    def __init__(
        self,
        chip_name: str,
        core_count: int,
        cube_m: int,
        cube_k: int,
        cube_n: int,
        lane_per_core: int,
        frequency_ghz: float,
        memory_bandwidth_gbps: float,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._chip_params = {
            "name": chip_name,
            "core_count": core_count,
            "cube_m": cube_m,
            "cube_k": cube_k,
            "cube_n": cube_n,
            "lane_per_core": lane_per_core,
            "frequency_ghz": frequency_ghz,
            "memory_bandwidth_gbps": round(memory_bandwidth_gbps, 2),
        }
        self._fingerprint = self._compute_fingerprint()

        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent.parent / ".cache" / "tile"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / f"tile_cache_{self._fingerprint}.json"

        # 内存缓存
        self._cache: Dict[str, dict] = {}

        # 统计
        self._hits = 0
        self._misses = 0

        self._load_from_disk()

    @classmethod
    def from_chip(cls, chip: Any, memory_bandwidth_gbps: float = 0.0,
                  cache_dir: Optional[Path] = None) -> "TilePersistentCache":
        """从 ChipSpecImpl 创建缓存"""
        return cls(
            chip_name=chip.name,
            core_count=chip.core_count,
            cube_m=chip.cube_m,
            cube_k=chip.cube_k,
            cube_n=chip.cube_n,
            lane_per_core=chip.lane_per_core,
            frequency_ghz=chip.frequency_ghz,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            cache_dir=cache_dir,
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _compute_fingerprint(self) -> str:
        key_str = json.dumps(self._chip_params, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:8]

    def _make_key(self, op_type: str, shape: dict, dtypes: tuple,
                  enable_tile_search: bool, enable_partition_search: bool) -> str:
        key_data = (
            op_type,
            tuple(sorted(shape.items())),
            dtypes,
            enable_tile_search,
            enable_partition_search,
        )
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def get(
        self,
        op_type: str,
        shape: dict,
        dtypes: tuple,
        enable_tile_search: bool = True,
        enable_partition_search: bool = False,
    ) -> Optional[dict]:
        """查询缓存

        Returns:
            缓存的 tile_meta dict，未命中返回 None
        """
        key = self._make_key(op_type, shape, dtypes, enable_tile_search, enable_partition_search)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(
        self,
        op_type: str,
        shape: dict,
        dtypes: tuple,
        tile_meta: dict,
        enable_tile_search: bool = True,
        enable_partition_search: bool = False,
        search_time_ms: float = 0.0,
    ) -> None:
        """写入缓存"""
        key = self._make_key(op_type, shape, dtypes, enable_tile_search, enable_partition_search)
        self._cache[key] = tile_meta
        self._save_entry(key, op_type, shape, dtypes, tile_meta, search_time_ms)

    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        total = self._hits + self._misses
        return {
            "total_requests": total,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate_percent": round(self._hits / total * 100, 1) if total > 0 else 0.0,
            "cached_entries": len(self._cache),
            "fingerprint": self._fingerprint,
            "chip_name": self._chip_params["name"],
        }

    def _load_from_disk(self) -> None:
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data.get("version") != CACHE_VERSION:
                logger.info(
                    "Tile cache version mismatch (file: %s, current: %s), ignoring",
                    data.get("version"), CACHE_VERSION,
                )
                return

            if data.get("fingerprint") != self._fingerprint:
                logger.info("Tile cache fingerprint mismatch, ignoring")
                return

            loaded = 0
            for entry_key, entry_data in data.get("entries", {}).items():
                try:
                    self._cache[entry_key] = entry_data["tile_meta"]
                    loaded += 1
                except (KeyError, TypeError):
                    pass

            if loaded > 0:
                logger.info(
                    "[OK] Loaded %d tile cache entries (chip: %s)",
                    loaded, self._chip_params["name"],
                )

        except Exception as e:
            logger.warning("Failed to load tile cache: %s", e)

    def _save_entry(
        self,
        key: str,
        op_type: str,
        shape: dict,
        dtypes: tuple,
        tile_meta: dict,
        search_time_ms: float,
    ) -> None:
        """增量保存单条记录到磁盘（线程安全）"""
        unique_suffix = f".{os.getpid()}_{uuid.uuid4().hex[:8]}.tmp"
        tmp_file = self.cache_file.parent / f"{self.cache_file.stem}{unique_suffix}"
        lock_file = self.cache_file.with_suffix('.lock')

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)

                with open(lock_file, 'w') as lf:
                    lock_acquired = False
                    try:
                        if _LOCK_AVAILABLE:
                            if sys.platform == 'win32':
                                try:
                                    msvcrt.locking(lf.fileno(), msvcrt.LK_NBLCK, 1)
                                    lock_acquired = True
                                except (IOError, OSError):
                                    raise IOError("Cannot acquire lock")
                            else:
                                fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                                lock_acquired = True
                    except (IOError, OSError):
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(0.1 * (attempt + 1))
                            continue
                        else:
                            return

                    try:
                        # 读取现有数据
                        if self.cache_file.exists():
                            try:
                                with open(self.cache_file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                            except (json.JSONDecodeError, ValueError):
                                data = self._empty_file()
                        else:
                            data = self._empty_file()

                        # 序列化 tile_meta（只保留可 JSON 序列化的值）
                        safe_meta = {}
                        for k, v in tile_meta.items():
                            if isinstance(v, (int, float, str, bool, type(None))):
                                safe_meta[k] = v
                            else:
                                safe_meta[k] = str(v)

                        data["entries"][key] = {
                            "op_type": op_type,
                            "shape": {str(k): v for k, v in shape.items()},
                            "tile_meta": safe_meta,
                            "timestamp": datetime.now().isoformat(),
                            "search_time_ms": search_time_ms,
                        }
                        data["stats"]["total_entries"] = len(data["entries"])
                        data["stats"]["last_updated"] = datetime.now().isoformat()

                        with open(tmp_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)

                        tmp_file.replace(self.cache_file)
                        return

                    finally:
                        if lock_acquired and _LOCK_AVAILABLE:
                            if sys.platform == 'win32':
                                try:
                                    msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
                                except (IOError, OSError):
                                    pass
                            else:
                                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

            except Exception:
                try:
                    if tmp_file.exists():
                        tmp_file.unlink()
                except OSError:
                    pass
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.05 * (attempt + 1))
                else:
                    return

    def _empty_file(self) -> dict:
        return {
            "version": CACHE_VERSION,
            "fingerprint": self._fingerprint,
            "chip": self._chip_params,
            "entries": {},
            "stats": {
                "total_entries": 0,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
            },
        }
