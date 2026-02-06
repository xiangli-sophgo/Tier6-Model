"""CHIPMathica 核心工具函数

提供跨模块使用的通用工具函数。
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# 文件操作
# ============================================================================


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在

    Args:
        path: 目录路径

    Returns:
        创建或已存在的目录路径
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_load_yaml(path: str | Path) -> dict[str, Any]:
    """安全加载 YAML 文件

    Args:
        path: YAML 文件路径

    Returns:
        解析后的字典

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: YAML 解析失败
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file {path}: {e}") from e


def safe_save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """安全保存 YAML 文件

    Args:
        data: 要保存的数据
        path: 目标文件路径
    """
    import yaml

    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)


def safe_load_json(path: str | Path) -> dict[str, Any]:
    """安全加载 JSON 文件

    Args:
        path: JSON 文件路径

    Returns:
        解析后的字典
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_save_json(
    data: dict[str, Any],
    path: str | Path,
    indent: int = 2,
) -> None:
    """安全保存 JSON 文件

    Args:
        data: 要保存的数据
        path: 目标文件路径
        indent: 缩进空格数
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# ============================================================================
# 字典操作
# ============================================================================


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并字典

    Args:
        base: 基础字典
        override: 覆盖字典

    Returns:
        合并后的新字典

    Example:
        >>> base = {"a": 1, "b": {"c": 2}}
        >>> override = {"b": {"d": 3}}
        >>> deep_merge(base, override)
        {"a": 1, "b": {"c": 2, "d": 3}}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """扁平化嵌套字典

    Args:
        d: 嵌套字典
        parent_key: 父键前缀
        sep: 键分隔符

    Returns:
        扁平化后的字典

    Example:
        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {"a.b": 1, "a.c": 2}
    """
    items: list[tuple[str, Any]] = []

    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))

    return dict(items)


def unflatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """反扁平化字典

    Args:
        d: 扁平化字典
        sep: 键分隔符

    Returns:
        嵌套字典
    """
    result: dict[str, Any] = {}

    for key, value in d.items():
        parts = key.split(sep)
        current = result

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return result


def get_nested(d: dict[str, Any], key: str, sep: str = ".") -> Any:
    """获取嵌套字典值

    Args:
        d: 字典
        key: 点分隔的键路径
        sep: 键分隔符

    Returns:
        对应的值，不存在则返回 None

    Example:
        >>> get_nested({"a": {"b": 1}}, "a.b")
        1
    """
    parts = key.split(sep)
    current = d

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]

    return current


def set_nested(d: dict[str, Any], key: str, value: Any, sep: str = ".") -> None:
    """设置嵌套字典值

    Args:
        d: 字典
        key: 点分隔的键路径
        value: 要设置的值
        sep: 键分隔符
    """
    parts = key.split(sep)
    current = d

    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


# ============================================================================
# 哈希与标识
# ============================================================================


def compute_hash(data: Any, algorithm: str = "md5") -> str:
    """计算数据哈希值

    Args:
        data: 要哈希的数据
        algorithm: 哈希算法 (md5, sha256, etc.)

    Returns:
        十六进制哈希字符串
    """
    if isinstance(data, dict):
        # 对字典进行排序以确保一致性
        data_str = json.dumps(data, sort_keys=True)
    elif isinstance(data, (list, tuple)):
        data_str = json.dumps(data)
    else:
        data_str = str(data)

    hasher = hashlib.new(algorithm)
    hasher.update(data_str.encode("utf-8"))
    return hasher.hexdigest()


def generate_run_id(prefix: str = "run") -> str:
    """生成运行 ID

    Args:
        prefix: ID 前缀

    Returns:
        格式: {prefix}_{timestamp}_{random}
    """
    import random
    import string

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{timestamp}_{random_suffix}"


# ============================================================================
# 格式化
# ============================================================================


def format_bytes(n: int) -> str:
    """格式化字节数为可读字符串

    Args:
        n: 字节数

    Returns:
        格式化字符串 (如 "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def format_flops(n: float) -> str:
    """格式化 FLOPs 为可读字符串

    Args:
        n: FLOPs 数值

    Returns:
        格式化字符串 (如 "1.5 TFLOPS")
    """
    for unit in ["FLOPS", "KFLOPS", "MFLOPS", "GFLOPS", "TFLOPS", "PFLOPS"]:
        if abs(n) < 1000:
            return f"{n:.2f} {unit}"
        n /= 1000
    return f"{n:.2f} EFLOPS"


def format_time(ns: float) -> str:
    """格式化时间（纳秒）为可读字符串

    Args:
        ns: 纳秒数

    Returns:
        格式化字符串 (如 "1.5 ms")
    """
    if ns < 1000:
        return f"{ns:.2f} ns"
    elif ns < 1_000_000:
        return f"{ns / 1000:.2f} µs"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    else:
        return f"{ns / 1_000_000_000:.2f} s"


def format_percent(value: float, decimals: int = 1) -> str:
    """格式化百分比

    Args:
        value: 0.0-1.0 的值
        decimals: 小数位数

    Returns:
        格式化字符串 (如 "75.5%")
    """
    return f"{value * 100:.{decimals}f}%"


# ============================================================================
# 单位转换
# ============================================================================


def to_bytes(value: int | float, unit: str = "B") -> int:
    """将带单位的值转换为字节

    Args:
        value: 数值
        unit: 单位 (B, KB, MB, GB, TB)

    Returns:
        字节数
    """
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }
    return int(value * multipliers.get(unit.upper(), 1))


def to_flops(value: float, unit: str = "FLOPS") -> float:
    """将带单位的值转换为 FLOPS

    Args:
        value: 数值
        unit: 单位 (FLOPS, GFLOPS, TFLOPS)

    Returns:
        FLOPS 数值
    """
    multipliers = {
        "FLOPS": 1,
        "KFLOPS": 1e3,
        "MFLOPS": 1e6,
        "GFLOPS": 1e9,
        "TFLOPS": 1e12,
        "PFLOPS": 1e15,
    }
    return value * multipliers.get(unit.upper(), 1)


def to_ns(value: float, unit: str = "ns") -> float:
    """将带单位的时间转换为纳秒

    Args:
        value: 数值
        unit: 单位 (ns, us, ms, s)

    Returns:
        纳秒数
    """
    multipliers = {
        "ns": 1,
        "us": 1e3,
        "µs": 1e3,
        "ms": 1e6,
        "s": 1e9,
    }
    return value * multipliers.get(unit.lower(), 1)
