"""层次化统计框架 (借鉴 gem5 Stats 体系)

统计类型:
  - ScalarStat: 标量 (计数器/累加器/峰值)
  - VectorStat: 向量 (按标签分组的一组数值)

层次化容器:
  - StatGroup: 每个 SimObject 持有一个，支持嵌套子组
  - dump() 递归收集所有统计 -> 扁平化 dict
  - reset() 重置所有统计 (用于分阶段统计)

参考设计: docs/design/instruction-level-simulator/10-statistics-framework.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class ScalarStat:
    """标量统计量

    用途: 计数、累加、峰值、最终状态值。

    Attributes:
        name: 统计量名称
        desc: 描述
        value: 当前值
    """

    name: str
    desc: str
    value: float = 0.0

    def inc(self, delta: float = 1.0) -> None:
        """累加"""
        self.value += delta

    def set_max(self, v: float) -> None:
        """取最大值 (用于记录峰值)"""
        if v > self.value:
            self.value = float(v)

    def reset(self) -> None:
        """重置为 0"""
        self.value = 0.0


@dataclass
class VectorStat:
    """向量统计量 (按标签分组)

    等价于 dict[str, float]，同一指标按不同类别拆开统计。

    Attributes:
        name: 统计量名称
        desc: 描述
        bins: 标签 -> 数值
    """

    name: str
    desc: str
    bins: dict[str, float] = field(default_factory=dict)

    def inc(self, label: str, delta: float = 1.0) -> None:
        """指定标签累加"""
        self.bins[label] = self.bins.get(label, 0.0) + delta

    def reset(self) -> None:
        """清空所有桶"""
        self.bins.clear()


# 统计量联合类型
StatType = Union[ScalarStat, VectorStat]


class StatGroup:
    """层次化统计容器

    每个模块 (SimObject) 持有一个 StatGroup，支持嵌套子组。
    dump() 递归收集所有统计到扁平化 dict。

    层次示例:
        kernel (SimKernel)
        +-- core0 (CoreSubsys)
        |   +-- tiu (TIU Engine)
        |   +-- dma (DMA Engine)
        |   ...
        +-- bus (BusModel)
    """

    def __init__(self, name: str, parent: StatGroup | None = None) -> None:
        self.name = name
        self._stats: dict[str, StatType] = {}
        self._children: dict[str, StatGroup] = {}
        if parent is not None:
            parent._add_child(self)

    def scalar(self, name: str, desc: str) -> ScalarStat:
        """创建并注册一个 ScalarStat"""
        stat = ScalarStat(name=name, desc=desc)
        self._stats[name] = stat
        return stat

    def vector(self, name: str, desc: str) -> VectorStat:
        """创建并注册一个 VectorStat"""
        stat = VectorStat(name=name, desc=desc)
        self._stats[name] = stat
        return stat

    def _add_child(self, child: StatGroup) -> None:
        """添加子统计组"""
        self._children[child.name] = child

    def dump(self) -> dict[str, Any]:
        """递归收集所有统计 -> 扁平化 dict

        key 格式: "parent.child.stat_name"
        VectorStat 的 value 是 dict[str, float]。
        """
        result: dict[str, Any] = {}
        for name, stat in self._stats.items():
            key = f"{self.name}.{name}"
            if isinstance(stat, VectorStat):
                result[key] = dict(stat.bins)
            else:
                result[key] = stat.value
        for child in self._children.values():
            for child_key, child_value in child.dump().items():
                result[f"{self.name}.{child_key}"] = child_value
        return result

    def reset(self) -> None:
        """重置所有统计 (用于分阶段统计，如 prefill/decode)"""
        for stat in self._stats.values():
            stat.reset()
        for child in self._children.values():
            child.reset()
