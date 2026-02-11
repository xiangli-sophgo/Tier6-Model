"""成本分析模块

生成部署成本分解报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from math_model.L2_arch.protocols import PodSpec
    from math_model.L4_evaluation.metrics import Aggregates


# 芯片定价表 (USD)
CHIP_PRICES = {
    "SG2262": 2371.275,
}

# 互联成本层级 ($/lane, 112Gbps per lane)
INTERCONNECT_COST_TIERS = {
    2: 1,  # 1-2 chips: PCIe direct
    8: 55,  # 8 chips: Ethernet switch
    16: 70,  # 16 chips: Switch + DAC
    32: 70,  # 32 chips: Switch + DAC
    64: 105.247,  # 64 chips: Switch + AEC
    128: 247,  # 64+ chips: Full optical
}


@dataclass
class CostBreakdown:
    """成本分解

    Attributes:
        server_cost: 服务器硬件成本
        interconnect_cost: 互联成本
        total_cost: 总成本
        cost_per_chip: 每芯片成本
        cost_per_million_tokens: 每百万 token 成本
        chip_count: 芯片数量
        chip_type: 芯片类型
        depreciation_years: 折旧年限
    """

    server_cost: float = 0.0
    interconnect_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_chip: float = 0.0
    cost_per_million_tokens: float = 0.0
    chip_count: int = 0
    chip_type: str = ""
    depreciation_years: int = 3

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "serverCost": self.server_cost,
            "interconnectCost": self.interconnect_cost,
            "totalCost": self.total_cost,
            "costPerChip": self.cost_per_chip,
            "costPerMillionTokens": self.cost_per_million_tokens,
            "chipCount": self.chip_count,
            "chipType": self.chip_type,
            "depreciationYears": self.depreciation_years,
        }


class CostAnalyzer:
    """成本分析器

    计算部署的硬件成本和运营成本。
    """

    def __init__(
        self,
        chip_prices: dict[str, float] | None = None,
        modules_per_server: int = 8,
        chips_per_module: int = 1,
        depreciation_years: int = 3,
    ) -> None:
        """初始化

        Args:
            chip_prices: 芯片价格表
            modules_per_server: 每服务器模块数
            chips_per_module: 每模块芯片数
            depreciation_years: 折旧年限
        """
        self.chip_prices = chip_prices or CHIP_PRICES
        self.modules_per_server = modules_per_server
        self.chips_per_module = chips_per_module
        self.depreciation_years = depreciation_years

    def get_chip_price(self, chip_type: str) -> float:
        """获取芯片价格

        Args:
            chip_type: 芯片类型

        Returns:
            float: 价格 (USD)
        """
        # 尝试精确匹配
        if chip_type in self.chip_prices:
            return self.chip_prices[chip_type]

        # 尝试严格大小写不敏感匹配
        chip_upper = chip_type.upper()
        for key, price in self.chip_prices.items():
            if key.upper() == chip_upper:
                return price

        # 未找到芯片价格
        available = ", ".join(self.chip_prices.keys())
        raise ValueError(
            f"Unknown chip type '{chip_type}' for cost analysis. "
            f"Available: {available}. Please add to CHIP_PRICES or use a known type."
        )

    def get_lane_cost(self, chip_count: int) -> float:
        """获取每 lane 互联成本

        Args:
            chip_count: 芯片数量

        Returns:
            float: $/lane
        """
        for threshold, cost in sorted(INTERCONNECT_COST_TIERS.items()):
            if chip_count <= threshold:
                return cost
        return INTERCONNECT_COST_TIERS[128]

    def calculate_server_cost(
        self,
        chip_type: str,
        chip_count: int,
    ) -> float:
        """计算服务器成本

        公式:
        server_cost = (chip_price × chips_per_module + 750) × modules_per_server + 12000 + 7500

        Args:
            chip_type: 芯片类型
            chip_count: 芯片数量

        Returns:
            float: 服务器成本 (USD)
        """
        chip_price = self.get_chip_price(chip_type)
        chips_per_server = self.modules_per_server * self.chips_per_module
        num_servers = (chip_count + chips_per_server - 1) // chips_per_server

        # 每服务器成本
        module_cost = chip_price * self.chips_per_module + 750  # 模块组装
        server_base = module_cost * self.modules_per_server + 12000  # 机箱主板
        server_power = 7500  # 电源散热

        return num_servers * (server_base + server_power)

    def calculate_interconnect_cost(
        self,
        chip_count: int,
        lanes_per_chip: int = 16,
    ) -> float:
        """计算互联成本

        Args:
            chip_count: 芯片数量
            lanes_per_chip: 每芯片 lane 数

        Returns:
            float: 互联成本 (USD)
        """
        if chip_count <= 1:
            return 0.0

        lane_cost = self.get_lane_cost(chip_count)
        total_lanes = chip_count * lanes_per_chip

        return total_lanes * lane_cost

    def analyze(
        self,
        chip_type: str,
        chip_count: int,
        tps: float = 0.0,
        lanes_per_chip: int = 16,
    ) -> CostBreakdown:
        """分析成本

        Args:
            chip_type: 芯片类型
            chip_count: 芯片数量
            tps: Tokens per second (用于计算运营成本)
            lanes_per_chip: 每芯片 lane 数

        Returns:
            CostBreakdown: 成本分解
        """
        server_cost = self.calculate_server_cost(chip_type, chip_count)
        interconnect_cost = self.calculate_interconnect_cost(chip_count, lanes_per_chip)
        total_cost = server_cost + interconnect_cost

        # 每芯片成本
        cost_per_chip = total_cost / chip_count if chip_count > 0 else 0.0

        # 每百万 token 成本 (基于 TPS 和折旧)
        cost_per_million_tokens = 0.0
        if tps > 0:
            # 年运行时间: 8760 小时 (假设 100% 利用率)
            hours_per_year = 8760
            total_tokens_per_year = tps * 3600 * hours_per_year
            tokens_over_depreciation = total_tokens_per_year * self.depreciation_years
            cost_per_million_tokens = (
                total_cost / tokens_over_depreciation * 1_000_000
                if tokens_over_depreciation > 0
                else 0.0
            )

        return CostBreakdown(
            server_cost=server_cost,
            interconnect_cost=interconnect_cost,
            total_cost=total_cost,
            cost_per_chip=cost_per_chip,
            cost_per_million_tokens=cost_per_million_tokens,
            chip_count=chip_count,
            chip_type=chip_type,
            depreciation_years=self.depreciation_years,
        )

    def analyze_from_pod(
        self,
        pod: "PodSpec",
        aggregates: "Aggregates | None" = None,
    ) -> CostBreakdown:
        """从 Pod 规格分析成本

        Args:
            pod: Pod 规格
            aggregates: 聚合指标 (可选，用于计算 TPS)

        Returns:
            CostBreakdown: 成本分解
        """
        chip = pod.get_primary_chip()
        chip_type = chip.name
        chip_count = pod.total_chips
        tps = aggregates.tps if aggregates else 0.0

        return self.analyze(chip_type, chip_count, tps)
