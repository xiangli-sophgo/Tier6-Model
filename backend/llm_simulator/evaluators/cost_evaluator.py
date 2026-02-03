"""
LLM 推理集群成本评估器

实现基于芯片规模和并行策略的分层成本计算模型。
成本模型包括：
1. 服务器成本（芯片 + 主板 + RDMA）
2. 互联成本（交换机 + 线缆 + 光模块，阶梯式定价）
"""

from typing import Dict, Optional


class CostEvaluator:
    """LLM 推理集群成本评估器

    成本模型：
    - 单服务器成本 = (A × n + 750) × m + 12000 + 7500
    - 互联成本 = CP_num × m × n × lanes × lane_cost(CP_num)
    - 总成本 = 服务器成本 + 互联成本

    其中：
    - A: 单芯片成本 ($)
    - n: 每个 QAM 模组的芯片数（默认 1）
    - m: 每服务器的 QAM 模组数（默认 8）
    - CP_num: 芯片总数
    - lanes: 互联带宽需求 (单位: 112Gbps lane)
    - lane_cost: 根据规模分层定价
    """

    # 默认芯片价格表 (USD)
    DEFAULT_CHIP_PRICES = {
        "B200": 6303,      # NVIDIA Blackwell
        "H100": 4500,      # NVIDIA Hopper
        "H800": 3800,      # H100 中国版
        "SG2262": 2500,    # 国产芯片
        "SG2260E": 2500,   # 国产芯片（别名）
    }

    def __init__(self, chip_prices: Optional[Dict[str, float]] = None):
        """初始化成本评估器

        Args:
            chip_prices: 自定义芯片价格表，格式 {芯片型号: 价格($)}
        """
        self.chip_prices = chip_prices if chip_prices is not None else self.DEFAULT_CHIP_PRICES

    def calculate_server_cost(
        self,
        chip_type: str,
        m: int = 8,
        n: int = 1
    ) -> float:
        """计算单服务器成本

        公式: (A × n + 750) × m + 12000 + 7500

        Args:
            chip_type: 芯片型号
            m: 每服务器的 QAM 模组数（默认 8）
            n: 每个 QAM 模组的芯片数（默认 1）

        Returns:
            单服务器成本 ($)
        """
        A = self.chip_prices.get(chip_type, 6303)  # 默认 B200 价格
        qam_cost = (A * n + 750) * m
        server_base_cost = 12000  # 主板、电源、机箱等
        rdma_cost = 7500          # RDMA 网卡 (<200Gbps)

        return qam_cost + server_base_cost + rdma_cost

    def get_lane_cost(self, cp_num: int) -> float:
        """获取互联 lane 成本（分层定价）

        根据芯片数量分层：
        - 1-2 芯片: $1/lane (PCIe 直连)
        - 8 芯片: $55/lane (Eth Switch)
        - 16-32 芯片: $70/lane (Switch + DAC)
        - 64 芯片: $105.247/lane (Switch + AEC)
        - 64+ 芯片: $247/lane (Switch + 光模块 + 光纤)

        Args:
            cp_num: 芯片总数

        Returns:
            单 lane 成本 ($/lane)
        """
        if cp_num < 8:
            return 1        # PCIe 直连
        elif cp_num == 8:
            return 55       # 标准交换机
        elif 16 <= cp_num < 32:
            return 70       # Switch + DAC
        elif 32 <= cp_num <= 64:
            return 105.247  # Switch + AEC
        else:  # cp_num > 64
            return 247      # 全光方案

    def estimate_interconnect_bandwidth(
        self,
        model_size_gb: float,
        tp: int,
        tpot_ms: float
    ) -> float:
        """估算互联带宽需求 (Gbps)

        经验公式（推理场景）：
        带宽需求 = 2 × 模型大小 × TP 并行度 / TPOT × 1000

        Args:
            model_size_gb: 模型大小 (GB)
            tp: Tensor Parallelism 并行度
            tpot_ms: Time Per Output Token (ms)

        Returns:
            带宽需求 (Gbps)
        """
        if tpot_ms <= 0:
            return 0

        # 推理场景：模型权重需在 TP 组内全部读取
        # 2x 系数：考虑 All-Reduce 通信（前向 + 反向）
        bandwidth_gbps = 2 * model_size_gb * tp / tpot_ms * 1000

        return bandwidth_gbps

    def calculate_total_cost(
        self,
        cp_num: int,
        chip_type: str,
        model_size_gb: float,
        tp: int,
        tpot_ms: float,
        m: int = 8,
        n: int = 1
    ) -> Dict[str, float]:
        """计算 CP 总成本

        Args:
            cp_num: 芯片总数
            chip_type: 芯片型号
            model_size_gb: 模型大小 (GB)
            tp: Tensor Parallelism 并行度
            tpot_ms: Time Per Output Token (ms)
            m: 每服务器的 QAM 模组数（默认 8）
            n: 每个 QAM 模组的芯片数（默认 1）

        Returns:
            成本明细字典：
            {
                "server_cost": 服务器总成本 ($),
                "interconnect_cost": 互联总成本 ($),
                "total_cost": 总成本 ($),
                "bandwidth_gbps": 互联带宽需求 (Gbps),
                "lanes": 所需 lane 数量,
                "lane_cost": 单 lane 成本 ($/lane),
                "cost_per_chip": 单芯片摊派成本 ($),
            }
        """
        # 1. 计算服务器成本
        server_cost = self.calculate_server_cost(chip_type, m, n)
        total_server_cost = cp_num * server_cost

        # 2. 计算互联成本
        bandwidth_gbps = self.estimate_interconnect_bandwidth(
            model_size_gb, tp, tpot_ms
        )
        lanes = bandwidth_gbps / 112  # 112Gbps per lane
        lane_cost = self.get_lane_cost(cp_num)
        interconnect_cost = cp_num * m * n * lanes * lane_cost

        # 3. 总成本
        total_cost = total_server_cost + interconnect_cost
        cost_per_chip = total_cost / cp_num if cp_num > 0 else 0

        return {
            "server_cost": total_server_cost,
            "interconnect_cost": interconnect_cost,
            "total_cost": total_cost,
            "bandwidth_gbps": bandwidth_gbps,
            "lanes": lanes,
            "lane_cost": lane_cost,
            "cost_per_chip": cost_per_chip,
        }

    def calculate_cost_per_million_tokens(
        self,
        total_cost: float,
        tps: float,
        utilization_hours_per_year: float = 8760  # 7x24小时
    ) -> float:
        """计算每百万 tokens 的成本 ($/M tokens)

        考虑设备折旧和利用率。

        Args:
            total_cost: 集群总成本 ($)
            tps: 集群总吞吐量 (tokens/s)
            utilization_hours_per_year: 年利用时长（小时）

        Returns:
            每百万 tokens 的成本 ($/M tokens)
        """
        if tps <= 0:
            return 0

        # 假设设备折旧 3 年
        depreciation_years = 3
        annual_cost = total_cost / depreciation_years

        # 年吞吐量 (M tokens)
        annual_throughput = tps * utilization_hours_per_year * 3600 / 1e6

        # 单位成本
        cost_per_m_tokens = annual_cost / annual_throughput if annual_throughput > 0 else 0

        return cost_per_m_tokens


# 便捷函数
def evaluate_deployment_cost(
    chips: int,
    chip_type: str,
    num_parameters: int,
    tp: int,
    tpot_ms: float,
    tps: float,
    bytes_per_param: int = 2
) -> Dict[str, float]:
    """评估部署成本（便捷函数）

    Args:
        chips: 芯片数量
        chip_type: 芯片型号
        num_parameters: 模型参数量
        tp: Tensor Parallelism 并行度
        tpot_ms: Time Per Output Token (ms)
        tps: 集群总吞吐量 (tokens/s)
        bytes_per_param: 每参数字节数（默认 2，FP16/BF16）

    Returns:
        成本评估结果
    """
    evaluator = CostEvaluator()

    # 模型大小 (GB)
    model_size_gb = num_parameters * bytes_per_param / 1e9

    # 计算总成本
    cost_result = evaluator.calculate_total_cost(
        cp_num=chips,
        chip_type=chip_type,
        model_size_gb=model_size_gb,
        tp=tp,
        tpot_ms=tpot_ms,
    )

    return {
        **cost_result,
        "model_size_gb": model_size_gb,
    }
