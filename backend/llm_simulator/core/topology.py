"""
拓扑解析模块

解析前端的 HierarchicalTopology 配置，构建芯片互联图，
并根据并行策略将芯片分配到 TP/PP/DP/EP 组。
"""

from typing import Any
from ..config import (
    HierarchicalTopology, PodConfig, RackConfig, BoardConfig, ChipConfig,
    ConnectionConfig, ChipNode, ChipLink, InterconnectGraph,
    ParallelismStrategy, ChipAssignment, ParallelGroupAssignment,
)


class TopologyParser:
    """拓扑解析器

    硬件参数现在直接嵌入在拓扑配置中：
    - ChipConfig: c2c_bandwidth_gbps, c2c_latency_us (同板芯片互联)
    - BoardConfig: b2b_bandwidth_gbps, b2b_latency_us (同 rack 不同 board)
    - RackConfig: r2r_bandwidth_gbps, r2r_latency_us (同 pod 不同 rack)
    - PodConfig: p2p_bandwidth_gbps, p2p_latency_us (跨 pod)
    """

    def __init__(self, topology_dict: dict[str, Any]):
        """
        初始化解析器

        Args:
            topology_dict: 前端传入的拓扑配置字典（包含硬件参数）
        """
        # 提取 hardware_params.interconnect 配置
        self.hardware_params = topology_dict.get("hardware_params", {})
        self.interconnect_params = self.hardware_params.get("interconnect", {})

        self.topology = self._parse_topology(topology_dict)
        self.interconnect: InterconnectGraph | None = None
        # 缓存芯片位置信息，用于快速查找
        self._chip_location_cache: dict[str, dict[str, Any]] = {}

    def _parse_topology(self, data: dict[str, Any]) -> HierarchicalTopology:
        """解析拓扑配置（包含嵌入的硬件参数）"""
        # 从 hardware_params 中获取芯片和互联参数
        chip_params = self.hardware_params.get("chip", {})
        c2c_params = self.interconnect_params.get("c2c", {})
        b2b_params = self.interconnect_params.get("b2b", {})
        r2r_params = self.interconnect_params.get("r2r", {})
        p2p_params = self.interconnect_params.get("p2p", {})

        pods = []
        for pod_data in data.get("pods", []):
            racks = []
            for rack_data in pod_data.get("racks", []):
                boards = []
                for board_data in rack_data.get("boards", []):
                    chips = []
                    for chip_data in board_data.get("chips", []):
                        pos = chip_data.get("position", [0, 0])
                        chip = ChipConfig(
                            id=chip_data["id"],
                            type=chip_data.get("type", "chip"),
                            position=tuple(pos) if isinstance(pos, list) else pos,
                            label=chip_data.get("label", ""),
                            # 硬件参数（优先使用 chip_data，否则从 chip_params 获取）
                            num_cores=chip_data.get("num_cores") or chip_params.get("num_cores", 0),
                            compute_tflops_fp8=chip_data.get("compute_tflops_fp8") or chip_params.get("compute_tflops_fp8", 0.0),
                            compute_tflops_bf16=chip_data.get("compute_tflops_bf16") or chip_params.get("compute_tflops_bf16", 0.0),
                            memory_capacity_gb=chip_data.get("memory_capacity_gb") or chip_params.get("memory_capacity_gb", 0.0),
                            memory_bandwidth_gbps=chip_data.get("memory_bandwidth_gbps") or chip_params.get("memory_bandwidth_gbps", 0.0),
                            memory_bandwidth_utilization=chip_data.get("memory_bandwidth_utilization") or chip_params.get("memory_bandwidth_utilization", 0.85),
                            lmem_capacity_mb=chip_data.get("lmem_capacity_mb") or chip_params.get("lmem_capacity_mb", 0.0),
                            lmem_bandwidth_gbps=chip_data.get("lmem_bandwidth_gbps") or chip_params.get("lmem_bandwidth_gbps", 0.0),
                            # 微架构参数（可选）
                            cube_m=chip_data.get("cube_m") or chip_params.get("cube_m"),
                            cube_k=chip_data.get("cube_k") or chip_params.get("cube_k"),
                            cube_n=chip_data.get("cube_n") or chip_params.get("cube_n"),
                            sram_size_kb=chip_data.get("sram_size_kb") or chip_params.get("sram_size_kb"),
                            sram_utilization=chip_data.get("sram_utilization") or chip_params.get("sram_utilization"),
                            lane_num=chip_data.get("lane_num") or chip_params.get("lane_num"),
                            align_bytes=chip_data.get("align_bytes") or chip_params.get("align_bytes"),
                            compute_dma_overlap_rate=chip_data.get("compute_dma_overlap_rate") or chip_params.get("compute_dma_overlap_rate"),
                        )
                        chips.append(chip)
                    boards.append(BoardConfig(
                        id=board_data["id"],
                        u_position=board_data.get("u_position", 1),
                        u_height=board_data.get("u_height", 1),
                        label=board_data.get("label", ""),
                        chips=chips,
                        # 互联参数（优先使用 board_data，否则从 b2b_params 获取）
                        b2b_bandwidth_gbps=board_data.get("b2b_bandwidth_gbps") or b2b_params.get("bandwidth_gbps", 0.0),
                        b2b_latency_us=board_data.get("b2b_latency_us") or b2b_params.get("latency_us", 0.0),
                    ))
                pos = rack_data.get("position", [0, 0])
                racks.append(RackConfig(
                    id=rack_data["id"],
                    position=tuple(pos) if isinstance(pos, list) else pos,
                    label=rack_data.get("label", ""),
                    total_u=rack_data.get("total_u", 42),
                    boards=boards,
                    # 互联参数（优先使用 rack_data，否则从 r2r_params 获取）
                    r2r_bandwidth_gbps=rack_data.get("r2r_bandwidth_gbps") or r2r_params.get("bandwidth_gbps", 0.0),
                    r2r_latency_us=rack_data.get("r2r_latency_us") or r2r_params.get("latency_us", 0.0),
                ))
            grid = pod_data.get("grid_size", [1, 1])
            pods.append(PodConfig(
                id=pod_data["id"],
                label=pod_data.get("label", ""),
                grid_size=tuple(grid) if isinstance(grid, list) else grid,
                racks=racks,
                # 互联参数（优先使用 pod_data，否则从 p2p_params 获取）
                p2p_bandwidth_gbps=pod_data.get("p2p_bandwidth_gbps") or p2p_params.get("bandwidth_gbps", 0.0),
                p2p_latency_us=pod_data.get("p2p_latency_us") or p2p_params.get("latency_us", 0.0),
            ))

        # 解析连接：根据 type 动态查找参数
        connections = []
        for conn_data in data.get("connections", []):
            conn_type = conn_data.get("type", "c2c")
            bandwidth = conn_data.get("bandwidth", 0)
            latency = conn_data.get("latency", 0)

            # 如果连接没有指定 bandwidth/latency，根据 type 从 interconnect_params 查找
            if bandwidth == 0 or latency == 0:
                if conn_type == "c2c":
                    bandwidth = bandwidth or c2c_params.get("bandwidth_gbps", 0.0)
                    latency = latency or (c2c_params.get("latency_us", 0.0) * 1000)  # us -> ns
                elif conn_type == "b2b":
                    bandwidth = bandwidth or b2b_params.get("bandwidth_gbps", 0.0)
                    latency = latency or (b2b_params.get("latency_us", 0.0) * 1000)  # us -> ns
                elif conn_type == "r2r":
                    bandwidth = bandwidth or r2r_params.get("bandwidth_gbps", 0.0)
                    latency = latency or (r2r_params.get("latency_us", 0.0) * 1000)  # us -> ns
                elif conn_type == "p2p":
                    bandwidth = bandwidth or p2p_params.get("bandwidth_gbps", 0.0)
                    latency = latency or (p2p_params.get("latency_us", 0.0) * 1000)  # us -> ns

            connections.append(ConnectionConfig(
                source=conn_data["source"],
                target=conn_data["target"],
                type=conn_type,
                bandwidth=bandwidth,
                latency=latency,
            ))

        return HierarchicalTopology(pods=pods, connections=connections)

    def validate_hardware_params(self) -> None:
        """验证拓扑中嵌入的硬件参数是否完整

        在需要使用硬件参数进行模拟时调用此方法。
        如果缺少必需的硬件参数，将抛出 ValueError。
        """
        errors = []

        for pod in self.topology.pods:
            # 验证 Pod 互联参数
            if pod.p2p_bandwidth_gbps <= 0:
                errors.append(f"Pod '{pod.id}' 缺少有效的 p2p_bandwidth_gbps")

            for rack in pod.racks:
                # 验证 Rack 互联参数
                if rack.r2r_bandwidth_gbps <= 0:
                    errors.append(f"Rack '{rack.id}' 缺少有效的 r2r_bandwidth_gbps")

                for board in rack.boards:
                    # 验证 Board 互联参数
                    if board.b2b_bandwidth_gbps <= 0:
                        errors.append(f"Board '{board.id}' 缺少有效的 b2b_bandwidth_gbps")

                    for chip in board.chips:
                        # 验证芯片硬件参数
                        chip_errors = []
                        if chip.compute_tflops_bf16 <= 0:
                            chip_errors.append("compute_tflops_bf16")
                        if chip.memory_capacity_gb <= 0:
                            chip_errors.append("memory_capacity_gb")
                        if chip.memory_bandwidth_gbps <= 0:
                            chip_errors.append("memory_bandwidth_gbps")
                        if chip.c2c_bandwidth_gbps <= 0:
                            chip_errors.append("c2c_bandwidth_gbps")

                        if chip_errors:
                            errors.append(f"Chip '{chip.id}' 缺少有效的硬件参数: {', '.join(chip_errors)}")

        if errors:
            raise ValueError(
                f"拓扑配置中缺少必需的硬件参数:\n" +
                "\n".join(f"  - {e}" for e in errors[:10]) +
                (f"\n  ... 还有 {len(errors) - 10} 个错误" if len(errors) > 10 else "")
            )

    def _build_location_cache(self) -> None:
        """构建芯片位置缓存，用于快速查找芯片所在的 board/rack/pod"""
        for pod in self.topology.pods:
            for rack in pod.racks:
                for board in rack.boards:
                    for chip in board.chips:
                        self._chip_location_cache[chip.id] = {
                            "chip": chip,
                            "board": board,
                            "rack": rack,
                            "pod": pod,
                        }

    def _get_chip_location(self, chip_id: str) -> dict[str, Any] | None:
        """获取芯片的位置信息"""
        if not self._chip_location_cache:
            self._build_location_cache()
        return self._chip_location_cache.get(chip_id)

    def get_chip_config(self, chip_id: str) -> ChipConfig | None:
        """获取芯片配置（包含硬件参数）"""
        loc = self._get_chip_location(chip_id)
        return loc["chip"] if loc else None

    def _get_link_params_by_location(
        self,
        loc1: ChipNode,
        loc2: ChipNode
    ) -> tuple[str, float, float]:
        """
        根据两个芯片的位置确定链路参数

        从拓扑配置中读取嵌入的硬件参数：
        - 同板芯片: chip.c2c_bandwidth_gbps, chip.c2c_latency_us
        - 同 rack 不同 board: board.b2b_bandwidth_gbps, board.b2b_latency_us
        - 同 pod 不同 rack: rack.r2r_bandwidth_gbps, rack.r2r_latency_us
        - 跨 pod: pod.p2p_bandwidth_gbps, pod.p2p_latency_us

        Args:
            loc1: 第一个芯片的位置信息
            loc2: 第二个芯片的位置信息

        Returns:
            (link_type, bandwidth_gbps, latency_us) 元组
        """
        # 获取芯片位置缓存
        chip1_loc = self._get_chip_location(loc1.chip_id)
        chip2_loc = self._get_chip_location(loc2.chip_id)

        if not chip1_loc or not chip2_loc:
            # 如果找不到位置信息，返回默认值
            return ("unknown", 0.0, 0.0)

        if loc1.board_id == loc2.board_id:
            # 同板芯片 - 使用 C2C 互联 (如 NVLink)
            chip: ChipConfig = chip1_loc["chip"]
            return (
                "nvlink",
                chip.c2c_bandwidth_gbps,
                chip.c2c_latency_us
            )
        elif loc1.rack_id == loc2.rack_id:
            # 同机柜不同板 - 使用 B2B 互联
            board: BoardConfig = chip1_loc["board"]
            return (
                "pcie",
                board.b2b_bandwidth_gbps,
                board.b2b_latency_us
            )
        elif loc1.pod_id == loc2.pod_id:
            # 同 Pod 不同机柜 - 使用 R2R 互联
            rack: RackConfig = chip1_loc["rack"]
            return (
                "ib",
                rack.r2r_bandwidth_gbps,
                rack.r2r_latency_us
            )
        else:
            # 跨 Pod - 使用 P2P 互联
            pod: PodConfig = chip1_loc["pod"]
            return (
                "ethernet",
                pod.p2p_bandwidth_gbps,
                pod.p2p_latency_us
            )

    def build_interconnect_graph(self) -> InterconnectGraph:
        """构建芯片互联图"""
        nodes: list[ChipNode] = []
        links: list[ChipLink] = []

        # 收集所有芯片
        chip_to_location: dict[str, ChipNode] = {}
        for pod in self.topology.pods:
            for rack in pod.racks:
                for board in rack.boards:
                    for chip in board.chips:
                        node = ChipNode(
                            chip_id=chip.id,
                            pod_id=pod.id,
                            rack_id=rack.id,
                            board_id=board.id,
                            position=chip.position,
                        )
                        nodes.append(node)
                        chip_to_location[chip.id] = node

        # 解析连接并确定链路参数
        for conn in self.topology.connections:
            src = conn.source
            dst = conn.target

            # 跳过非芯片连接
            if src not in chip_to_location or dst not in chip_to_location:
                continue

            # 确定链路类型和参数
            src_node = chip_to_location[src]
            dst_node = chip_to_location[dst]
            link_type, bandwidth, latency = self._get_link_params_by_location(src_node, dst_node)

            # 使用连接配置中的显式值覆盖默认值
            if conn.bandwidth > 0:
                bandwidth = conn.bandwidth
            if conn.latency > 0:
                latency = conn.latency / 1000  # ns -> us

            links.append(ChipLink(
                source=src,
                target=dst,
                bandwidth_gbps=bandwidth,
                latency_us=latency,
                link_type=link_type,
            ))

        self.interconnect = InterconnectGraph(nodes=nodes, links=links)
        return self.interconnect

    def get_all_chip_ids(self) -> list[str]:
        """获取所有芯片ID（按层级顺序）"""
        chip_ids = []
        for pod in self.topology.pods:
            for rack in pod.racks:
                for board in rack.boards:
                    for chip in board.chips:
                        chip_ids.append(chip.id)
        return chip_ids

    def map_parallelism(self, strategy: ParallelismStrategy, is_moe: bool = False) -> ParallelGroupAssignment:
        """
        根据并行策略将芯片分配到各并行组

        分配顺序（从内到外）：TP -> EP -> PP -> DP
        - TP组优先放在同板芯片（高带宽NVLink）
        - PP组可以跨板（P2P通信）
        - DP组可以跨机柜甚至跨Pod

        Args:
            strategy: 并行策略
            is_moe: 是否为 MoE 模型（影响芯片数计算）

        Returns:
            ParallelGroupAssignment: 并行组分配结果
        """
        chip_ids = self.get_all_chip_ids()
        # 芯片数计算：
        # - MoE 模型：DP × TP（因为 MoE 约束 DP×TP = MoE_TP×EP）
        # - 非 MoE 模型：DP × TP × EP
        if is_moe:
            total_chips = strategy.dp * strategy.tp
            formula_desc = f"DP={strategy.dp} × TP={strategy.tp}"
        else:
            total_chips = strategy.dp * strategy.tp * strategy.ep
            formula_desc = f"DP={strategy.dp} × TP={strategy.tp} × EP={strategy.ep}"

        if len(chip_ids) < total_chips:
            raise ValueError(
                f"芯片数量不足: 需要 {total_chips} 个芯片 "
                f"({formula_desc})，"
                f"但只有 {len(chip_ids)} 个芯片"
            )

        assignments: list[ChipAssignment] = []
        tp_groups: list[list[str]] = []
        pp_groups: list[list[str]] = []
        dp_groups: list[list[str]] = []
        ep_groups: list[list[str]] = []

        # 初始化组列表
        num_tp_groups = strategy.dp * strategy.pp * strategy.ep
        num_pp_groups = strategy.dp * strategy.tp * strategy.ep
        num_dp_groups = strategy.tp * strategy.pp * strategy.ep
        num_ep_groups = strategy.dp * strategy.tp * strategy.pp

        for _ in range(num_tp_groups):
            tp_groups.append([])
        for _ in range(num_pp_groups):
            pp_groups.append([])
        for _ in range(num_dp_groups):
            dp_groups.append([])
        for _ in range(num_ep_groups):
            ep_groups.append([])

        # 分配芯片到各组
        global_rank = 0
        for dp in range(strategy.dp):
            for pp in range(strategy.pp):
                for ep in range(strategy.ep):
                    for tp in range(strategy.tp):
                        if global_rank >= len(chip_ids):
                            break

                        chip_id = chip_ids[global_rank]

                        # 创建分配记录
                        assignment = ChipAssignment(
                            chip_id=chip_id,
                            global_rank=global_rank,
                            dp_rank=dp,
                            tp_rank=tp,
                            pp_rank=pp,
                            ep_rank=ep,
                            sp_rank=0,  # SP通常与TP绑定
                        )
                        assignments.append(assignment)

                        # 计算组索引
                        tp_group_idx = dp * strategy.pp * strategy.ep + pp * strategy.ep + ep
                        pp_group_idx = dp * strategy.tp * strategy.ep + tp * strategy.ep + ep
                        dp_group_idx = tp * strategy.pp * strategy.ep + pp * strategy.ep + ep
                        ep_group_idx = dp * strategy.tp * strategy.pp + tp * strategy.pp + pp

                        # 添加到对应组
                        tp_groups[tp_group_idx].append(chip_id)
                        pp_groups[pp_group_idx].append(chip_id)
                        dp_groups[dp_group_idx].append(chip_id)
                        if ep_group_idx < len(ep_groups):
                            ep_groups[ep_group_idx].append(chip_id)

                        global_rank += 1

        return ParallelGroupAssignment(
            assignments=assignments,
            tp_groups=tp_groups,
            pp_groups=pp_groups,
            dp_groups=dp_groups,
            ep_groups=ep_groups,
        )

    def get_link_params_for_group(
        self,
        group_chips: list[str],
        comm_type: str  # 'allreduce' | 'p2p' | 'alltoall'
    ) -> tuple[float, float]:
        """
        获取组内通信的链路参数

        对于 AllReduce，使用组内最慢链路的参数
        对于 P2P，使用源目标之间的链路参数

        Args:
            group_chips: 组内芯片ID列表
            comm_type: 通信类型

        Returns:
            (带宽 Gbps, 延迟 us) 元组
        """
        if not self.interconnect:
            self.build_interconnect_graph()

        if len(group_chips) <= 1:
            # 单芯片不需要通信
            # 尝试获取芯片的 C2C 带宽作为默认值
            if group_chips:
                chip_config = self.get_chip_config(group_chips[0])
                if chip_config:
                    return chip_config.c2c_bandwidth_gbps, 0.0
            return 0.0, 0.0

        # 找到组内所有芯片的位置
        chip_locations = {}
        for node in self.interconnect.nodes:
            if node.chip_id in group_chips:
                chip_locations[node.chip_id] = node

        # 确定链路类型
        min_bandwidth = float('inf')
        max_latency = 0.0

        for i, chip1 in enumerate(group_chips):
            for chip2 in group_chips[i + 1:]:
                loc1 = chip_locations.get(chip1)
                loc2 = chip_locations.get(chip2)

                if not loc1 or not loc2:
                    continue

                # 使用公共方法获取链路参数
                _, bw, lat = self._get_link_params_by_location(loc1, loc2)
                min_bandwidth = min(min_bandwidth, bw)
                max_latency = max(max_latency, lat)

        
        return min_bandwidth, max_latency
