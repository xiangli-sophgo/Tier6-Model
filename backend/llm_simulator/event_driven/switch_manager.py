"""
Switch 管理器模块

管理网络中的 Switch 节点，提供：
- Switch 拓扑构建
- 路由计算
- 端口资源管理
- 数据包路径追踪

设计理念：包级事件驱动仿真
- 通信被分解为多个数据包
- 每个包经过 Switch 端口时请求资源
- 排队通过 ResourceManager 的 idle_at 机制自然发生
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any

from ..config.types import (
    SwitchType, SwitchLayer, SwitchHardwareConfig, SwitchInstanceConfig,
    SwitchPortState, Packet, SwitchGraph,
)

if TYPE_CHECKING:
    from ..core.topology import TopologyParser

logger = logging.getLogger(__name__)


# ============================================
# 默认 Switch 硬件配置
# ============================================

DEFAULT_SWITCH_CONFIGS: dict[str, SwitchHardwareConfig] = {
    "leaf_72": SwitchHardwareConfig(
        name="leaf_72",
        switch_type=SwitchType.LEAF,
        port_count=72,
        port_bandwidth_gbps=100.0,
        processing_delay_us=0.5,
        cut_through_delay_us=0.1,
        buffer_size_mb=32.0,
        buffer_per_port_kb=512.0,
    ),
    "leaf_128": SwitchHardwareConfig(
        name="leaf_128",
        switch_type=SwitchType.LEAF,
        port_count=128,
        port_bandwidth_gbps=400.0,
        processing_delay_us=0.3,
        cut_through_delay_us=0.08,
        buffer_size_mb=64.0,
        buffer_per_port_kb=512.0,
    ),
    "spine_512": SwitchHardwareConfig(
        name="spine_512",
        switch_type=SwitchType.SPINE,
        port_count=512,
        port_bandwidth_gbps=400.0,
        processing_delay_us=0.8,
        cut_through_delay_us=0.15,
        buffer_size_mb=128.0,
        buffer_per_port_kb=256.0,
    ),
    "core_1024": SwitchHardwareConfig(
        name="core_1024",
        switch_type=SwitchType.CORE,
        port_count=1024,
        port_bandwidth_gbps=400.0,
        processing_delay_us=1.0,
        cut_through_delay_us=0.2,
        buffer_size_mb=256.0,
        buffer_per_port_kb=256.0,
    ),
}


@dataclass
class SwitchNode:
    """Switch 节点（运行时状态）

    封装 SwitchInstanceConfig 并添加运行时状态。
    """
    # 配置信息
    config: SwitchInstanceConfig
    hardware: SwitchHardwareConfig

    # 端口状态（端口号 → 状态）
    port_states: dict[int, SwitchPortState] = field(default_factory=dict)

    # 路由表缓存（目标设备 ID → 输出端口号）
    routing_cache: dict[str, int] = field(default_factory=dict)

    @property
    def switch_id(self) -> str:
        return self.config.switch_id

    @property
    def switch_type(self) -> SwitchType:
        return self.hardware.switch_type

    @property
    def layer(self) -> SwitchLayer:
        return self.config.layer

    @property
    def port_count(self) -> int:
        return self.hardware.port_count

    @property
    def processing_delay_us(self) -> float:
        return self.hardware.processing_delay_us

    @property
    def port_bandwidth_gbps(self) -> float:
        return self.hardware.port_bandwidth_gbps

    def get_output_port(self, next_hop: str) -> int:
        """获取到下一跳设备的输出端口号"""
        return self.config.device_to_port.get(next_hop, -1)

    def get_port_state(self, port_number: int) -> SwitchPortState:
        """获取端口状态（不存在则创建）"""
        if port_number not in self.port_states:
            port_id = f"{self.switch_id}:port_{port_number}"
            connected_device = self.config.port_to_device.get(port_number, "")
            self.port_states[port_number] = SwitchPortState(
                port_id=port_id,
                switch_id=self.switch_id,
                port_number=port_number,
                connected_device=connected_device,
                link_bandwidth_gbps=self.port_bandwidth_gbps,
            )
        return self.port_states[port_number]

    def get_serialization_delay_us(self, packet_size_bytes: float) -> float:
        """计算串行化延迟（数据包传输时间）

        Args:
            packet_size_bytes: 数据包大小（字节）

        Returns:
            延迟（微秒）
        """
        # 带宽单位: GB/s = 1e9 bytes/s = 1e3 bytes/us
        rate_bytes_per_us = self.port_bandwidth_gbps * 1e9 / 8 / 1e6
        return packet_size_bytes / rate_bytes_per_us


class SwitchManager:
    """Switch 管理器

    负责管理网络中所有 Switch 节点，提供：
    - Switch 拓扑构建和解析
    - 路由计算（多跳路径）
    - 端口资源管理
    """

    def __init__(
        self,
        topology_parser: Optional[TopologyParser] = None,
        switch_graph: Optional[SwitchGraph] = None,
    ):
        """初始化 Switch 管理器

        Args:
            topology_parser: 拓扑解析器（用于从拓扑配置构建 Switch）
            switch_graph: 已构建的 Switch 图（可选）
        """
        # Switch 节点（switch_id → SwitchNode）
        self.switches: dict[str, SwitchNode] = {}

        # Switch 硬件配置
        self.hardware_configs: dict[str, SwitchHardwareConfig] = DEFAULT_SWITCH_CONFIGS.copy()

        # 路由表缓存（(src_chip, dst_chip) → [switch_ids]）
        self.routing_table: dict[tuple[str, str], list[str]] = {}

        # 芯片到 Switch 的连接（chip_id → switch_id）
        self.chip_to_switch: dict[str, str] = {}

        # Switch 之间的连接（(switch_id, switch_id) → (bandwidth, latency)）
        self.switch_connections: dict[tuple[str, str], tuple[float, float]] = {}

        # 拓扑解析器引用
        self._topology_parser = topology_parser

        # 如果提供了 switch_graph，直接使用
        if switch_graph:
            self._build_from_switch_graph(switch_graph)
        elif topology_parser:
            self._build_from_topology()

    def _build_from_switch_graph(self, switch_graph: SwitchGraph) -> None:
        """从 SwitchGraph 构建 Switch 节点"""
        # 加载硬件配置
        for name, config in switch_graph.hardware_configs.items():
            self.hardware_configs[name] = config

        # 创建 Switch 节点
        for switch_config in switch_graph.switches:
            hardware = self.hardware_configs.get(switch_config.hardware_type)
            if not hardware:
                logger.warning(
                    f"未找到 Switch 硬件配置: {switch_config.hardware_type}，"
                    f"使用默认 leaf_72"
                )
                hardware = self.hardware_configs["leaf_72"]

            node = SwitchNode(config=switch_config, hardware=hardware)
            self.switches[switch_config.switch_id] = node

        # 加载连接关系
        self.switch_connections = switch_graph.connections.copy()

    def _build_from_topology(self) -> None:
        """从拓扑配置自动生成 Switch

        根据拓扑层级自动创建 Switch：
        - 每个 Board 连接一个 Leaf Switch
        - 同 Rack 的 Leaf Switch 通过 Spine Switch 连接
        - 跨 Rack 的通信通过多级 Switch
        """
        if not self._topology_parser:
            return

        topology = self._topology_parser.topology
        interconnect_params = self._topology_parser.interconnect_params

        switch_id_counter = 0

        # 遍历拓扑，为每个 Board 创建 Leaf Switch
        for pod in topology.pods:
            for rack in pod.racks:
                # 为 Rack 创建一个 Spine Switch
                spine_id = f"spine_{rack.id}"
                spine_config = SwitchInstanceConfig(
                    switch_id=spine_id,
                    hardware_type="spine_512",
                    layer=SwitchLayer.INTER_BOARD,
                    pod_id=pod.id,
                    rack_id=rack.id,
                )
                spine_hardware = self.hardware_configs["spine_512"]
                self.switches[spine_id] = SwitchNode(
                    config=spine_config, hardware=spine_hardware
                )

                port_counter = 0
                for board in rack.boards:
                    # 为每个 Board 创建一个 Leaf Switch
                    leaf_id = f"leaf_{board.id}"
                    leaf_config = SwitchInstanceConfig(
                        switch_id=leaf_id,
                        hardware_type="leaf_128",
                        layer=SwitchLayer.INTER_CHIP,
                        pod_id=pod.id,
                        rack_id=rack.id,
                    )
                    leaf_hardware = self.hardware_configs["leaf_128"]

                    # 建立 Chip → Leaf Switch 的连接
                    chip_port = 0
                    for chip in board.chips:
                        leaf_config.port_to_device[chip_port] = chip.id
                        leaf_config.device_to_port[chip.id] = chip_port
                        self.chip_to_switch[chip.id] = leaf_id
                        chip_port += 1

                    # 建立 Leaf → Spine 的连接
                    uplink_port = chip_port
                    leaf_config.port_to_device[uplink_port] = spine_id
                    leaf_config.device_to_port[spine_id] = uplink_port

                    # Spine → Leaf 的连接
                    spine_config.port_to_device[port_counter] = leaf_id
                    spine_config.device_to_port[leaf_id] = port_counter

                    self.switches[leaf_id] = SwitchNode(
                        config=leaf_config, hardware=leaf_hardware
                    )

                    # 记录 Switch 之间的连接
                    b2b_params = interconnect_params.get("b2b", {})
                    bw = b2b_params.get("bandwidth_gbps", 400.0)
                    lat = b2b_params.get("latency_us", 2.0)
                    self.switch_connections[(leaf_id, spine_id)] = (bw, lat)
                    self.switch_connections[(spine_id, leaf_id)] = (bw, lat)

                    port_counter += 1
                    switch_id_counter += 1

        logger.info(f"从拓扑自动生成 {len(self.switches)} 个 Switch 节点")

    def add_hardware_config(self, config: SwitchHardwareConfig) -> None:
        """添加 Switch 硬件配置"""
        self.hardware_configs[config.name] = config

    def add_switch(self, switch_config: SwitchInstanceConfig) -> SwitchNode:
        """添加 Switch 实例"""
        hardware = self.hardware_configs.get(switch_config.hardware_type)
        if not hardware:
            raise ValueError(f"未知的 Switch 硬件类型: {switch_config.hardware_type}")

        node = SwitchNode(config=switch_config, hardware=hardware)
        self.switches[switch_config.switch_id] = node
        return node

    def get_switch(self, switch_id: str) -> Optional[SwitchNode]:
        """获取 Switch 节点"""
        return self.switches.get(switch_id)

    def compute_route(self, src_chip: str, dst_chip: str) -> list[str]:
        """计算从源芯片到目标芯片的路由路径

        Args:
            src_chip: 源芯片 ID
            dst_chip: 目标芯片 ID

        Returns:
            经过的 Switch ID 列表（按顺序）
            如果是同板芯片（直连），返回空列表
        """
        # 检查缓存
        cache_key = (src_chip, dst_chip)
        if cache_key in self.routing_table:
            return self.routing_table[cache_key]

        # 获取芯片连接的 Leaf Switch
        src_leaf = self.chip_to_switch.get(src_chip)
        dst_leaf = self.chip_to_switch.get(dst_chip)

        # 如果芯片没有 Switch 连接（可能是直连），返回空路径
        if not src_leaf or not dst_leaf:
            self.routing_table[cache_key] = []
            return []

        # 如果连接到同一个 Leaf Switch，只经过一个 Switch
        if src_leaf == dst_leaf:
            route = [src_leaf]
            self.routing_table[cache_key] = route
            return route

        # 不同 Leaf Switch，需要经过 Spine
        # 查找两个 Leaf 的公共 Spine
        src_spine = self._find_uplink_switch(src_leaf)
        dst_spine = self._find_uplink_switch(dst_leaf)

        if src_spine == dst_spine and src_spine:
            # 同 Rack，经过一个 Spine
            route = [src_leaf, src_spine, dst_leaf]
        else:
            # 跨 Rack，可能需要更多跳（简化处理：假设有 Core Switch）
            # TODO: 支持更复杂的多级拓扑
            route = [src_leaf, src_spine, dst_spine, dst_leaf] if src_spine and dst_spine else [src_leaf, dst_leaf]

        # 过滤 None
        route = [s for s in route if s]

        self.routing_table[cache_key] = route
        return route

    def _find_uplink_switch(self, switch_id: str) -> Optional[str]:
        """查找 Switch 的上行连接（Leaf → Spine, Spine → Core）"""
        switch = self.switches.get(switch_id)
        if not switch:
            return None

        # 遍历端口，查找连接到更高层级 Switch 的端口
        for port_num, device_id in switch.config.port_to_device.items():
            other = self.switches.get(device_id)
            if other and self._is_higher_layer(other.layer, switch.layer):
                return device_id

        return None

    def _is_higher_layer(self, layer1: SwitchLayer, layer2: SwitchLayer) -> bool:
        """判断 layer1 是否比 layer2 更高层级"""
        layer_order = {
            SwitchLayer.INTER_CHIP: 0,
            SwitchLayer.INTER_BOARD: 1,
            SwitchLayer.INTER_RACK: 2,
            SwitchLayer.INTER_POD: 3,
        }
        return layer_order.get(layer1, 0) > layer_order.get(layer2, 0)

    def get_switch_count(self) -> int:
        """获取 Switch 数量"""
        return len(self.switches)

    def get_total_ports(self) -> int:
        """获取总端口数"""
        return sum(s.port_count for s in self.switches.values())

    def get_switches_by_type(self, switch_type: SwitchType) -> list[SwitchNode]:
        """按类型获取 Switch 列表"""
        return [s for s in self.switches.values() if s.switch_type == switch_type]

    def get_switches_by_layer(self, layer: SwitchLayer) -> list[SwitchNode]:
        """按层级获取 Switch 列表"""
        return [s for s in self.switches.values() if s.layer == layer]

    def get_output_port(self, switch_id: str, next_hop: str) -> int:
        """获取 Switch 到下一跳的输出端口

        Args:
            switch_id: Switch ID
            next_hop: 下一跳设备 ID（Switch 或 Chip）

        Returns:
            端口号，如果找不到返回 -1
        """
        switch = self.switches.get(switch_id)
        if not switch:
            return -1
        return switch.get_output_port(next_hop)

    def get_serialization_delay(
        self, switch_id: str, packet_size_bytes: float
    ) -> float:
        """计算 Switch 的串行化延迟

        Args:
            switch_id: Switch ID
            packet_size_bytes: 数据包大小（字节）

        Returns:
            串行化延迟（微秒）
        """
        switch = self.switches.get(switch_id)
        if not switch:
            return 0.0
        return switch.get_serialization_delay_us(packet_size_bytes)

    def get_processing_delay(self, switch_id: str) -> float:
        """获取 Switch 的处理延迟

        Args:
            switch_id: Switch ID

        Returns:
            处理延迟（微秒）
        """
        switch = self.switches.get(switch_id)
        if not switch:
            return 0.0
        return switch.processing_delay_us

    def get_hop_delay(
        self, switch_id: str, packet_size_bytes: float
    ) -> float:
        """计算经过 Switch 的总延迟（处理 + 串行化）

        Args:
            switch_id: Switch ID
            packet_size_bytes: 数据包大小（字节）

        Returns:
            总延迟（微秒）
        """
        processing = self.get_processing_delay(switch_id)
        serialization = self.get_serialization_delay(switch_id, packet_size_bytes)
        return processing + serialization

    def create_packet(
        self,
        flow_id: str,
        packet_index: int,
        size_bytes: float,
        src_chip: str,
        dst_chip: str,
        created_at: float,
        layer_index: int = -1,
        micro_batch: int = 0,
        comm_type: str = "",
    ) -> Packet:
        """创建数据包

        Args:
            flow_id: 通信流 ID
            packet_index: 包索引
            size_bytes: 包大小
            src_chip: 源芯片
            dst_chip: 目标芯片
            created_at: 创建时间
            layer_index: 层索引
            micro_batch: 微批次
            comm_type: 通信类型

        Returns:
            创建的数据包
        """
        route = self.compute_route(src_chip, dst_chip)

        return Packet(
            packet_id=f"{flow_id}_pkt_{packet_index}",
            flow_id=flow_id,
            size_bytes=size_bytes,
            src_chip=src_chip,
            dst_chip=dst_chip,
            current_location=src_chip,
            route=route,
            hop_index=0,
            created_at=created_at,
            last_hop_time=created_at,
            layer_index=layer_index,
            micro_batch=micro_batch,
            comm_type=comm_type,
        )

    def get_statistics(self) -> dict[str, Any]:
        """获取 Switch 统计信息"""
        stats = {
            "total_switches": len(self.switches),
            "total_ports": self.get_total_ports(),
            "switches_by_type": {},
            "switches_by_layer": {},
            "cached_routes": len(self.routing_table),
        }

        # 按类型统计
        for switch_type in SwitchType:
            count = len(self.get_switches_by_type(switch_type))
            if count > 0:
                stats["switches_by_type"][switch_type.value] = count

        # 按层级统计
        for layer in SwitchLayer:
            count = len(self.get_switches_by_layer(layer))
            if count > 0:
                stats["switches_by_layer"][layer.value] = count

        return stats

    def reset(self) -> None:
        """重置所有运行时状态"""
        # 清除端口状态
        for switch in self.switches.values():
            switch.port_states.clear()

        # 清除路由缓存
        self.routing_table.clear()

        logger.debug("SwitchManager 状态已重置")
