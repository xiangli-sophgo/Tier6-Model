"""链路流量分析模块

基于 L4 StepMetrics 的真实仿真数据生成网络链路流量统计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from perf_model.L4_evaluation.common.metrics import StepMetrics


class LinkType(str, Enum):
    """链路类型"""

    C2C = "c2c"  # Chip-to-Chip (NVLink)
    B2B = "b2b"  # Board-to-Board (PCIe)
    R2R = "r2r"  # Rack-to-Rack (InfiniBand)
    P2P = "p2p"  # Pod-to-Pod (Ethernet)


@dataclass
class LinkTraffic:
    """链路流量

    Attributes:
        src: 源设备 ID (前端格式: pod_X/rack_X/board_X/chip_X)
        dst: 目标设备 ID
        link_type: 链路类型
        total_bytes: 总传输字节数
        total_time_us: 总传输时间 (us)
        bandwidth_gbps: 带宽 (Gbps)
        utilization: 带宽利用率 (0-1)
        comm_breakdown: 按通信类型分解 (reason -> bytes)
    """

    src: str
    dst: str
    link_type: LinkType
    total_bytes: int = 0
    total_time_us: float = 0.0
    bandwidth_gbps: float = 0.0
    utilization: float = 0.0
    comm_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def link_id(self) -> str:
        """链路 ID"""
        return f"{self.src}->{self.dst}"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "linkId": self.link_id,
            "src": self.src,
            "dst": self.dst,
            "linkType": self.link_type.value,
            "totalBytes": self.total_bytes,
            "totalTimeUs": self.total_time_us,
            "bandwidthGbps": self.bandwidth_gbps,
            "utilization": self.utilization,
            "commBreakdown": self.comm_breakdown,
        }


@dataclass
class DeviceTraffic:
    """设备流量

    Attributes:
        device_id: 设备 ID
        total_send_bytes: 总发送字节数
        total_recv_bytes: 总接收字节数
        send_breakdown: 发送分解
        recv_breakdown: 接收分解
    """

    device_id: str
    total_send_bytes: int = 0
    total_recv_bytes: int = 0
    send_breakdown: dict[str, int] = field(default_factory=dict)
    recv_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        """总流量"""
        return self.total_send_bytes + self.total_recv_bytes

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "deviceId": self.device_id,
            "totalSendBytes": self.total_send_bytes,
            "totalRecvBytes": self.total_recv_bytes,
            "totalBytes": self.total_bytes,
            "sendBreakdown": self.send_breakdown,
            "recvBreakdown": self.recv_breakdown,
        }


@dataclass
class TrafficReport:
    """流量报告

    Attributes:
        total_bytes: 总流量字节数
        total_time_us: 总通信时间 (us)
        links: 链路流量列表
        devices: 设备流量列表
        comm_breakdown: 按通信类型分解
        phase_breakdown: 按阶段分解
    """

    total_bytes: int = 0
    total_time_us: float = 0.0
    links: list[LinkTraffic] = field(default_factory=list)
    devices: list[DeviceTraffic] = field(default_factory=list)
    comm_breakdown: dict[str, int] = field(default_factory=dict)
    phase_breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为前端格式"""
        return {
            "totalBytes": self.total_bytes,
            "totalTimeUs": self.total_time_us,
            "links": [link.to_dict() for link in self.links],
            "devices": [d.to_dict() for d in self.devices],
            "commBreakdown": self.comm_breakdown,
            "phaseBreakdown": self.phase_breakdown,
        }


class TrafficAnalyzer:
    """流量分析器

    基于 L4 StepMetrics 的真实仿真数据分析通信流量，
    将芯片索引映射为前端设备 ID，按通信类型分配链路流量。
    """

    def analyze(
        self,
        step_metrics: list[StepMetrics],
        topology_config: dict[str, Any],
        chip_id_mapping: dict[int, str],
    ) -> TrafficReport:
        """分析流量

        Args:
            step_metrics: L4 输出的 StepMetrics 列表 (含增强的 meta 字段)
            topology_config: 拓扑配置 (含 interconnect 信息)
            chip_id_mapping: 芯片索引 -> 前端设备 ID 映射

        Returns:
            TrafficReport: 流量报告
        """
        link_map: dict[str, LinkTraffic] = {}
        device_map: dict[str, DeviceTraffic] = {}
        comm_breakdown: dict[str, int] = {}
        total_bytes = 0
        total_time_us = 0.0

        # 获取互联带宽配置
        if "interconnect" not in topology_config:
            raise ValueError(
                "topology_config 中缺少 'interconnect' 字段"
            )
        interconnect = topology_config["interconnect"]
        if "links" not in interconnect:
            raise ValueError(
                "topology_config.interconnect 中缺少 'links' 字段"
            )
        links_config = interconnect["links"]

        for step in step_metrics:
            if step.t_comm <= 0:
                continue

            meta = step.meta
            if meta.get("evaluator") != "comm":
                continue

            # CommEvaluator 保证以下字段存在
            if "chip_ids" not in meta:
                raise ValueError(f"comm step '{step.op_id}' meta 中缺少 'chip_ids'")
            if "comm_bytes" not in meta:
                raise ValueError(f"comm step '{step.op_id}' meta 中缺少 'comm_bytes'")
            if "comm_type" not in meta:
                raise ValueError(f"comm step '{step.op_id}' meta 中缺少 'comm_type'")
            if "path_key" not in meta:
                raise ValueError(f"comm step '{step.op_id}' meta 中缺少 'path_key'")

            chip_ids: list[int] = meta["chip_ids"]
            comm_bytes: int = meta["comm_bytes"]
            comm_type: str = meta["comm_type"]
            path_key: str = meta["path_key"]
            reason: str = meta.get("reason", comm_type)
            participants: int = meta.get("participants", len(chip_ids))

            if not chip_ids or comm_bytes <= 0:
                continue

            # 获取链路配置
            if path_key not in links_config:
                raise ValueError(
                    f"links_config 中缺少 '{path_key}' 链路配置"
                )
            link_config = links_config[path_key]
            if "bandwidth_gbps" not in link_config:
                raise ValueError(
                    f"links_config.{path_key} 中缺少 'bandwidth_gbps' 字段"
                )
            bandwidth_gbps = link_config["bandwidth_gbps"]

            # 通信时间 (ms -> us)
            comm_time_us = step.t_comm * 1000.0

            # 将芯片索引转换为前端设备 ID
            device_ids = []
            for cid in chip_ids:
                did = chip_id_mapping.get(cid)
                if did is not None:
                    device_ids.append(did)

            if len(device_ids) < 2:
                continue

            # 确定链路类型
            try:
                link_type = LinkType(path_key)
            except ValueError:
                link_type = LinkType.C2C

            # 根据通信类型分配流量到链路
            n = len(device_ids)
            link_entries = self._distribute_traffic(
                comm_type, device_ids, comm_bytes, n,
            )

            # 更新链路和设备统计
            label = reason if reason else comm_type
            for src_id, dst_id, per_link_bytes in link_entries:
                # 链路统计
                lk = f"{src_id}->{dst_id}"
                if lk not in link_map:
                    link_map[lk] = LinkTraffic(
                        src=src_id,
                        dst=dst_id,
                        link_type=link_type,
                        bandwidth_gbps=bandwidth_gbps,
                    )
                link = link_map[lk]
                link.total_bytes += per_link_bytes
                link.total_time_us += comm_time_us / max(len(link_entries), 1)
                link.comm_breakdown[label] = (
                    link.comm_breakdown.get(label, 0) + per_link_bytes
                )

                # 设备发送统计
                if src_id not in device_map:
                    device_map[src_id] = DeviceTraffic(device_id=src_id)
                src_dev = device_map[src_id]
                src_dev.total_send_bytes += per_link_bytes
                src_dev.send_breakdown[label] = (
                    src_dev.send_breakdown.get(label, 0) + per_link_bytes
                )

                # 设备接收统计
                if dst_id not in device_map:
                    device_map[dst_id] = DeviceTraffic(device_id=dst_id)
                dst_dev = device_map[dst_id]
                dst_dev.total_recv_bytes += per_link_bytes
                dst_dev.recv_breakdown[label] = (
                    dst_dev.recv_breakdown.get(label, 0) + per_link_bytes
                )

            # 全局统计
            total_bytes += comm_bytes
            total_time_us += comm_time_us
            comm_breakdown[label] = comm_breakdown.get(label, 0) + comm_bytes

        # 计算利用率
        for link in link_map.values():
            if link.bandwidth_gbps > 0 and link.total_time_us > 0:
                bw_bytes_per_us = link.bandwidth_gbps * 1e9 / 8 / 1e6  # bytes/us
                max_bytes = bw_bytes_per_us * link.total_time_us
                link.utilization = min(1.0, link.total_bytes / max_bytes) if max_bytes > 0 else 0.0

        return TrafficReport(
            total_bytes=total_bytes,
            total_time_us=total_time_us,
            links=list(link_map.values()),
            devices=list(device_map.values()),
            comm_breakdown=comm_breakdown,
        )

    def _distribute_traffic(
        self,
        comm_type: str,
        device_ids: list[str],
        total_bytes: int,
        n: int,
    ) -> list[tuple[str, str, int]]:
        """根据通信类型分配流量到各链路

        Returns:
            list of (src_device_id, dst_device_id, bytes_per_link)
        """
        entries: list[tuple[str, str, int]] = []

        if n <= 1:
            return entries

        # P2P: 直接 src -> dst
        if comm_type == "p2p" or n == 2:
            entries.append((device_ids[0], device_ids[1], total_bytes))
            return entries

        # AllReduce / AllGather / ReduceScatter: Ring 模式
        if comm_type in {"allreduce", "allgather", "reduce_scatter", "reducescatter"}:
            # Ring AllReduce: 每条链路传输 total_bytes * (n-1) / n / n
            # 总共 n 条链路 (环形)，每条链路上的流量
            per_link = int(total_bytes * (n - 1) / n / n) if n > 0 else 0
            for i in range(n):
                src = device_ids[i]
                dst = device_ids[(i + 1) % n]
                entries.append((src, dst, per_link))
            return entries

        # All2All: 全连接模式
        if comm_type in {"all2all", "alltoall"}:
            per_pair = int(total_bytes / (n * (n - 1))) if n > 1 else 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        entries.append((device_ids[i], device_ids[j], per_pair))
            return entries

        # 其他未知类型: 退化为 Ring 模式
        per_link = int(total_bytes / n) if n > 0 else 0
        for i in range(n):
            src = device_ids[i]
            dst = device_ids[(i + 1) % n]
            entries.append((src, dst, per_link))
        return entries
