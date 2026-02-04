"""链路流量分析模块

生成网络链路流量统计和可视化数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tier6.L3_mapping.plan import ExecPlan, ExecStep
    from tier6.L3_mapping.protocols import ParallelGroupAssignment


class LinkType(str, Enum):
    """链路类型"""

    C2C = "c2c"  # Chip-to-Chip (NVLink)
    B2B = "b2b"  # Board-to-Board (PCIe)
    R2R = "r2r"  # Rack-to-Rack (InfiniBand)
    P2P = "p2p"  # Pod-to-Pod (Ethernet)


class CommType(str, Enum):
    """通信类型"""

    TP_ALLREDUCE = "tp_allreduce"
    PP_P2P = "pp_p2p"
    DP_ALLREDUCE = "dp_allreduce"
    EP_ALLTOALL = "ep_alltoall"
    SP_ALLGATHER = "sp_allgather"
    SP_REDUCESCATTER = "sp_reducescatter"


@dataclass
class LinkTraffic:
    """链路流量

    Attributes:
        src: 源设备 ID
        dst: 目标设备 ID
        link_type: 链路类型
        total_bytes: 总传输字节数
        total_time_us: 总传输时间 (us)
        bandwidth_gbps: 带宽 (Gbps)
        utilization: 带宽利用率 (0-1)
        comm_breakdown: 按通信类型分解
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
            "links": [l.to_dict() for l in self.links],
            "devices": [d.to_dict() for d in self.devices],
            "commBreakdown": self.comm_breakdown,
            "phaseBreakdown": self.phase_breakdown,
        }


class TrafficAnalyzer:
    """流量分析器

    分析执行计划中的通信流量。
    """

    def __init__(self) -> None:
        """初始化"""
        self._link_traffic: dict[str, LinkTraffic] = {}
        self._device_traffic: dict[str, DeviceTraffic] = {}
        self._comm_breakdown: dict[str, int] = {}
        self._phase_breakdown: dict[str, int] = {}
        self._total_bytes = 0
        self._total_time_us = 0.0

    def _get_link_type(self, src: str, dst: str) -> LinkType:
        """推断链路类型

        简化实现：根据设备 ID 前缀判断

        Args:
            src: 源设备 ID
            dst: 目标设备 ID

        Returns:
            LinkType: 链路类型
        """
        # 简化实现：假设 ID 格式为 "pod_rack_board_chip"
        # 实际应根据拓扑信息判断
        src_parts = src.split("_")
        dst_parts = dst.split("_")

        if len(src_parts) >= 4 and len(dst_parts) >= 4:
            if src_parts[:3] == dst_parts[:3]:  # 同 board
                return LinkType.C2C
            if src_parts[:2] == dst_parts[:2]:  # 同 rack
                return LinkType.B2B
            if src_parts[0] == dst_parts[0]:  # 同 pod
                return LinkType.R2R

        return LinkType.P2P

    def _get_link(self, src: str, dst: str) -> LinkTraffic:
        """获取或创建链路流量对象"""
        link_id = f"{src}->{dst}"
        if link_id not in self._link_traffic:
            self._link_traffic[link_id] = LinkTraffic(
                src=src,
                dst=dst,
                link_type=self._get_link_type(src, dst),
            )
        return self._link_traffic[link_id]

    def _get_device(self, device_id: str) -> DeviceTraffic:
        """获取或创建设备流量对象"""
        if device_id not in self._device_traffic:
            self._device_traffic[device_id] = DeviceTraffic(device_id=device_id)
        return self._device_traffic[device_id]

    def add_comm(
        self,
        src: str,
        dst: str,
        bytes_transferred: int,
        time_us: float,
        comm_type: str,
        phase: str,
    ) -> None:
        """添加一次通信

        Args:
            src: 源设备 ID
            dst: 目标设备 ID
            bytes_transferred: 传输字节数
            time_us: 传输时间 (us)
            comm_type: 通信类型
            phase: 阶段 (prefill/decode)
        """
        # 更新链路流量
        link = self._get_link(src, dst)
        link.total_bytes += bytes_transferred
        link.total_time_us += time_us
        link.comm_breakdown[comm_type] = (
            link.comm_breakdown.get(comm_type, 0) + bytes_transferred
        )

        # 更新设备流量
        src_device = self._get_device(src)
        src_device.total_send_bytes += bytes_transferred
        src_device.send_breakdown[comm_type] = (
            src_device.send_breakdown.get(comm_type, 0) + bytes_transferred
        )

        dst_device = self._get_device(dst)
        dst_device.total_recv_bytes += bytes_transferred
        dst_device.recv_breakdown[comm_type] = (
            dst_device.recv_breakdown.get(comm_type, 0) + bytes_transferred
        )

        # 更新总计
        self._total_bytes += bytes_transferred
        self._total_time_us += time_us
        self._comm_breakdown[comm_type] = (
            self._comm_breakdown.get(comm_type, 0) + bytes_transferred
        )
        self._phase_breakdown[phase] = (
            self._phase_breakdown.get(phase, 0) + bytes_transferred
        )

    def analyze(self, exec_plan: "ExecPlan") -> TrafficReport:
        """分析执行计划中的流量

        Args:
            exec_plan: 执行计划

        Returns:
            TrafficReport: 流量报告
        """
        # 分析 prefill 步骤
        for step in exec_plan.prefill_steps:
            if step.comm and step.comm.bytes_to_transfer > 0:
                self._analyze_step(step, "prefill")

        # 分析 decode 步骤
        for step in exec_plan.decode_steps:
            if step.comm and step.comm.bytes_to_transfer > 0:
                self._analyze_step(step, "decode")

        return self._build_report()

    def _analyze_step(self, step: "ExecStep", phase: str) -> None:
        """分析单个步骤"""
        comm = step.comm
        if not comm:
            return

        bytes_transferred = comm.bytes_to_transfer
        time_us = step.estimated_ns / 1000 if step.estimated_ns > 0 else 0.0
        comm_type = comm.comm_type.value

        # P2P 通信
        if comm.src_device and comm.dst_device:
            self.add_comm(
                src=comm.src_device,
                dst=comm.dst_device,
                bytes_transferred=bytes_transferred,
                time_us=time_us,
                comm_type=comm_type,
                phase=phase,
            )
        # 组通信 (AllReduce, AllGather, AllToAll)
        elif comm.group_devices and len(comm.group_devices) > 1:
            # 简化：假设 Ring 拓扑，每个设备向下一个发送
            devices = comm.group_devices
            per_device_bytes = bytes_transferred // len(devices)
            per_device_time = time_us / len(devices)

            for i in range(len(devices)):
                src = devices[i]
                dst = devices[(i + 1) % len(devices)]
                self.add_comm(
                    src=src,
                    dst=dst,
                    bytes_transferred=per_device_bytes,
                    time_us=per_device_time,
                    comm_type=comm_type,
                    phase=phase,
                )

    def _build_report(self) -> TrafficReport:
        """构建报告"""
        return TrafficReport(
            total_bytes=self._total_bytes,
            total_time_us=self._total_time_us,
            links=list(self._link_traffic.values()),
            devices=list(self._device_traffic.values()),
            comm_breakdown=self._comm_breakdown,
            phase_breakdown=self._phase_breakdown,
        )


def analyze_traffic_from_exec_plan(exec_plan: "ExecPlan") -> TrafficReport:
    """从执行计划分析流量

    Args:
        exec_plan: 执行计划

    Returns:
        TrafficReport: 流量报告
    """
    analyzer = TrafficAnalyzer()
    return analyzer.analyze(exec_plan)
