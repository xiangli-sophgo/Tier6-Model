"""拓扑配置格式工具

统一的 grouped_pods 格式:
    pods:
      - count: 2
        racks:
          - count: 4
            boards:
              - count: 1
                chips: [{name: SG2262, count: 8}]
"""

from typing import Any


def extract_chip_names(config: dict[str, Any]) -> set[str]:
    """从 grouped_pods 格式提取所有芯片名称

    Args:
        config: 拓扑配置 (必须包含 pods 字段)

    Returns:
        芯片名称集合

    Raises:
        ValueError: 缺少 pods 字段
    """
    pods = config.get("pods")
    if not pods:
        raise ValueError("Missing 'pods' in topology config")

    chip_names: set[str] = set()
    for pod_group in pods:
        for rack_group in pod_group.get("racks", []):
            for board in rack_group.get("boards", []):
                for chip_group in board.get("chips", []):
                    name = chip_group.get("name")
                    if name:
                        chip_names.add(name)
    return chip_names


def count_chips(config: dict[str, Any]) -> int:
    """统计 grouped_pods 格式的芯片总数

    Args:
        config: 拓扑配置 (必须包含 pods 字段)

    Returns:
        芯片总数

    Raises:
        ValueError: 缺少 pods 字段或计算结果为 0
    """
    pods = config.get("pods")
    if not pods:
        raise ValueError("Missing 'pods' in topology config")

    total = 0
    for pod_group in pods:
        pod_count = pod_group.get("count", 1)
        for rack_group in pod_group.get("racks", []):
            rack_count = rack_group.get("count", 1)
            for board in rack_group.get("boards", []):
                board_count = board.get("count", 1)
                for chip_group in board.get("chips", []):
                    chip_count = chip_group.get("count", 1)
                    total += pod_count * rack_count * board_count * chip_count

    if total == 0:
        raise ValueError("Computed 0 chips from topology config")
    return total


def grouped_pods_to_expanded(config: dict[str, Any]) -> dict[str, Any]:
    """grouped_pods -> 完全展开格式 (用于仿真和3D渲染)

    展开所有 count 字段, 为每个实例生成唯一 id.

    Args:
        config: grouped_pods 格式配置

    Returns:
        展开后的配置 (每个 pod/rack/board/chip 有 id, 无 count)
    """
    import copy
    result = copy.deepcopy(config)

    pods_config = result.get("pods", [])
    chips_dict = result.get("chips", {})

    expanded_pods = []
    pod_global_idx = 0

    for pod_group in pods_config:
        pod_count = pod_group.get("count", 1)
        racks_groups = pod_group.get("racks", [])

        for _ in range(pod_count):
            expanded_racks = []
            rack_global_idx = 0

            for rack_group in racks_groups:
                rack_count = rack_group.get("count", 1)
                boards_groups = rack_group.get("boards", [])
                total_u = rack_group.get("total_u", 42)

                for _ in range(rack_count):
                    expanded_boards = []
                    board_global_idx = 0

                    for board_group in boards_groups:
                        board_count = board_group.get("count", 1)
                        chips_groups = board_group.get("chips", [])

                        for _ in range(board_count):
                            expanded_chips = []
                            chip_global_idx = 0

                            for chip_group in chips_groups:
                                chip_name = chip_group.get("name", "unknown")
                                chip_count = chip_group.get("count", 1)
                                chip_params = chips_dict.get(chip_name, {})

                                for _ in range(chip_count):
                                    chip_id = (
                                        f"pod_{pod_global_idx}/"
                                        f"rack_{rack_global_idx}/"
                                        f"board_{board_global_idx}/"
                                        f"chip_{chip_global_idx}"
                                    )
                                    chip_data: dict[str, Any] = {
                                        "id": chip_id,
                                        "type": "chip",
                                        "name": chip_name,
                                        "position": [chip_global_idx % 4, chip_global_idx // 4],
                                    }
                                    for param_key in (
                                        "num_cores",
                                        "compute_tflops_fp8",
                                        "compute_tflops_bf16",
                                        "memory_capacity_gb",
                                        "memory_bandwidth_gbps",
                                        "memory_bandwidth_utilization",
                                        "lmem_capacity_mb",
                                        "lmem_bandwidth_gbps",
                                        "cube_m", "cube_k", "cube_n",
                                        "sram_size_kb", "sram_utilization",
                                        "lane_num", "align_bytes",
                                        "compute_dma_overlap_rate",
                                    ):
                                        if param_key in chip_params:
                                            chip_data[param_key] = chip_params[param_key]

                                    expanded_chips.append(chip_data)
                                    chip_global_idx += 1

                            board_id = (
                                f"pod_{pod_global_idx}/"
                                f"rack_{rack_global_idx}/"
                                f"board_{board_global_idx}"
                            )
                            expanded_boards.append({
                                "id": board_id,
                                "u_height": board_group.get("u_height", 2),
                                "label": board_group.get("name", "Board"),
                                "chips": expanded_chips,
                            })
                            board_global_idx += 1

                    rack_id = f"pod_{pod_global_idx}/rack_{rack_global_idx}"
                    expanded_racks.append({
                        "id": rack_id,
                        "position": [rack_global_idx % 4, rack_global_idx // 4],
                        "label": f"Rack {rack_global_idx}",
                        "total_u": total_u,
                        "boards": expanded_boards,
                    })
                    rack_global_idx += 1

            expanded_pods.append({
                "id": f"pod_{pod_global_idx}",
                "label": f"Pod {pod_global_idx}",
                "grid_size": [rack_global_idx, 1],
                "racks": expanded_racks,
            })
            pod_global_idx += 1

    result["pods"] = expanded_pods
    return result
