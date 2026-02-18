"""FlashAttention-2 fused operator implementation."""

from __future__ import annotations

from math_model.L0_entry.types import DataType
from math_model.L1_workload.operators import op_registry
from math_model.L1_workload.operators.base import OpBase, OpRole
from math_model.L1_workload.operators.utils import get_first_int
from math_model.L1_workload.tensor import TensorDesc


@op_registry.register("fa2")
class FA2Op(OpBase):
    """FlashAttention-2 fused attention operator.

    Fuses QK^T + softmax + PV into a single tiled operation.

    Shape keys:
        B  - batch * num_heads (head-parallel dimension)
        QS - query sequence length
        KS - key/value sequence length
        QD - query/key head dimension
        VD - value head dimension

    FLOPs:
        QK^T: 2 * B * QS * KS * QD
        PV:   2 * B * QS * KS * VD
        Total: 2 * B * QS * KS * (QD + VD)
    """

    def __init__(
        self,
        name: str,
        parallel_params: dict[str, int],
        *,
        dtype: DataType = DataType.FP16,
        weight_dtype: DataType | None = None,
        output_dtype: DataType | None = None,
        accum_dtype: DataType | None = None,
    ):
        super().__init__(name, dtype=dtype)
        params = parallel_params or {}
        for key_group, label in [
            (("B", "b"), "B (batch*heads)"),
            (("QS", "qs", "q_seq_len"), "QS (query seq len)"),
            (("KS", "ks", "kv_seq_len"), "KS (kv seq len)"),
            (("QD", "qd", "q_dim"), "QD (query dim)"),
            (("VD", "vd", "v_dim"), "VD (value dim)"),
        ]:
            if not any(k in params for k in key_group):
                raise ValueError(
                    f"Missing required FA2 parameter '{label}' in op '{name}'. "
                    f"Available keys: {list(params.keys())}"
                )
        self._b = get_first_int(params, ("B", "b"), 0)
        self._qs = get_first_int(params, ("QS", "qs", "q_seq_len"), 0)
        self._ks = get_first_int(params, ("KS", "ks", "kv_seq_len"), 0)
        self._qd = get_first_int(params, ("QD", "qd", "q_dim"), 0)
        self._vd = get_first_int(params, ("VD", "vd", "v_dim"), 0)
        self._weight_dtype = weight_dtype or dtype
        self._output_dtype = output_dtype or dtype
        self._accum_dtype = accum_dtype or DataType.FP32

    @property
    def op_type(self) -> str:
        return "fa2"

    @property
    def role(self) -> OpRole:
        return OpRole.CUBE

    def get_inputs(self) -> list[TensorDesc]:
        # Q: [B, QS, QD], K: [B, KS, QD], V: [B, KS, VD]
        return [
            TensorDesc(name="Q", shape=[self._b, self._qs, self._qd], dtype=self._dtype),
            TensorDesc(name="K", shape=[self._b, self._ks, self._qd], dtype=self._dtype),
            TensorDesc(name="V", shape=[self._b, self._ks, self._vd], dtype=self._dtype),
        ]

    def get_outputs(self) -> list[TensorDesc]:
        # O: [B, QS, VD]
        return [
            TensorDesc(name="output", shape=[self._b, self._qs, self._vd], dtype=self._output_dtype),
        ]

    def compute_ops(self) -> int:
        # QK^T: 2*B*QS*KS*QD + PV: 2*B*QS*KS*VD
        return 2 * self._b * self._qs * self._ks * (self._qd + self._vd)

    def compute_memory_access(self) -> tuple[int, int]:
        # Read: Q + K + V
        q_bytes = self._b * self._qs * self._qd * self._dtype.bytes
        k_bytes = self._b * self._ks * self._qd * self._dtype.bytes
        v_bytes = self._b * self._ks * self._vd * self._dtype.bytes
        read_bytes = q_bytes + k_bytes + v_bytes
        # Write: O
        write_bytes = self._b * self._qs * self._vd * self._output_dtype.bytes
        return read_bytes, write_bytes
