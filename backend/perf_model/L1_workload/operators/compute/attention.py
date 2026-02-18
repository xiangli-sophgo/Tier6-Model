"""Attention-like operators (MHA/MQA/GQA)."""

from __future__ import annotations

from perf_model.L0_entry.types import DataType
from perf_model.L1_workload.operators import op_registry
from perf_model.L1_workload.operators.base import OpBase, OpRole
from perf_model.L1_workload.operators.utils import get_first_int
from perf_model.L1_workload.tensor import TensorDesc


class _AttentionBase(OpBase):
    def __init__(self, name: str, parallel_params: dict[str, int], *, dtype: DataType = DataType.FP16):
        super().__init__(name, dtype=dtype)
        params = parallel_params or {}
        self._b = get_first_int(params, ("B", "b", "batch"), 1)
        self._h = get_first_int(params, ("H", "h", "heads"), 1)
        self._qs = get_first_int(params, ("QS", "q_seq_len", "seq_len"), 1)
        self._ks = get_first_int(params, ("KS", "kv_seq_len"), self._qs)
        self._qd = get_first_int(params, ("QD", "q_dim"), 1)
        self._vd = get_first_int(params, ("VD", "v_dim"), self._qd)

    @property
    def role(self) -> OpRole:
        return OpRole.VECTOR

    def get_inputs(self) -> list[TensorDesc]:
        q_shape = [self._b, self._h, self._qs, self._qd]
        k_shape = [self._b, self._h, self._ks, self._qd]
        v_shape = [self._b, self._h, self._ks, self._vd]
        return [
            TensorDesc(name="q", shape=q_shape, dtype=self._dtype),
            TensorDesc(name="k", shape=k_shape, dtype=self._dtype),
            TensorDesc(name="v", shape=v_shape, dtype=self._dtype),
        ]

    def get_outputs(self) -> list[TensorDesc]:
        out_shape = [self._b, self._h, self._qs, self._vd]
        return [TensorDesc(name="output", shape=out_shape, dtype=self._dtype)]

    def compute_ops(self) -> int:
        return 2 * self._b * self._h * self._qs * self._ks * (self._qd + self._vd)

    def compute_memory_access(self) -> tuple[int, int]:
        read_bytes = sum(t.bytes for t in self.get_inputs())
        write_bytes = sum(t.bytes for t in self.get_outputs())
        return read_bytes, write_bytes


@op_registry.register("mha")
class MHAOp(_AttentionBase):
    @property
    def op_type(self) -> str:
        return "mha"


@op_registry.register("mqa")
class MQAOp(_AttentionBase):
    @property
    def op_type(self) -> str:
        return "mqa"


@op_registry.register("gqa")
class GQAOp(_AttentionBase):
    @property
    def op_type(self) -> str:
        return "gqa"
