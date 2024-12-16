from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Generic, Hashable, Literal, TypeAlias, TypeVar
from pydantic import BaseModel, model_validator

T = TypeVar('T', bound=Hashable)
V = TypeVar('V', bound=Hashable)

T_AxesIn = TypeVar("T_AxesIn", bound=Hashable)
T_AxesOut = TypeVar("T_AxesOut", bound=Hashable)

VectorMap: TypeAlias = Mapping[T, float]
MatrixMap: TypeAlias = Mapping[T, VectorMap[V]]

class AffMapSym(BaseModel, Generic[T_AxesIn]):
    """
    Model a homogeneous affine transformation with named axes. The transform is decomposed into
    a translation transform and an affine transform.
    """

    translation: VectorMap[T_AxesIn]
    affine: MatrixMap[T_AxesIn, T_AxesIn]

    _ensure_same_output_axes = model_validator(mode='after')(ensure_same_output_axes)
    _ensure_same_input_axes = model_validator(mode='after')(ensure_same_input_axes)

    @property
    def axes_in(self):
        return tuple(self.translation.keys())
    
        @property
    @property
    def axes_out(self):
        return tuple(self.translation.keys())



    def flatten(self, axes_out: Iterable[T_AxesOut], axes_in: Iterable[T_AxesIn]) -> tuple[float, ...]:
        out: tuple[float, ...] = ()
        for ax_o in axes_out:
            for ax_i in axes_in:
                out += (self.affine[ax_o][ax_i],)
            out += (self.translation[ax_o],)
        return out
