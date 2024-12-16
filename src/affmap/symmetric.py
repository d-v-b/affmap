from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Generic, Hashable, Literal, TypeAlias, TypeVar
from pydantic import BaseModel, model_validator

T = TypeVar('T', bound=Hashable)
V = TypeVar('V', bound=Hashable)

T_Axes = TypeVar("T_Axes", bound=Hashable)

VectorMap: TypeAlias = Mapping[T, float]
MatrixMap: TypeAlias = Mapping[T, VectorMap[V]]

class AffMapSym(BaseModel, Generic[T_Axes]):
    """
    Model a homogeneous affine transformation with named axes. The transform is decomposed into
    a translation transform and an affine transform.
    """

    translation: VectorMap[T_Axes]
    affine: MatrixMap[T_Axes, T_Axes]

    _ensure_same_output_axes = model_validator(mode='after')(ensure_same_output_axes)
    _ensure_same_input_axes = model_validator(mode='after')(ensure_same_input_axes)

    @property
    def axes(self):
        return tuple(self.translation.keys())
    

    def flatten(self, axes: Iterable[T_Axes]) -> tuple[float, ...]:
        out: tuple[float, ...] = ()
        for ax_o in axes:
            for ax_i in axes:
                out += (self.affine[ax_o][ax_i],)
            out += (self.translation[ax_o],)
        return out
