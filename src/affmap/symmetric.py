from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Generic, Hashable, Literal, Self, TypeAlias, TypeVar
from pydantic import BaseModel, model_validator
import itertools
import numpy as np

T = TypeVar('T', bound=Hashable)
V = TypeVar('V', bound=Hashable)

T_Axes = TypeVar("T_Axes", bound=Hashable)

VectorMap: TypeAlias = Mapping[T, float]
MatrixMap: TypeAlias = Mapping[T, VectorMap[V]]

def _ensure_same_output_axes(data: AffMapSym) -> AffMapSym:
    trans_key_set = set(data.translation.keys())
    aff_key_set = set(data.affine.keys())
    if trans_key_set != aff_key_set:
        msg = (
            'The translation and affine attributes must have the exact same keys.'
            f'Got keys {trans_key_set} for translation, and {aff_key_set} for affine.'
            'They are not identical.'
            )
        raise ValueError(msg) 
    return data
def _ensure_same_input_axes(data: AffMapSym) -> AffMapSym:
    aff_key_sets = {k: set(v.keys()) for k,v in data.affine.items()}
    problems: dict[tuple[Hashable, Hashable], tuple[Hashable, Hashable]] = {}

    for (a_key, a_val) in aff_key_sets.items():
        for (b_key, b_val) in aff_key_sets.items():
            # todo: remove the upper triangle of this matrix
                if a_key != b_key and a_val != b_val:
                    problems[(a_key, b_key)] = (a_val, b_val)
    if len(problems) > 0:
        # Todo: better message
        msg = (
            'All of the components of the affine transform must have the exact same keys.'
            'The following components have mismatched keys:'
            f'{tuple(problems.keys())}'
            )
        raise ValueError(msg)
    return data

class AffMapSym(BaseModel, Generic[T_Axes]):
    """
    Model a homogeneous affine transformation with named axes, where the input space 
    and the output space have the same axis names. The transform is decomposed into
    a translation transform and an affine transform.
    """

    translation: VectorMap[T_Axes]
    affine: MatrixMap[T_Axes, T_Axes]

    @model_validator(mode='after')
    def _ensure_same_output_axes(data: Any) -> Any:
        return _ensure_same_output_axes(data)
    
    @model_validator(mode='after')
    def _ensure_same_input_axes(data: Any) -> Any:
        return _ensure_same_input_axes(data)

    @property
    def axes(self) -> T_Axes:
        return set(self.translation.keys())
    
    def flatten(self, axes: Iterable[T_Axes]) -> tuple[float, ...]:
        """
        Return the values of the transformation as a tuple of floats. The order of the 
        resulting tuple is based on the `axes` parameter.
        """
        out: tuple[float, ...] = ()
        for ax_o in axes:
            _aff = self.affine[ax_o]
            _trans = self.translation[ax_o]
            for ax_i in axes:
                out += (_aff[ax_i],)
            out += (_trans,)
        return out

    def to_array(self, axes: Iterable[T_Axes]) -> np.ndarray:
        """
        Return the values of the transformation as a numpy array. Parametrized by an axis order.
        """
        return np.array(self.flatten(axes=axes)).reshape((len(self.axes), len(self.axes) + 1))