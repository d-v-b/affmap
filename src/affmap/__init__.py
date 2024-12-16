from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Generic, Hashable, Literal, TypeAlias, TypeVar
from pydantic import BaseModel, model_validator
import itertools

T = TypeVar('T', bound=Hashable)
V = TypeVar('V', bound=Hashable)

T_AxesIn = TypeVar("T_AxesIn", bound=Hashable)
T_AxesOut = TypeVar("T_AxesOut", bound=Hashable)

VectorMap: TypeAlias = Mapping[T, float]
MatrixMap: TypeAlias = Mapping[T, VectorMap[V]]

def ensure_same_output_axes(data: AffMap):
    trans_key_set = set(data.translation.keys())
    aff_key_set = set(data.affine.keys())
    if trans_key_set != aff_key_set:
        msg = (
            'The translation and affine attributes must have the exact same keys.'
            f'Got keys {trans_key_set} for translation, and {aff_key_set} for affine.'
            'They are not identical.'
            )
        raise ValueError(msg) 
    
def ensure_same_input_axes(data: AffMap):
    aff_key_sets = {k: set(v.keys()) for k,v in data.affine.items()}
    problems: dict[tuple[Hashable, Hashable], tuple[Hashable, Hashable]] = {}
    for (a_key, a_val), (b_key, b_val) in itertools.product(aff_key_sets.items()):
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

class AffMap(BaseModel, Generic[T_AxesIn, T_AxesOut]):
    """
    Model a homogeneous affine transformation with named axes. The transform is decomposed into
    a translation transform and an affine transform.
    """

    translation: VectorMap[T_AxesOut]
    affine: MatrixMap[T_AxesOut, T_AxesIn]

    _ensure_same_output_axes = model_validator(mode='after')(ensure_same_output_axes)
    _ensure_same_input_axes = model_validator(mode='after')(ensure_same_input_axes)

    @property
    def axes(self) -> dict[Literal['in', 'out'], T_AxesIn, T_AxesOut]:
        return {
            'in': tuple(self.translation.keys()),
            'out': tuple(self.translation.keys())
            }

    def flatten(self, axes_out: Iterable[T_AxesOut], axes_in: Iterable[T_AxesIn]) -> tuple[float, ...]:
        out: tuple[float, ...] = ()
        for ax_o in axes_out:
            for ax_i in axes_in:
                out += (self.affine[ax_o][ax_i],)
            out += (self.translation[ax_o],)
        return out
