from typing import Literal
import pytest
from affmap import AffMap
import numpy as np

def test_validation() -> None:
    """
    Test that pydantic will not allow keys which are not a subset of the input keys
    """
    T_Dims = Literal['a', 'b', 'c']
    AffCls = AffMap[T_Dims, T_Dims]
    with pytest.raises(ValueError):
        AffCls(
            translation={'1': 1, '2': 2}, 
            affine=
                {'a': 
                    {'a': 1}, 
                 'b': {'b': 1}}
                )
        
def test_basic():
    from affmap.symmetric import AffMapSym
    affmap = AffMapSym(affine={'a': {'a': 10}}, translation={'a': 0})
    assert affmap.flatten(axes=('a',)) == (10,  0)
    assert affmap.axes == {'a'}
    assert np.array_equal(
        affmap.to_array(axes=('a',)), 
        np.array(affmap.flatten(axes=affmap.axes)).reshape(1, 2))