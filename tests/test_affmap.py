from typing import Literal
import pytest
from affmap import AffMap

def test_validation() -> None:
    """
    Test that pydantic will not allow keys which are not a subset of the input keys
    """
    T_Dims = Literal['a', 'b', 'c']
    AffCls = AffMap[T_Dims]
    with pytest.raises(ValueError):
        AffCls(
            translation={'1': 1, '2': 2}, 
            affine=
                {'a': 
                    {'a': 1}, 
                 'b': {'b': 1}}
                )