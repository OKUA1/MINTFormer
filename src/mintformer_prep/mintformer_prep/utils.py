import functools
from typing import Dict
try:
    from numba import njit as _njit, typeof
    _has_numba = True
    dict_type = typeof({})
except Exception:
    _has_numba = False
    dict_type = Dict

def numba_wrapper(func):
    if _has_numba:
        # Apply Numba njit if available
        compiled_func = _njit(func)
    else:
        # If Numba is not installed, use the original function
        print("Numba is not available")
        compiled_func = func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return compiled_func(*args, **kwargs)

    return wrapper
