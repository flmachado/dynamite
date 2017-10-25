
from . import config
from .backend import backend as bck
import numpy as np

def idx_to_state(idx, L = None, sz = None):
    """
    Get the states, represented as integers, which correspond to the
    index or indices supplied in ``idx``.

    Parameters
    ----------

    idx : int or np.ndarray
        The input index or indices

    L : int, optional
        The length of the spin chain. Must be supplied if ``config.global_L``
        is not set.

    sz : int, optional
        The number of up spins in the spin-conserving subspace. Must be supplied
        if ``config.global_sz`` is not set.

    Returns
    -------

    np.ndarray
        An array containing the states
    """
    if L is None:
        if config.global_L is None:
            raise ValueError('Must supply spin chain length, or set it with config.global_L.')
        L = config.global_L

    if sz is None:
        if config.global_sz is None:
            raise ValueError('Must supply sz, or set it with config.global_sz.')
        sz = config.global_sz

    if not isinstance(idx,np.ndarray):
        idx = np.array(idx,ndmin=1,dtype=bck.MSC_dtype[0])

    return bck.map_forward(L,sz,idx.astype(bck.MSC_dtype[0]))


def state_to_idx(state, L = None, sz = None):
    """
    Get the index or indices corresponding to the state or
    states in ``state``.

    Parameters
    ----------

    state : int or np.ndarray
        The input state or states

    L : int, optional
        The length of the spin chain. Must be supplied if ``config.global_L``
        is not set.

    sz : int, optional
        The number of up spins in the spin-conserving subspace. Must be supplied
        if ``config.global_sz`` is not set.

    Returns
    -------
    np.ndarray
        An array containing the indices
    """
    if L is None:
        if config.global_L is None:
            raise ValueError('Must supply spin chain length, or set it with config.global_L.')
        L = config.global_L

    if sz is None:
        if config.global_sz is None:
            raise ValueError('Must supply sz, or set it with config.global_sz.')
        sz = config.global_sz

    if not isinstance(state,np.ndarray):
        state = np.array(state,ndmin=1,dtype=bck.MSC_dtype[0])

    return bck.map_reverse(L,sz,state.astype(bck.MSC_dtype[0]))


def subspace_dim(L = None,sz = None):
    """
    Return the dimension of the subspace with chain length L
    and number of up spins sz.

    Parameters
    ----------

    L : int, optional
        The length of the spin chain. Must be supplied if ``config.global_L``
        is not set.

    sz : int, optional
        The number of up spins in the spin-conserving subspace. Must be supplied
        if ``config.global_sz`` is not set.

    Returns
    -------
    int
        The size of the subspace
    """
    if L is None:
        if config.global_L is None:
            raise ValueError('Must supply spin chain length, or set it with config.global_L.')
        L = config.global_L

    if sz is None:
        if config.global_sz is None:
            raise ValueError('Must supply sz, or set it with config.global_sz.')
        sz = config.global_sz

    return bck.max_idx(L,sz)
