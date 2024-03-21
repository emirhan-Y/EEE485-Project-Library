"""
Quick sort module

Functions
---------
* quick_sort(values): sorts the given array with comparable items
* quick_sort_with_trace(values, trace): sorts the given array with comparable items alongside a trace array
"""
import numpy as np

from util import log


def quick_sort(values: np.ndarray) -> np.ndarray:
    """
    Sort a given list of comparable values (int, float, str etc.)

    Parameters
    ------------
    values : np.ndarray or list
        List with values to sort

    Returns
    -------
    values : np.ndarray or list
        Sorted version of the original input list (redant as the operations are destructive, and the original variable
        changes anyway)
    """
    pivot = values[0]
    last_index_before_pivot = 0
    for i in range(1, len(values)):
        if values[i] < pivot:
            last_index_before_pivot += 1
            values[last_index_before_pivot], values[i] = values[i], values[last_index_before_pivot]
    values[last_index_before_pivot], values[0] = values[0], values[last_index_before_pivot]
    if last_index_before_pivot != 0:
        values[:last_index_before_pivot] = quick_sort(values[:last_index_before_pivot])
    else:
        pass
    if last_index_before_pivot + 1 < len(values):
        values[last_index_before_pivot + 1:] = quick_sort(values[last_index_before_pivot + 1:])
    else:
        pass
    return values


def quick_sort_with_trace(values: np.ndarray, trace: np.ndarray) -> tuple:
    """
    Sort a given list of comparable values (int, float, str etc.), and apply all same swapping operations to the trace
    array

    Parameters
    ----------
    values : np.ndarray or list
        List with values to sort
    trace : np.ndarray or list
        The tracer list. All swaps applied to the value array will be applied to the trace array

    returns
    ------
    values: np.ndarray or list
        Sorted version of the original input list (redant as the operations are destructive, and the original variable
        changes anyway)
    trace: np.ndarray or list
        The modified trace list (redant as the operations are destructive, and the original variable changes anyway)
    """
    if len(values) != len(trace):
        log.e('Quick sort', 'Trace vector size is not equal to the data vector')
        raise RuntimeError('Quick sort trace size mismatch!')
    pivot = values[0]
    last_index_before_pivot = 0
    for i in range(1, len(values)):
        if values[i] < pivot:
            last_index_before_pivot += 1
            values[last_index_before_pivot], values[i] = values[i], values[last_index_before_pivot]
            trace[last_index_before_pivot], trace[i] = trace[i], trace[last_index_before_pivot]
    values[last_index_before_pivot], values[0] = values[0], values[last_index_before_pivot]
    trace[last_index_before_pivot], trace[0] = trace[0], trace[last_index_before_pivot]
    if last_index_before_pivot != 0:
        values[:last_index_before_pivot], trace[:last_index_before_pivot] = quick_sort_with_trace(
            values[:last_index_before_pivot], trace[:last_index_before_pivot])
    else:
        pass
    if last_index_before_pivot + 1 < len(values):
        values[last_index_before_pivot + 1:], trace[last_index_before_pivot + 1:] = quick_sort_with_trace(
            values[last_index_before_pivot + 1:], trace[last_index_before_pivot + 1:])
    else:
        pass
    return values, trace
