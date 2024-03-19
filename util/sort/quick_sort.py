from util import log


def quick_sort(values):
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


def quick_sort_with_trace(values, trace):
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
