"""
Quick sort trace/no trace test script.
==============================
* Quick sort without trace: TEST: PASSED
* Quick sort with trace: TEST: PASSED
"""
import random

from util import quick_sort
from util import quick_sort_with_trace

if __name__ == "__main__":
    length = random.randint(16, 1024)
    old_values = [random.randint(1, 4 * length) for i in range(length)]  # create random array to sort, as archive,
    # as sort destroys the original array
    values = old_values.copy()  # real array to sort generated from the archive
    quick_sort(values)  # sort the array
    success = True  # check if the array was sorted correctly
    for i in range(len(values) - 1):
        if values[i] > values[i + 1]:  # if smaller index has larger value, the sort is a fail
            success = False
            break
    if success:
        print('quick_sort test without trace, TEST: PASSED')
    else:
        print('quick_sort test without trace, TEST: FAILED')

    values = old_values.copy()  # reload the array
    trace = [i for i in range(len(values))]  # generate the trace array, in this case it is just the index array
    old_trace = trace.copy()  # save the trace array
    quick_sort_with_trace(values, trace)  # sort the array with its trace

    success = True  # check if the array was sorted correctly. The order was checked above already
    for i in range(len(values)):  # here we check if the trace followed the swaps correctly
        if values[i] != old_values[trace[i]]:  # check if the trace value of all values are valid
            success = False
            break
    if success:
        print('quick_sort test with trace, TEST: PASSED')
    else:
        print('quick_sort test with trace, TEST: FAILED')
