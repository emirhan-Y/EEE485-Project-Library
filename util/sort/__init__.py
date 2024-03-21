"""
sort
====

General sort package. Includes
* Quick sort
* Quick sort with trace

Usage
=====
This package can be used to sort a ``np.ndarray`` type array of any comparable data type. Our project requires the use
of trace arrays in some cases, where another array must have its elements reordered in the same way as the data array.
Note that the operations are destructive, and the original arrays will be lost. To sort an array::

    >>> import numpy as np
    ... from util import sort
    ... x = np.array([1,3,2,5,4])
    ... sort.quick_sort(x)
    ... print(x)
    [1 2 3 4 5]

To sort an array with a trace array::

    >>> import numpy as np
    ... from util import sort
    ... x = np.array([2,3,1,5,4])
    ... t = np.array([1,2,3,4,5])
    ... sort.quick_sort_with_trace(x, t)
    ... print(x, t)
    [1 2 3 4 5] [3 1 2 5 4]

In this case, the tracer array shows the original locations of the elements of the ordered array. For example, the
element `3` used to be 2nd element of the original array. This is useful in the some cases where the tracer array
elements are strings.
"""

from .quick_sort import quick_sort
from .quick_sort import quick_sort_with_trace
