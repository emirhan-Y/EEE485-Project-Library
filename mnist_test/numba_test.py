from numba import jit
import time
import numpy as np


@jit(nopython=True)
def sum_of_squares_out(random_array):
    total = 0
    for i in range(len(random_array)):
        total += random_array[i] ** 2
    return total


class HeavyComputation:
    def __init__(self, n):
        self.random_array = np.random.randn(n)

    def sum_of_squares(self):
        total = 0
        for i in range(len(self.random_array)):
            total += self.random_array[i] ** 2
        return total

    def sum_of_squares_np(self):
        total = np.sum(self.random_array ** 2)
        return total

    def sum_of_squares_jit(self):
        return sum_of_squares_out(self.random_array.astype(np.float32))

    def refresh(self):
        self.random_array = np.random.randn(n)


# Usage
n = 400000000  # A large number to make the computation heavy

obj = HeavyComputation(n)

# Measuring performance without Numba, with numpy
obj.refresh()
start_time = time.time()
result = obj.sum_of_squares_np()
end_time = time.time()
print("Without Numba, with numpy:", result, "Time:", end_time - start_time, "seconds")

# Measuring performance with Numba
obj.refresh()
start_time = time.time()
result_jit = obj.sum_of_squares_jit()
end_time = time.time()
print("With Numba:", result_jit, "Time:", end_time - start_time, "seconds")

# Measuring performance with Numba
obj.refresh()
start_time = time.time()
result_jit = obj.sum_of_squares_jit()
end_time = time.time()
print("With Numba:", result_jit, "Time:", end_time - start_time, "seconds")
