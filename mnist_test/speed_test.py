from multiprocessing import Pool
import numpy as np
import time


def vectorized_operation(vectors):
    # Assuming 'vectors' is a 2D numpy array where each row is a vector
    result = np.sin(vectors) + np.log(vectors + 1)
    return result


def process_batch(vectors):
    return np.sin(vectors) + np.log(vectors + 1)


def batch_main(data):
    start = time.time()
    pool = Pool(processes=6)  # Adjust the number of processes based on your CPU
    results = pool.map(process_batch, np.array_split(data, 100))
    results = np.vstack(results)
    elapsed = time.time() - start
    print(elapsed)

def regular_main(data):
    start = time.time()
    results = []
    for i in range(len(data)):
        results.append(vectorized_operation((data[i])))
    elapsed = time.time() - start
    print(elapsed)

if __name__ == '__main__':
    data = np.random.rand(10000000, 10)  # 1,000,000 vectors of length 10
    batch_main(data)
    regular_main(data)