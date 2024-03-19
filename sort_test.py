from util import quick_sort
import random

if __name__ == "__main__":
    values = []
    for i in range(100):
        values.append(random.randint(0,100))
    print(quick_sort(values))
