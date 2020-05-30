import numpy as np
import sys

boardtensor = (np.zeros((1, 8, 8, 7)))
print(sys.getsizeof(boardtensor))

boardtensor = np.int8(np.zeros((1, 8, 8, 7)))
print(sys.getsizeof(boardtensor))

boardtensor = np.bool_(np.zeros((1, 8, 8, 7)))
print(sys.getsizeof(boardtensor))

