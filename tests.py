import numpy as np

# TODO: TEST 1
k = 3
dist = [10, 4, 9, 2, 5, 6, 7, 8, 3, 1]
indices = np.argsort(dist)[:k]  # sort and return the indices from the original array
print(indices)

# TODO: TEST 2
print(np.e)  # euler constant

# TODO: TEST 3
y = np.array([4, 5, 6, 5, 4])
hist = np.bincount(y)  # how many times the index occured as an item in the list
print(hist)
