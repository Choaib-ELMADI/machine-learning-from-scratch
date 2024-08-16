import numpy as np

k = 3
distances = [10, 4, 9, 2, 5, 6, 7, 8, 3, 1]
indices = np.argsort(distances)[:k]  # sort and return the index from the original array
print(indices)
