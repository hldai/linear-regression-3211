import numpy as np
from utils import plot_data

dataset = [(6.2, 26.3), (6.5, 26.65), (5.48, 25.03), (6.54, 26.01), (7.18, 27.9),
           (7.93, 30.47)]

N = len(dataset)
X = np.ones((N, 2), np.float32)
for i in range(N):
    X[i, 1] = dataset[i][0]
Y = np.array([y for _, y in dataset], np.float32)

w = np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose())
w = np.matmul(w, Y)
print(w)
plot_data(dataset, w)
