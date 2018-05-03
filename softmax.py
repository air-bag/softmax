import math
import numpy as np

# softmax(y_i) = exp(y_i) / sum(exp(y_i))
# each column represents a sample
def softmax(y):
    y2 = np.array(y)
    if y2.ndim == 1:
        return softmax_1d(y2)
    return softmax_2d(y2)


def softmax_1d(array_1d):
    sum_y = sum(map(lambda x: math.exp(x), array_1d))
    return list(map(lambda x: math.exp(x) / sum_y, array_1d))


def softmax_2d(array_2d):
    return np.apply_along_axis(softmax_1d, 0, array_2d)


scores = [1.0, 2.0, 3.0]
print(softmax(scores))

scores = [11, 12, 13]
print(softmax(scores))

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

print(softmax(scores))




# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()