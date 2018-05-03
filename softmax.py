# softmax(y_i) = exp(y_i) / sum(exp(y_i))

import math
import numpy as np


def softmax(y):
    y2 = np.array(y)
    return softmax_2d(y2)
#    map = {list: softmax_1d,
#           np.ndarray: softmax_2d}
#    return map[y](y)


def softmax_1d(array_1d):
    sum_y = sum(map(lambda x: math.exp(x), array_1d))
    return list(map(lambda x: math.exp(x)/sum_y, array_1d))


def softmax_2d(array_2d):
    return np.apply_along_axis(softmax_1d, 0, array_2d)


scores = [1.0, 2.0, 3.0]

print(softmax(scores))


scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

#print(softmax(scores))
