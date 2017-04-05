# coding:utf-8
import numpy as np


# 激活函数sigmoid
def sigmoid(input):
    return 1.0 / (1.0 + np.exp(-input))
