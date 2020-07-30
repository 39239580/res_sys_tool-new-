from numpy.linalg.linalg import svd as svd
import numpy as np
from numpy import mat

class SVD_python(object):
    def __init__(self):


    def fit(self, data):   # 进行svd  分解操作
        if type(data) is np.ndarray:  # 如果是数组
            data = mat(data)
        else:
            ValueError("data type must be np.ndarray or np.matrix")
        U, D, V= svd(data)
        return U, D, V

    def top_k(self, k):





