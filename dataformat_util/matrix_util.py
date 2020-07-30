from itertools import product, chain
from copy import deepcopy


class Matrix(object):
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]))

    def row(self, row_no):
        """
        :param row_no: 矩阵行号
        :return:  返回矩阵
        """
        return Matrix([self.data[row_no]])

    def col(self,col_no):
        """
        :param col_no: 矩阵列号
        :return:  返回矩阵
        """
        m = self.shape[0]
        return Matrix([[self.data[i][col_no]]for i in range(m)])

    @property
    def is_square(self):
        """
        检查矩阵 是否是个方阵
        :return:
        """
        return self.shape[0] == self.shape[1]

    @property
    def transpose(self):
        """
        原始矩阵的转置换
        :return:
        """
        data = list(map(list, zip(*self.data)))
        return Matrix(data)

    def _eye(self, n):
        """
        获取一个单位矩阵  尺寸为  n*n
        :param n:
        :return:
        """
        return [[0 if i != j else 1 for j in range(n)]for i in range(n)]

    @property
    def eye(self):
        """
        获取与自身矩阵大小相同的单位矩阵
        :return:
        """
        assert self.is_square,"The matrix has to be Square"
        data = self._eye(self.shape[0])
        return Matrix(data)

    def _gaussian_elimination(self,aug_matrix):
        n = len(aug_matrix)
        m = len(aug_matrix[0])
        for col_idx in range(n):
            if aug_matrix[col_idx][col_idx]==0:
                row_idx = col_idx
                while row_idx< n and aug_matrix[row_idx][col_idx] ==0:
                    row_idx +=1
                for i in range(col_idx,m):
                    aug_matrix[col_idx][i] += aug_matrix[row_idx][i]

            for i in range(col_idx +1 , n):
                if aug_matrix[i][col_idx]==0:
                    continue
                k = aug_matrix[i][col_idx] /aug_matrix[col_idx][col_idx]
                for j in range(col_idx, m):
                    aug_matrix[i][j] -=k*aug_matrix[col_idx][j]

        for col_idx in range(n -1, -1, -1):
            for i in range(col_idx):
                if aug_matrix[i][col_idx]==0:
                    continue
                k = aug_matrix[i][col_idx] /aug_matrix[col_idx][col_idx]
                for j in chain(range(i, col_idx+1),range(n,m)):
                    aug_matrix[i][j] -= k*aug_matrix[col_idx][j]

        for i in range(n):
            k =1/aug_matrix[i][i]
            aug_matrix[i][i] *=k
            for j in range(n,m):
                aug_matrix[i][j]*=k

        return aug_matrix

    def _inverse(self,data):
        n = len(data)
        unit_matrix = self._eye(n)
        aug_matrix = [a+b for a, b in zip(self.data,unit_matrix)]
        ret =self._gaussian_elimination(aug_matrix)
        return list(map(lambda x: x[n:],ret))

    @property
    def inverse(self):
        # 找到自身矩阵的逆矩阵
        """
        :return:
        """
        assert  self.is_square,"The matrix has to be squre!"
        data =self._inverse(self.data)
        return Matrix(data)

    def _row_mul(self, row_A, row_B):
        """
        将两个数组中下标相同的元素相乘并求和
        :param row_A:  list   float  或int
        :param row_B:  list    float 或int
        :return:
        """
        return sum(x[0]*x[1] for x in zip(row_A, row_B))

    def _mat_mul(self, row_A, B):
        # mat_mul函数的辅助函数。
        """
        :param row_A: list   1d列表，float int
        :param B:   matrix
        :return:
        """
        row_pairs = product([row_A], B.transpose.data)
        return [self._row_mul(*row_pairs) for row_pairs in row_pairs]

    def mat_mul(self, B):
        """
        矩阵相乘
        :param B:
        :return:
        """
        error_msg = "A列数与B行数不匹配"
        assert self.shape[1]== B.shape[0], error_msg
        return Matrix([self._mat_mul(row_A, B)for row_A in self.data])

    def _mean(self,data):
        """
         计算所有样本的平均数
        :param data:
        :return:
        """
        m = len(data)
        n = len(data[0])
        ret = [0 for _ in range(n)]
        for row in data:
            for j in range(n):
                ret[j] += row[j]/m
        return ret

    def mean(self):
        # "计算所有样本的均值"
        return Matrix(self._mean(self.data))

    def scala_mul(self, scala):
        """
        scala乘法
        :param scala:
        :return:
        """
        m, n = self.shape
        data = deepcopy(self.data)
        for i in range(m):
            for j in range(n):
                data[i][j] *=scala
        return Matrix(data)
