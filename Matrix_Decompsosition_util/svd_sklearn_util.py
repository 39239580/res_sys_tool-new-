from  sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from sklearn.random_projection import sparse_random_matrix

class TSVD(object):
    def __init__(self, n_components=2, *, algorithm="randomized", n_iter=5,
                 random_state=None, tol=0.):
        """
        :param n_components:  想要输出的数据纬度。 int   默认为2， 必须小于特征的纬度。  对于LSA 推荐取小于100的数
        :param algorithm: b 字符串，使用SVD求解, 默认使用随即
        :param n_iter: 随即SVD的得代次数，，可选， 默认为5。 ARPACK不使用 int
        :param random_state:  int , 默认None   随机种子
        :param tol: ARPACK的公差， 随机不用
        """
        assert algorithm in ["randomized", "arpack"]

        self.model = TruncatedSVD(n_components=n_components,   # 想要输出的数据维度
                                  algorithm=algorithm,
                                  n_iter=n_iter,
                                  random_state=random_state,
                                  tol=tol)


    def fit(self,x,y=None): # 进行分解
        """
        :param X:    {array-like, sparse matrix},shape (n_samples, n_features)
        :return:
        """
        self.model.fit(X=x, y=y)

    def fit_transform(self,x,y=None):  # 返回分解后的结果
        """
        :param X:    {array-like, sparse matrix},shape (n_samples, n_features)
        :return:  X_newarray, shape (n_samples, n_components)    稠密数组
        """
        return self.model.fit_transform(X=x, y=y)

    def get_params(self):  # 获取评估器的参数
        return self.model.get_params(deep=True)   #

    def inverse_transform(self,x): # 逆向分解， 并返回原始数据
        """
        :param x:  X: array-like, shape (n_samples, n_components)
        :return: X_original: array, shape (n_samples, n_features)
        """
        return self.model.inverse_transform(X=x)

    def set_params(self,params):
        self.model.set_params(**params)

    def transform(self,X):
        return self.model.transform(X=X)

    def get_components(self):
        return self.model.components_   # 数组， shape(n_components, n_features)   数据的维度

    def get_explained_variance(self):   # 训练样本的方差通过投影转换每个分量
        return self.model.explained_variance_  # 数组 shape(n_components)

    def get_explained_variance_ratio(self):   # 每个选定组建解释的方差百分比
        return self.model.explained_variance_ratio_    # 数组 shape(n_components)

    def get_singular_values(self):  #  每个选定components 的奇异值
        return self.model.singular_values_ # 数组 shape(n_components)


if __name__ =="__main__":
    X = sparse_random(100, 100, density=0.01, format="csr", random_state=42)
    svd = TSVD(n_components=5, n_iter=7, random_state=42)
    svd.fit(X)
    print(svd.get_explained_variance_ratio())
    print(svd.get_explained_variance_ratio().sum())
    print(svd.get_singular_values())