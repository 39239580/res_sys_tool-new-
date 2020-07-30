from sklearn.cluster import MiniBatchKMeans
import numpy as np


class MBKmeans(object):
    def __init__(self, n_clusters=8, init='k-means++', max_iter=100,
                 batch_size=100, verbose=0, compute_labels=True,
                 random_state=None, tol=0.0, max_no_improvement=10,
                 init_size=None, n_init=3, reassignment_ratio=0.01):
        """

        Parameters
        ----------
        n_clusters : TYPE, optional
        簇数
            DESCRIPTION. The default is 8.
        init : TYPE, optional
         {"k-means++","random", ndarray}
            DESCRIPTION. The default is 'k-means++'.
        max_iter : TYPE, optional
        单次运行的k均值算法的最大迭代次数
            DESCRIPTION. The default is 100.
        batch_size : TYPE, optional
        批量大小
            DESCRIPTION. The default is 100.
        verbose : TYPE, optional
            DESCRIPTION. The default is 0.
        compute_labels : TYPE, optional
        一旦小批量优化收敛到合适状态，就可以为整个数据及计算标签分配和惯性
            DESCRIPTION. The default is True.
        random_state : TYPE, optional
            DESCRIPTION. The default is None.
        tol : TYPE, optional
        根据相对中心变化控制提前停止
            DESCRIPTION. The default is 0.0.
        max_no_improvement : TYPE, optional
        根据连续的小批量数量控制提前停止，
            DESCRIPTION. The default is 10.
        init_size : TYPE, optional
            DESCRIPTION. The default is None.
        n_init : TYPE, optional
            DESCRIPTION. The default is 3.
        reassignment_ratio : TYPE, optional
            DESCRIPTION. The default is 0.01.

        Returns
        -------
        None.

        """        
        
        self.mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, init=init, 
                                        max_iter=max_iter, batch_size=batch_size,
                                        verbose=verbose, compute_labels=compute_labels,
                                        random_state=random_state, tol=tol, 
                                        max_no_improvement=max_no_improvement,
                                        init_size=init_size, n_init=n_init, 
                                        reassignment_ratio=reassignment_ratio)
       
        
    def fit(self,  x, y=None, sample_weight=None):
        self.mbkmeans.fit( X=x, y=y, sample_weight=sample_weight)
        
    def fit_transform(self,  x, y=None, sample_weight=None):
        return self.mbkmeans.fit_transform(X=x, y=y, sample_weight=sample_weight)
        
    def transform(self, x):
        self.mbkmeans.transform(X=x)
    
    def fit_predict(self, x, y=None, sample_weight=None):
        return self.mbkmeans.fit_predict(X=x, y=y, sample_weight=sample_weight)
    
    def get_params(self,deep=True):
        self.mbkmeans.get_params(deep=deep)
    
    def predict(self,  x, sample_weight=None):
        return self.mbkmeans.predict(X=x, sample_weight=sample_weight)
    
    def set_params(self,params):
        self.mbkmeans.set_params(**params)
    
    def score(self, x, y=None, sample_weight=None):
        return self.mbkmeans.score(X=x, y=y, sample_weight=sample_weight)
    
    def partial_fit(self, x, y=None, sample_weight=None):
        self.mbkmeans.partial_fit(X=x, y=y, sample_weight=sample_weight)
    
    def get_cluster_centers(self):
        return self.mbkmeans.cluster_centers_
    
    def get_labels(self):
        return self.mbkmeans.labels_
    
    def get_inertial(self): # 样本到最近的簇中心的平方距离和
        return self.mbkmeans.inertia_
    
    # def get_n_iter(self):  # 迭代次数
    #     return self.mbkmeans.n_iter_


if __name__ =="__main__":
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 0], [4, 4],
                  [4, 5], [0, 1], [2, 2],
                  [3, 2], [5, 5], [1, -1]
                  ])
    
    kmeans = MBKmeans(n_clusters=2, random_state=0, batch_size=6)
    
    kmeans.partial_fit(X[0:6,:])
    kmeans.partial_fit(X[6:12,:])
    # print(kmeans)
    print(kmeans.get_cluster_centers())
    print(kmeans.predict([[0,0],[4,4]]))
    kmeans2 = MBKmeans(n_clusters=2, random_state=0, batch_size=6, max_iter=10)
    kmeans2.fit(X)
    print(kmeans2.get_cluster_centers())
    print(kmeans2.predict([[0,0],[4,4]]))
    