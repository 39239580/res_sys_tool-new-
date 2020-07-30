from sklearn.cluster import AffinityPropagation
import numpy as np


class AP(object):
    def __init__(self,damping=.5, max_iter=200, convergence_iter=15,
                 copy=True, preference=None, affinity='euclidean',
                 verbose=False, random_state='warn'):
        """

        Parameters
        ----------
        damping : TYPE, optional
        阻尼系数   0.5~1 之间
            DESCRIPTION. The default is .5.
        max_iter : TYPE, optional
        最大迭代次数
            DESCRIPTION. The default is 200.
        convergence_iter : TYPE, optional
        停止收敛的估计簇数没有变化的迭代数
            DESCRIPTION. The default is 15.
        copy : TYPE, optional
        复制输入数据 True
            DESCRIPTION. The default is True.
        preference : TYPE, optional
        
            DESCRIPTION. The default is None.
        affinity : TYPE, optional
        {"euclidean","precomputed"}
        欧氏距离 与与计算
            DESCRIPTION. The default is 'euclidean'.
        verbose : TYPE, optional
            DESCRIPTION. The default is False.
        random_state : TYPE, optional
            DESCRIPTION. The default is 'warn'.

        Returns
        -------
        None.

        """
        self.ap_cluster = AffinityPropagation(damping=damping, max_iter=max_iter,
                                              convergence_iter=convergence_iter,
                                              copy=copy, preference=preference,
                                              affinity=affinity,
                                              verbose=verbose, 
                                              random_state=random_state)
        
    
    def fit(self,x,y=None):
        self.ap_cluster.fit(X=x, y=y)
    
    def fit_predict(self, x, y=None):
        return self.ap_cluster.fit_predict(X=x, y=y)
    
    def get_params(self, deep=True):
        return self.ap_cluster.get_params(deep=deep)
    
    def set_params(self, params):
        self.ap_cluster.set_params(**params)
        
    def predict(self, x):
        return self.ap_cluster.predict(X=x)
    
    def get_cluster_centers_indices(self):
        return self.ap_cluster.cluster_centers_indices_
    
    def get_cluster_centers(self):
        return self.ap_cluster.cluster_centers_
    
    def get_labels(self):
        return self.ap_cluster.labels_
    
    def get_affinity_matrix(self):
        return self.ap_cluster.affinity_matrix_
    
    def get_n_iter(self):
        return self.ap_cluster.n_iter_
    
    
    
    

if __name__ =="__main__":
    
    X= np.array([[1,2], [1,4], [1,0],
                 [4,2],[4,4], [4,0]])
    
    AP_cluster=AP(random_state=5)
    AP_cluster.fit(X)
    print(AP_cluster.get_labels())
    print(AP_cluster.predict(X))
    print(AP_cluster.get_cluster_centers())
    
    
    
    