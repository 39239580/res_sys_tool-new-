from sklearn.cluster import KMeans
import numpy as np


class KMEANS(object):
    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='deprecated',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs='deprecated', algorithm='auto'):
        """
        Parameters
        ----------
        n_clusters : TYPE, optional   簇数
            DESCRIPTION. The default is 8.
        init : TYPE, optional
        {"k-means++","random", ndarray}
            DESCRIPTION. The default is 'k-means++'.
        n_init : TYPE, optional
        kmeans 运行的次数
            DESCRIPTION. The default is 10.
        max_iter : TYPE, optional
        单次运行的k均值算法的最大迭代次数
            DESCRIPTION. The default is 300.

        tol : TYPE, optional
        该范数表示两个连续迭代的聚类中心的差异，用于声明收敛
            DESCRIPTION. The default is 1e-4.
        
        precompute_distances : TYPE, optional
        {"auto", True, False}
        预计算距离， "auto": 如果n_samples*n_clusters>1200w，则不预计算距离，使用双精度
            DESCRIPTION. The default is 'deprecated'.
        verbose : TYPE, optional
            DESCRIPTION. The default is 0.
        random_state : TYPE, optional
            DESCRIPTION. The default is None.
        copy_x : TYPE, optional
            DESCRIPTION. The default is True.
        n_jobs : TYPE, optional
            DESCRIPTION. The default is 'deprecated'.
        algorithm : TYPE, optional
        {"auto","full","elkan"}   
            DESCRIPTION. The default is 'auto'.

        Returns
        -------
        None.

        """
        self.kmeans = KMeans(n_clusters=n_clusters, init= init, n_init=n_init,
                             max_iter=max_iter, tol=tol, 
                             precompute_distances=precompute_distances,
                             verbose=verbose, random_state=random_state, 
                             copy_x=copy_x, n_jobs=n_jobs, 
                             algorithm=algorithm)
    
    def fit(self,  x, y=None, sample_weight=None):
        self.kmeans.fit( X=x, y=y, sample_weight=sample_weight)
        
    def fit_transform(self,  x, y=None, sample_weight=None):
        return self.kmeans.fit_transform(X=x, y=y, sample_weight=sample_weight)
        
    def transform(self, x):
        self.kmeans.transform(X=x)
    
    def fit_predict(self, x, y=None, sample_weight=None):
        return self.kmeans.fit_predict(X=x, y=y, sample_weight=sample_weight)
    
    def get_params(self,deep=True):
        self.kmeans.get_params(deep=deep)
    
    def predict(self,  x, sample_weight=None):
        return self.kmeans.predict(X=x, sample_weight=sample_weight)
    
    def set_params(self,params):
        self.kmeans.set_params(**params)
    
    def score(self, x, y=None, sample_weight=None):
        return self.kmeans.score(X=x, y=y, sample_weight=sample_weight)
    
    def get_cluster_centers(self):
        return self.kmeans.cluster_centers_
    
    def get_labels(self):
        return self.kmeans.labels_
    
    def get_inertial(self): # 样本到最近的簇中心的平方距离和
        return self.kmeans.inertia_
    
    def get_n_iter(self):  # 迭代次数
        return self.kmeans.n_iter_
    

if __name__ == "__main__":
    x= np.array([[1, 2], [1, 4], [1, 0],
                 [10, 2], [10, 4], [10,0]])
    
    kmeans=KMEANS(n_clusters=2, random_state=0)
    kmeans.fit(x)
    print(kmeans.get_labels())
    print(kmeans.predict([[0,0],[12, 3]]))
    print(kmeans.get_cluster_centers())

