from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from multiprocessing import cpu_count


class KNN(object):
    def __init__(self, task_type="cla",module_type="performance", **params):

        assert task_type in ["cla", "reg"]  # 两种类型
        assert module_type in ["balance", "debug", "performance",None]  # 三种 性能模型
        self.module_type = module_type


        if self.module_type == "debug":
            params["n_jobs"] = 1
        elif self.module_type == "performance":  # 性能模型
            params["n_jobs"] = cpu_count()  # cpu核心数
        elif self.module_type == "balance":# 均衡模型
            params["n_jobs"] = cpu_count() // 2
        else:
            params["n_jobs"] = None

        self.task_type = task_type
        # weights  取值{"uniform", "distance",None}   # 默认使用的uniform
        # "algorithm" 取值 {"auto", "ball_tree", "kd_tree", "brute",None}s
        # 权重 uniform 均匀权重， distance  按照其距离的倒数
        # "ball_tree" 使用BallTree算法，
        # kd_tree   使用的KDTree 算法
        # brute   使用暴力搜索 算法。

        # p的取值，  int 类型数据， 默认为2  马尔科夫功率参数
        # p=1 时，等校于p=2使用manhattan_distance(l1)和euclidean_distance(l2).
        # 对于任意的p， 使用minkowskidistance(l_p)





        if self.task_type =="cla":
            self.model = KNeighborsClassifier(n_neighbors=params.get("n_neighbors",5),
                                              weights=params.get("weights",'uniform'),
                                              algorithm=params.get("algorithm",'auto'),
                                              leaf_size=params.get("leaf_size",30),  # 叶子大小
                                              p=params.get("p",2),
                                              metric=params.get("metric",'minkowski'),
                                              metric_params=params.get("metric_params",None),
                                              n_jobs=params.get("n_jobs",None)  # 并行数
                                              )

        else:
            self.model = KNeighborsRegressor(n_neighbors=params.get("n_neighbors",5),
                                              weights=params.get("weights",'uniform'),
                                              algorithm=params.get("algorithm",'auto'),
                                              leaf_size=params.get("leaf_size",30),
                                              p=params.get("p",2),
                                              metric=params.get("metric",'minkowski'),
                                              metric_params=params.get("metric_params",None),
                                              n_jobs=params.get("n_jobs",None)
                                              )

    def fit(self,x,y=None):
        self.model.fit(X=x,y=y)

    def get_params(self):
        return self.model.get_params(deep=True)

    def set_params(self,params):
        self.model.set_params(**params)

    def predict(self,x):
        return self.model.predict(X=x)

    def predict_proba(self,x):
        if self.task_type =="cla":
            return self.model.predict_proba(X=x)
        else:
            ValueError("回归任务无法使用")

    def get_score(self,x,y,sample_weight):
        return self.model.score(X=x,y=y, sample_weight=sample_weight)

    def search_kneighbors(self,x=None, n_neighbors=None, return_distance=True):  # 查找K近邻居
        return self.model.kneighbors(X=x,n_neighbors=n_neighbors, return_distance=return_distance)

    def get_kneighbors_graph(self,x=None,n_neighbors=None ,mode='connectivity'):  # 获取最近邻图
        """
        :param x:
        :param n_neighbors:
        :param mode: "distance","connectivity"
        :return:
        """
        return self.model.kneighbors_graph(X=x,n_neighbors=n_neighbors,mode=mode)


if __name__ =="__main__":
    X = [[0],[1],[2],[3]]
    y = [0, 0, 1 ,1]
    neigh = KNN(task_type="cla",module_type=None, n_neighbors=3)
    neigh.fit(x=X,y=y)
    print(neigh.predict([[1.1]]))

    print(neigh.predict_proba([[0.9]]))


    neigh1 = KNN(task_type="reg",module_type=None, n_neighbors=2)
    neigh1.fit(x=X,y=y)
    print(neigh1.predict([[1.5]]))
