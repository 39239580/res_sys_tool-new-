from collaborative_filtering_util.distance_sim import cal_similar

class commonCF(object):
    def __init__(self, algo_object, similar_tye):
        assert algo_object in {"item_cf", "user_cf"}
        assert algo_object in {"euclidean", "cosin", "pearson", "jaccard"}
        self.algo_object = algo_object
        self.similar_type = similar_tye

    def _cal_similar(self, x, y=None):  # 输入矩阵
        """
        :param x:  x可以为数值型，也可以为矩阵型数据
        :return:
        """
        return cal_similar(cal_type=self.similar_type, x=x, y=y)

    def fit(self, trainset, y=None):
        self.sim = self._cal_similar(x=trainset, y=y)  # 计算所有的的相似度, 得到相似度矩阵

    def _sort(self): # 相似度矩阵进行排序操作
        sim_list = self.sim.tolist()





    def get_neighbors(self, iid, k): # 获取 k个邻居
        self.




    def model_save(self):　# 模型进行保存


    def predict(self):


    def















