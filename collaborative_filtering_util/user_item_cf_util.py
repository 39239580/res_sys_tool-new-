from collaborative_filtering_util.distance_sim import cal_similar
from collaborative_filtering_util.scikit_surprise_util.cf_util import innerid_2_rawid,  rawid_2_innerid
from collaborative_filtering_util.scikit_surprise_util.cf_util import load_data, data_split
import numpy as np
import pickle as pk
import os
import pandas as pd


# 简单版本的协同过滤
class commonCF(object):
    def __init__(self, algo_object, similar_type, topk=200):
        assert algo_object in {"item_cf", "user_cf"}
        assert similar_type in {"euclidean", "cosin", "pearson", "jaccard"}
        self.algo_object = algo_object
        self.similar_type = similar_type
        self.topk = topk

    def _cal_similar(self, x, y=None):  # 输入矩阵, 得到相似度矩阵
        """
        :param x:  x可以为数值型，也可以为矩阵型数据, 列表或者为数组，列表
        :return:
        """
        # assert (type(x) in {np.ndarray, np.matrix}) or isinstance(x,list)
        self.x = x
        if type(self.x) is np.ndarray:  # 若是数组，进行转换
            self.array = np.mat(self.x)
        elif isinstance(self.x, list):
            self.array = np.mat(np.array(self.x))
        else:
            self.array = self.x

        self.sim = cal_similar(cal_type=self.similar_type, x=x, y=y)
        self.n_user = x.shape[0]  # 用户数
        self.n_item = x.shape[1]  # item数量

    def fit(self, trainset, y=None, file_path=None):
        self._cal_similar(x=trainset, y=y)  # 计算所有的的相似度, 得到相似度矩阵
        # raise
        self._sort()
        self._get_all_neighbors()
        self.res_dict=self._cal_neighbors(topk=self.topk)
        self._save_model(file_path=file_path)


    def _save_model(self,file_path):
        if not  file_path:
            file_path ="./model"
        if not os.path.exists(file_path):  # 不存在相应的模型文件，　进行文件的创建
            os.mkdir(file_path)
        save_model(file_path=file_path+"sim.pk",model_param=self.neighbors_sim)
        save_model(file_path=file_path+"sim_index", model_param=self.neighbors_index)
        save_model(file_path=file_path+"res_list", model_param=self.res_dict)

    def _sort(self): # 相似度矩阵进行排序操作
        self.index = np.argsort(-self.sim, axis=1)  # 降序的排列序号,
        self.sortvalue = self.sim.copy()  #　复制一个副本
        for i in range(self.index.shape[0]):
            self.sortvalue[i] = self.sortvalue[i][self.index[i]]

    def _get_all_neighbors(self, k=None): # topk个用户
        if not k:
            self.neighbors_index = self.index[:,1:]
            self.neighbors_sim = self.sortvalue[:,1:]
        else:
            self.neighbors_index = self.index[:, 1:k+1] # topｋ个对应的索引值
            self.neighbors_sim = self.sortvalue[:, 1:k+1] # topk个对应的相似度值

    def _get_neighbors(self, iid, k): # 获取 k个邻居　获取某个iid的k个邻居
        print(k)
        print(self.neighbors_sim.shape[0]-1)
        assert k<= self.neighbors_sim.shape[0]-1
        self.iid_neighbors = self.neighbors_sim[iid,1:k+1]  #相似度值
        self.iid_index = self.neighbors_index[iid, 1:k+1]

    def _cal_iid_sum_sim(self): #　计算单个iid的相似度总和
        self.iid_sum_sim = np.sum(self.iid_neighbors)

    def _cal_sum_sim(self): # 计算所有的iid的相似度总和
        self.sum_sim = np.sum(self.neighbors_sim, axis=1)

    # def transform_action_matrix(self, x, y=None):


    def _transform_action_matrix_iid(self):
        """
        :param x: x 为dataframe格式的数据　　或者使用matrix　　　以及使用
        :return:
        """
        if isinstance(self.x, pd.DataFrame):
            # print("ok///////////////////////////////////")
            x_array = self.dataframe2array(self.x)
        elif type(self.x) is np.matrix:
            x_array = np.array(self.x)
        else:
            x_array = self.x
        # print(x)
        # print(self.iid_index) # shape = [1, topk]
        # print(x_array[self.iid_index])  # shape =[topk, n_item]　数量
        # print("+++++++++++++++++++++------------------")
        # print("尺寸输出")
        # print(self.iid_neighbors.shape, x_array[self.iid_index].shape)


        self.iid_ctr = np.dot(self.iid_neighbors, x_array[self.iid_index])  # 计算出对应的总的分数，　shape=[1, n_item]
        # print(self.iid_ctr)
        # print(self.iid_ctr.shape)
        # print("test1")
        # print(self.iid_neighbors)   # 相似度的值
        # raise

        iid_sim_sum =self._iid_sum_sim_tool(x_array[self.iid_index],self.iid_neighbors)
        # print(iid_sim_sum)
        # print("test2")
        # raise
        # print(sum(self.iid_neighbors))
        # print(self.iid_ctr)
        # print(iid_sim_sum)
        self.iid_ctr = self.iid_ctr/iid_sim_sum

    @staticmethod
    def dataframe2array(x):
        return x.values

    @staticmethod    # 查找非０的索引
    def calnonzero(array):

        dicts= {}
        indexs=np.nonzero(array)
        # print(indexs)
        # raise
        for i, j in zip(indexs[0], indexs[1]):  # 行与列
            if i not in dicts:
                dicts[i] = [j]
            else:
                dicts[i].append(j)
        # print(dicts)
        # print("ok")
        return dicts

    def _nonzero2one(self, array):
        return np.float32(array > 0)


    def _iid_sum_sim_tool(self, array, sim_array):
        print(array)
        iid_sum_similar = np.dot(sim_array, self._nonzero2one(array))
        iid_sum_similar[(iid_sum_similar ==0)] = 0.0001
        return iid_sum_similar


    def _filter_iid_self_item(self, iid):  # 过滤掉自身已经有过行为的数据
        """
        :param iid:  用户id 或item 的id
        :param array:  使用的为mat 格式的数据
        :return:
        """

        arrays=np.array(self.array[iid]).flatten()  # 获取用户对应的交互数据
        cal_rc = self.iid_ctr
        candidate_index = np.where(arrays==0)[0] # 候选索引　
        # p= np.nonzero(arrays)
        # print(arrays)
        # print(candidate_index)
        # print(cal_rc)
        # print(cal_rc[candidate_index])
        candidate_set=cal_rc[candidate_index]
        return candidate_index, candidate_set


    def _iid_rank(self, array, index):
        sortindex=np.argsort(-array)
        # print(array[sortindex])
        # print(index[sortindex])
        sorted_candidate_set = array[sortindex]
        sorted_candidate_index = index[sortindex]
        return sorted_candidate_set, sorted_candidate_index

    def _cal_neighbors(self, topk=None):
        if not topk: # topk 为200
            topk = 200
        res_dict ={}

        for i in range(self.n_user):
            self._get_neighbors(iid=i, k=topk)
            self._transform_action_matrix_iid()
            candidate_index, candidate_set = self._filter_iid_self_item(iid=i)
            candidate_set_sort ,candidate_index_sort= self._iid_rank(array=candidate_set, index=candidate_index)
            if i not in res_dict:
                res_dict[i] = {"candidate_set_sort":candidate_set_sort[:topk], "candidate_index_sort":candidate_index_sort[:topk]}
            else:
                res_dict.update({i:{"candidate_set_sort":candidate_set_sort[:topk], "candidate_index_sort":candidate_index_sort[:topk]}})
        return res_dict

    def predict(self):
        self.res_dict






def save_model(file_path, model_param):
    with open(file_path, "wb") as f:
        pk.dump(model_param, f)

def load_model(file_path):
    with open(file_path, "rb") as f:
        model_param=pk.load(f)
    return model_param
