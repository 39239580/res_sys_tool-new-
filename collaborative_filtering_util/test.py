# import numpy as np
# from collaborative_filtering_util.distance_sim import cal_similar
# from math import sqrt
# from sklearn.metrics import mean_squared_error
# import pandas as pd
# from sklearn.model_selection import train_test_split
# pd.set_option('display.max_columns', None)
#
#
# class CollaFitUserItem(object):   #
#     def __init__(self, data, cf_type="user_cf", order_type="user_item", verbose=True):
#         assert cf_type in {"user_cf", "item_cf"}
#         assert order_type in {"item_user", "user_item"}
#         self.df = data
#         self.order_type = order_type
#         if order_type == "item_user":   # 将item-user转成user-item矩阵
#             self.df = self.df.stack().unstack()
#
#         self.n_user = self.df.user_id.unique().shape[0]   # 不同的用户数
#         self.n_item = self.df.item_id.unique().shape[0]   # 不同的物品数
#         print(u"总的数据长度为:", self.df.shape[0])
#         print("用户总数：%d,电影总数：%d"%(self.n_user, self.n_item))
#
#     def gen_user_item_matrix(self,):
#         user_item_matrix = np.
import pandas as pd
import numpy as np
from  numpy import mat
from collaborative_filtering_util.distance_sim import cal_similar
from scipy.spatial.distance import squareform
from collaborative_filtering_util.user_item_cf_util import commonCF


def load_data():
    # 读取u.data文件
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv("./data/ml-100k/ml-100k/u.data", sep="\t", names=header)
    print(df)
    return df


df = load_data()
f = df.pivot(index="user_id", columns="item_id", values="rating")
print(f)
f.fillna(0.0, inplace=True)
print(f)
print(type(f))

# print(f["user_id"])
r= mat(f)
print(r)
print(r.shape)
# 用户之间的相互矩阵
sim1 = cal_similar("euclidean",x=r)
sim2 = cal_similar("cosin", x=r)
sim3 = cal_similar("pearson", x=r)
sim4 = cal_similar("jaccard", x=r)


print(sim1)
print(sim2)
print(sim3)
print(sim4)

print(sim1.shape)
print(sim2.shape)
print(sim3.shape)
print(sim4.shape)

print(f)

sim1[sim1==1] = 0
print(sim1)
print(f.shape)
print(sim1.shape)

# print("+++++++++++++++++++++++++++++")
# CF = commonCF(algo_object="user_cf", similar_type="euclidean")
# CF._cal_similar(x=r)
# CF._sort()
# CF._get_all_neighbors()
# print(CF.neighbors_sim)
# print(CF.neighbors_index)
# # raise
# CF._get_neighbors(iid=4,k=4)  #获取topk
# print("测试")
# print(CF.iid_index) # 测试的　ｋ个邻居索引
# print(CF.iid_neighbors) # 几个邻居的相似度
#
# # raise
#
# # CF._cal_iid_sum_sim()
# # print(CF.iid_sum_sim)
# CF._transform_action_matrix_iid()
# # raise
# print("输出用户可能对所有item 的感兴趣度")
# print(CF.iid_ctr)
#
# candidate_index, candidate_set = CF._filter_iid_self_item(iid=4)
# print("候选集合的索引")
# print(candidate_index)
# print(candidate_index.shape)
# print("候选集合的值")
# print(candidate_set)
# print(candidate_set.shape)
# candidate_index_sort, candidate_set_sort = CF._iid_rank(candidate_set, candidate_index)
# print(candidate_index_sort)
# print(candidate_set_sort)
# h= {4:{"candidate_set_sort":candidate_set_sort[:2], "candidate_index_sort":candidate_index_sort[:2]}}
# print(h)


# 测试写的类
print("++++++++++++++++++++++++++++++++++++++++++++++----------")
CF = commonCF(algo_object="user_cf", similar_type="euclidean")
CF.fit(trainset=r)
print("测试完毕")

print(CF.res_dict)


