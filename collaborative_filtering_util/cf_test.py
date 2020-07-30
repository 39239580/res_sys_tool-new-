import numpy as np
from collaborative_filtering_util.distance_sim import cal_similar
from math import sqrt
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)


class CollabFilter(object):
    def __init__(self, data, order_type="user_item"):
        assert order_type in {"user_item", "item_user"}
        self.df = data
        self.order_type = order_type
        if order_type == "item_user":
            self.df = self.df.stack().unstack()   # 进行行列转换操作
        self.data_row = self.df.shape[0]
        self.data_col = self.df.shape[1]
        self.n_users = self.df.user_id.unique().shape[0]   # 用户数
        self.n_items = self.df.item_id.unique().shape[0]   # 物品数
        print("用户总数：%d, 电影总数：%d"%(self.n_users, self.n_items))

    def generate_user_item_matrix(self, data_set):
        user_item_matrix = np.zeros((self.n_users, self.n_items))
        # print(data_set)
        # raise
        for line in data_set.itertuples():  # 遍历每一行，
            # print(line[1],[2],[3])
            user_item_matrix[line[1]-1, line[2]-1] = line[3]   # 将 df 中的数据
        return user_item_matrix

    # 计算用户的相似度矩阵
    def cal_sim(self, object_type, data_set, cal_type="cosin"):
        assert object_type in {"user", "item"}
        assert cal_type in {"cosin", "euclidean", "pearson", "jaccard"}
        if object_type == "user":
            print("计算用户的相似度矩阵......")
            similar_matrix = cal_similar(cal_type=cal_type, x=data_set)
            print(u"user相似度矩阵：", similar_matrix)
        else:
            print("计算item的相似度矩阵......")
            similar_matrix = cal_similar(cal_type=cal_type, x=data_set)
            print(u"item相似度矩阵：", similar_matrix)
        return similar_matrix

    def predict(self, action_data, similar_matrix, cf_type):
        # 基于用户相似度矩阵
        if cf_type == "user_cf":
            mean_user_rating = action_data.mean(axis=1)   # 计算每个电影的平均评分
            print(action_data)
            print(mean_user_rating)
            # raise
            ratings_diff = action_data - mean_user_rating[:, np.newaxis]  # 新增一个纬度
            # print(ratings_diff)
            # raise

            pred = mean_user_rating[:, np.newaxis] + np.dot(similar_matrix, ratings_diff) / np.array(
                [np.abs(similar_matrix).sum(axis=1)]).T
            print(similar_matrix.shape)
            print(ratings_diff.shape)

        elif cf_type == "item_cf":
            rr = np.dot(similar_matrix, action_data)
            print(rr.shape)
            pred = np.dot(similar_matrix, action_data) / np.array([np.abs(similar_matrix).sum(axis=1)]).T
        else:
            ValueError("cf_type must be user_cf or item_cf")
        print(u"预测值：", pred.shape)
        return pred

    @staticmethod
    def evaluation(predict, truth):
        print(truth.nonzero())
        # array 中输出非零的行列索引
        # (array([  0,   0,   0, ..., 942, 942, 942]), array([   4,    8,   14, ..., 1010, 1046, 1066]))
        # (array([  0,   0,   0, ..., 942, 942, 942]), array([   4,    8,   14, ..., 1010, 1046, 1066]))
        prdict_flatten = predict[truth.nonzero()].flatten()   # 展平
        true_flatten = truth[truth.nonzero()].flatten()
        res = sqrt(mean_squared_error(prdict_flatten, true_flatten))
        return res


def load_data():
    # 读取u.data文件
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv("./data/ml-100k/ml-100k/u.data",sep="\t", names= header)
    print(df)
    return df


if __name__ == "__main__":
    # 先加载数据
    df = load_data()
    # 数据中的user_id    item_id, 对应的为整形数值
    # 是实例化算法对象
    CF = CollabFilter(data=df)
    # 数据集划分
    train_data, test_data = train_test_split(df, test_size=0.3, random_state=2020)
    # 产生训练集和测试集的user_item_matrix
    user_item_matrix_train = CF.generate_user_item_matrix(train_data)
    user_item_matrix_test = CF.generate_user_item_matrix(test_data)
    print(u"训练集大小：", user_item_matrix_train.shape)
    print(u"测试集大小：", user_item_matrix_test.shape)
    user_similar = CF.cal_sim(object_type="user", data_set=user_item_matrix_train)
    item_similar = CF.cal_sim(object_type="item", data_set=user_item_matrix_train)
    print(u"用户相似度矩阵尺寸：", user_similar.shape)
    print(u"用户相似度矩阵：", user_similar)
    print(u"item相似度矩阵尺寸：", item_similar.shape)
    print(u"item相似度矩阵：", item_similar)
    user_prediction = CF.predict(user_item_matrix_train, user_similar, cf_type="user_cf")   # 基于用户的推荐
    item_prediction = CF.predict(user_item_matrix_train, item_similar, cf_type="item_cf")   # 基于item 的推荐
    print(user_prediction)
    print(item_prediction)
    rmse = CF.evaluation(user_prediction, user_item_matrix_test)
    rmse1 = CF.evaluation(item_prediction, user_item_matrix_test)
    print(rmse, rmse1)
