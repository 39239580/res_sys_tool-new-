from sklearn.metrics.pairwise import pairwise_distances  # 余弦距离
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度
from sklearn.metrics.pairwise import euclidean_distances   # 欧式距离
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# 计算相似度的。
def cal_similarity(data_obj, train_data_matrix, default_fun="cosine_similarity"):

    assert data_obj == "user" or data_obj == "item"
    ValueError("data_obj must be user or item")

    if data_obj == "user":
        train_data_matrix = train_data_matrix
    else:
        train_data_matrix = train_data_matrix.T
    if default_fun == "cosine_distance":
        similarity_matrix = pairwise_distances(train_data_matrix)
    elif default_fun == "euclidean_distances":
        similarity_matrix = euclidean_distances(train_data_matrix)
    else:
        similarity_matrix = cosine_similarity(train_data_matrix)
    return similarity_matrix


def show_similarity(matrix, types):
    print(types + u"相似度矩阵大小：", matrix.shape)
    print(types + u"相似度矩阵：", matrix)


def create_user_item_matrix(data, shape):
    init_matrix = np.zeros(shape=shape)
    print(init_matrix.shape)
    print(data.shape)
    for line in data.itertuples():
        init_matrix[line[1]-1, line[2]-1] = line[3]
    return init_matrix


def cf_predict(label, similarity, types):
    assert types == "user" or types == "item"
    ValueError("Type must be user or item")
    if types == "user":
        mean_user_label = label.mean(axis=1)
        label_diff = (label - mean_user_label[:, np.newaxis])
        pred = mean_user_label[:, np.newaxis] + np.dot(similarity, label_diff)/np.array(
            [np.abs(similarity).sum(axis=1)]).T
    else:
        pred = label.dot(similarity) / (np.array([np.abs(similarity).sum(axis=1)])+0.00000001)
    print(u"预测值尺寸：", pred.shape)
    return pred


def rmse(pred, truth):
    prediction = pred[truth.nonzero()].flatten()
    ground_truth = truth[truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


def get_rmse(pred, data_matrix):
    rmses = rmse(pred=pred, truth=data_matrix)
    return rmses


# -----------测试电影评分数据集-----------#
def fn():
    BASE_PATH = "J:/data_set_0926/program/code_tools/movies_ratings_data"
    file_name = "ratings"
    path = os.path.join(BASE_PATH, "%s.csv" % file_name)

    data_df = pd.read_csv(path)
    print(data_df.head(5))
    n_users = max(data_df["userId"].unique())
    n_items = max(data_df["movieId"].unique())
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=10)
    user_item_matrix_train = create_user_item_matrix(train_data, shape=[n_users, n_items])
    user_item_matrix_test = create_user_item_matrix(test_data, shape=[n_users, n_items])
    print(user_item_matrix_train.shape)
    print(user_item_matrix_test.shape)

    user_similarity = cal_similarity(data_obj="user", train_data_matrix=user_item_matrix_train)
    item_similarity = cal_similarity(data_obj="item", train_data_matrix=user_item_matrix_train)

    show_similarity(user_similarity, "user")
    show_similarity(item_similarity, "item")

    user_prediction = cf_predict(user_item_matrix_train, user_similarity, types='user')
    item_prediction = cf_predict(user_item_matrix_train, item_similarity, types='item')
    print(user_prediction)
    print(item_prediction)

    print('User-based CF RMSE: ' + str(rmse(user_prediction, user_item_matrix_train)))
    item_prediction = np.nan_to_num(item_prediction)
    print('Item-based CF RMSE: ' + str(rmse(item_prediction, user_item_matrix_test)))


if __name__ == "__main__":
    fn()