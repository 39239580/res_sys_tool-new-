from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import BaselineOnly
from surprise import Reader
from surprise.model_selection import cross_validate
import os
import pandas as pd
from surprise import NormalPredictor
from surprise.model_selection import KFold, PredefinedKFold
from surprise.model_selection import GridSearchCV # 网格搜索


# ---------------------svd------------------------------
# # 加载数据集
# data = Dataset.load_builtin("ml-100k")
#
# trainset, testset = train_test_split(data, test_size=0.25)
# algo = SVD()
# # algo.fit(trainset)
# # predictions = algo.test(testset)
# # accuracy.rmse(predictions)
#
# # 或使用 fit().test（）进行训练与测试
# predictions = algo.fit(trainset).test(testset)
# accuracy.rmse(predictions)


# -----------------------------------KNNBasic----------------------------
# # 在整个数据集上进行训练
#
# data = Dataset.load_builtin("ml-100k")
# trainset = data.build_full_trainset()
#
# algo = KNNBasic()
# algo.fit(trainset)
#
# # 预测用户196， item302 的评分
# uid = str(196)
# iid = str(302)
# pred = algo.predict(uid=uid, iid=iid, r_ui=4, verbose=True)


# ----------------自定义数据集----------------------
# ---------------------从文件中加载数据------------------------
# 使用的ALS算法
# file_path = os.path.expanduser("../ml-100k/ml-100k/u.data")
# reader = Reader(line_format="user item rating timestamp", sep="\t")
# data = Dataset.load_from_file(file_path, reader=reader)
# print(data)
# cross_validate(BaselineOnly(), data, verbose=True)

# ---------------------从df加载数据-------------------------------
# NormalPredictor算法
# ratings_dict = {"itemID": [1, 1, 1, 2, 2],
#                 "userID": [9, 32, 2, 45, "user_foo"],
#                 "rating": [3, 2, 4, 3, 1]}
# df = pd.DataFrame(ratings_dict)
# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader=reader)
# cross_validate(NormalPredictor(), data=data, cv=2, verbose=True)

# --------------------------交叉验证---------------------------------
# data = Dataset.load_builtin("ml-100k")
# kf = KFold(n_splits=3)
# algo = SVD()
# for trainset, testset in kf.split(data):
#     algo.fit(trainset=trainset)
#     predictions = algo.test(testset=testset, verbose=True)
#     accuracy.rmse(predictions=predictions, verbose=True)


# --------------------------交叉验证---------------------------------
# 已经进行数据集划分的操作
files_dir = os.path.expanduser("../ml-100k/ml-100k/")

# This time, we'll use the built-in reader.
# reader = Reader('ml-100k')
# train_file = files_dir + 'u%d.base'
# test_file = files_dir + 'u%d.test'
# folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]
# data = Dataset.load_from_folds(folds_files, reader=reader)
# pkf = PredefinedKFold()
# algo = SVD()
# for trainset, testset in pkf.split(data):
#     algo.fit(trainset=trainset)
#     predictions = algo.test(testset=testset, verbose=True)
#     accuracy.rmse(predictions=predictions, verbose=True)

# ---------------------------网格搜索---------------------------
# Use movielens-100K
data = Dataset.load_builtin('ml-100k')
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)
# best RMSE score
print(gs.best_score['rmse'])
# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# # 获取最佳的模型参数，进行训练
algo = gs.best_estimator['rmse']
algo.fit(data.build_full_trainset())

# param_grid = {'bsl_options': {'method': ['als', 'sgd'],
#                               'reg': [1, 2]},
#               'k': [2, 3],
#               'sim_options': {'name': ['msd', 'cosine'],
#                               'min_support': [1, 5],
#                               'user_based': [False]}
#               }
