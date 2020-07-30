from collaborative_filtering_util.scikit_surprise_util.cf_util import CFSurprise, load_data, data_split
import pandas as pd
from collaborative_filtering_util.scikit_surprise_util.cf_util import rawid_2_innerid
from collaborative_filtering_util.scikit_surprise_util.cf_util import innerid_2_rawid
from surprise import Dataset
import os
import collections

data = Dataset.load_builtin('ml-100k')
data_df = pd.read_csv("./ml-100k/ml-100k/u.data", sep="\t",  header=None,
                      names=['user', 'item', 'rating', 'timestamp'])
item_df = pd.read_csv(os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.item'), sep='|',
                      encoding='ISO-8859-1', header=None, names=['mid', 'mtitle'] + [ x for x in range(22)])
# 每列都转换为字符串类型

data_df = data_df.astype(str)
item_df = item_df.astype(str)
# 电影id到电影标题的映射
item_dict = {item_df.loc[x, 'mid']: item_df.loc[x, 'mtitle'] for x in range(len(item_df))}
print(item_dict)
print(type(data))


# ----------------协同过滤， 基于用户，与基于item--------------------------
# 为用户推荐n部电影，基于用户的协同过滤算法，先获取10个相似度最高的用户，把这些用户评分高的电影加入推荐列表。
def get_similar_users_recommendations(uid, n=10, similar="PEARSON_BASELINE"):
    # 获取训练集，这里取数据集全部数据
    trainset, _ = data_split(data=data, data_type="all")
    # 考虑基线评级的协同过滤算法
    cf_user = CFSurprise(module_type="KNNbase", baseline_type="default", cf_type="base_user",
                         similar=similar, sim_type=None, params={})

    # 拟合训练集
    cf_user.fit(trainset)
    # 将原始id转换为内部id
    inner_id = rawid_2_innerid(trainset=trainset, raw_id=uid, data_type="user")
    # 使用get_neighbors方法得到10个最相似的用户
    neighbors = cf_user.get_neighbors(iid=inner_id, k=10)
    neighbors_uid = (innerid_2_rawid(trainset=trainset, inner_id=x, data_type="user") for x in neighbors)
    recommendations = set()
    # 把评分为5的电影加入推荐列表
    for user in neighbors_uid:
        if len(recommendations) > n:
            break
        item = data_df[data_df['user'] == user]
        item = item[item['rating'] == '5']['item']
        for i in item:
            recommendations.add(item_dict[i])
    print('\nrecommendations for user %s:'% user)
    for i, j in enumerate(list(recommendations)):
        if i >= 10:
            break
        print(j)


# 与某电影相似度最高的n部电影，基于物品的协同过滤算法。
def get_similar_item_recommendations(iid, n=10, similar="PEARSON_BASELINE"):
    # 获取训练集，这里取数据集全部数据
    trainset, _ = data_split(data=data, data_type="all")
    # 考虑基线评级的协同过滤算法
    cf_item = CFSurprise(module_type="KNNbase", baseline_type="default", cf_type="base_item",
                         similar=similar, sim_type=None, params={})
    cf_item.fit(trainset)
    inner_id = rawid_2_innerid(trainset=trainset, raw_id=iid, data_type="item")
    # 使用get_neighbors方法得到n个最相似的电影
    neighbors = cf_item.get_neighbors(iid=inner_id, k=n)
    neighbors_iid = (innerid_2_rawid(trainset=trainset, inner_id=x, data_type="item") for x in neighbors)
    recommendations = [item_dict[x] for x in neighbors_iid]
    print('\nten movies most similar to the %s:' % item_dict[iid])
    for i in recommendations:
        print(i)


# SVD算法，预测所有用户的电影的评分，把每个用户评分最高的n部电影加入字典。
def get_recommendations_dict(n=10):
    trainset, _ = data_split(data=data, data_type="all")
    # 测试集，所有未评分的值
    testset = trainset.build_anti_testset()
    # 使用SVD算法
    svd_model = CFSurprise(module_type="SVD", baseline_type="default", cf_type=None, similar=None, sim_type="default",
                           params={})
    svd_model.fit(trainset)
    # 预测
    predictions = svd_model.test(testset)
    # 均方根误差
    print("RMSE: %s" % svd_model.metric(predictions, verbose=False, metric_type="rmse"))

    # 字典保存每个用户评分最高的十部电影
    user_recommendations = collections.defaultdict(list)
    for uid, iid, r_ui, est, details in predictions:
        user_recommendations[uid].append((iid, est))
    for uid, user_ratings in user_recommendations.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        user_recommendations[uid] = user_ratings[:n]
    return user_recommendations


# 显示为用户推荐的电影名
def rec_for_user(uid, user_recommendations):
    print("recommendations for user %s:" % uid)
    # [ item_dict[x[0]] for x in user_recommendations[uid] ]
    for i in user_recommendations[uid]:
        print(item_dict[i[0]])


def get_svd_recommendation(uid):
    # 获取每个用户评分最高的10部电影
    user_recommendations = get_recommendations_dict(10)
    rec_for_user(uid=uid, user_recommendations=user_recommendations)


# -----------------------皮尔逊----------------------
#基于 用户的相似度进行推荐
get_similar_users_recommendations("1", 10)
# 基于 电影的相似度进行推荐
get_similar_item_recommendations("2", 10)
# -----------------------余弦----------------------
# get_similar_users_recommendations("1", 10, similar="cosine")
# # 基于 电影的相似度进行推荐
# get_similar_item_recommendations("2", 10, similar="cosine")


# -----------------------皮尔逊基础版本----------------------
# get_similar_users_recommendations("1", 10, similar="pearson")
# # 基于 电影的相似度进行推荐
# get_similar_item_recommendations("2", 10, similar="pearson")

# surprise 暂时不支持
# # -----------------------杰卡德----------------------
# get_similar_users_recommendations("1", 10, similar="jaccard")
# # 基于 电影的相似度进行推荐
# get_similar_item_recommendations("2", 10, similar="jaccard")
# surprise 暂时不支持
# # -----------------------欧氏距离----------------------
# get_similar_users_recommendations("1", 10, similar="euclidean")
# # # 基于 电影的相似度进行推荐
# get_similar_item_recommendations("2", 10, similar="euclidean")

# # ------------------------msd----------------------
# get_similar_users_recommendations("1", 10, similar="msd")
# # # 基于 电影的相似度进行推荐
# get_similar_item_recommendations("2", 10, similar="msd")
#
# # 基于SVD进行的分解
# get_svd_recommendation("2")
