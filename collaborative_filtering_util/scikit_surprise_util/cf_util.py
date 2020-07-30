from surprise import KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore
from surprise import BaselineOnly, NormalPredictor
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.model_selection import KFold, PredefinedKFold
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise import PredictionImpossible
import os
import pandas as pd
from surprise.dump import dump, load
from surprise.model_selection import train_test_split
import numpy as np

"""
NormalPredictor   基于训练集分布的随机等级预测算法，该算法正常。
BaselineOnly   预测给定用户和项目的基线估计的算法。
KNNBasic  基本的协同过滤算法
KNNWithMeans   一种基本的协同过滤算法，考虑到每个用户的平均评级。
KNNWithZScore  一种基本的协同过滤算法。  Z变换
KNNBaseline 一种考虑基线评分的基本的协同过滤算法
SVD  SVD  算法
SVDpp   SVD++ 算法
NMF    一种基于非负矩阵分解的协同过滤算法。
SlopeOne    一种简单而准确的协同过滤算法。
CoClustering  一种基于聚类的协同过滤. 类似Kmeans
"""


class CFSurprise(object):
    def __init__(self, module_type, baseline_type, cf_type, similar, sim_type, params):
        assert baseline_type in {"ALS", "SGD", "default"}
        assert cf_type in {None, "base_user", "base_item"}
        assert similar in {None, "COSINE", "cosine", "MSD", "msd", "PEARSON", "pearson",
                           "PEARSON_BASELINE", "pearson_baseline", "JACCARD", "jaccard",
                           "EUCLIDEAN", "euclidean"}
        assert sim_type in {None, "default"}
        self.module_type = module_type
        self.baseline_type = baseline_type
        self.cf_type = cf_type
        self.similar = similar
        self.sim_type = sim_type
        self.bu = None
        self.bi = None
        self.sim = None
        if self.baseline_type == "ALS":
            bsl_options = {'method': params["bsl_options"].get("method", 'als'),
                           'n_epochs': params["bsl_options"].get("n_epochs", 10),
                           'reg_u': params["bsl_options"].get("reg_u", 15),
                           'reg_i': params["bsl_options"].get("reg_i", 10)
                           }
        elif self.baseline_type == "SGD":
            bsl_options = {'method':  params["bsl_options"].get("method", 'sgd'),
                           'n_epochs': params["bsl_options"].get("n_epochs", 20),
                           'reg': params["bsl_options"].get("reg", 0.02),
                           'learning_rate': params["bsl_options"].get("learning_rate", 0.005)
                           }
        else:   # 默认值
            bsl_options = {}
        params["sim_options"] = {}

        if self.cf_type == "base_user":
            params["sim_options"]["user_based"] = True
        elif self.cf_type == "base_item":
            params["sim_options"]["item_based"] = False
        else:
            params["sim_options"]["user_based"] = True

        if self.similar == "COSINE" or self.similar == "cosine":
            params["sim_options"]["name"] = "cosine"
        elif self.similar == "MSD" or self.similar == "msd":
            params["sim_options"]["name"] = "msd"
        elif self.similar == "PEARSON" or self.similar == "pearson":
            params["sim_options"]["name"] = "pearson"
        elif self.similar == "PEARSON_BASELINE" or self.similar == "pearson_baseline":
            params["sim_options"]["name"] = "pearson_baseline"
        elif self.similar == "JACCARD" or self.similar == "jaccard":
            params["sim_options"]["name"] = "jaccard"
        elif self.similar == "EUCLIDEAN" or self.similar == "euclidean":
            params["sim_options"]["name"] = "euclidean"
        else:
            params["sim_options"]["name"] = "msd"

        if self.sim_type == "default":
            sim_options = {}
        else:
            sim_options = {"name": params["sim_options"].get("name", "MSD"),
                           "user_based": params["sim_options"].get("user_based", True),
                           "min_support": params["sim_options"].get("min_support", 5),
                           "shrinkage": params["sim_options"].get("shrinkage", 100)
                           }

            """
            'name'：要使用的相似性名称，如similarities模块中所定义 。默认值为'MSD'。
            'user_based'：将计算用户之间还是项目之间的相似性。这对预测算法的性能有巨大影响。默认值为True。
            'min_support'：相似度不为零的最小公共项数（'user_based' 为'True'时）或最小公共用户数（'user_based'为 'False'时）。
            简单地说，如果 |Iuv|<min_support 然后 sim(u,v)=0。项目也是如此。
            'shrinkage'：
            """
        if self.module_type == "KNNmeans":
            # 在KNNBasic算法的基础上，考虑用户均值或项目均值
            self.model = KNNWithMeans(k=params.get("k", 40),
                                      min_k=params.get("min_k", 1),
                                      sim_options=sim_options,
                                      verbose=params.get("verbose", True))
        elif self.module_type == "KNNzscore":
            # 引入Z - Score的思想
            self.model = KNNWithZScore(k=params.get("k", 40),
                                       min_k=params.get("min_k", 1),
                                       sim_options=sim_options,
                                       verbose=params.get("verbose", True))
        elif self.module_type == "KNNbase":
            # 和KNNWithMeans的区别在于，用的不是均值而是bias
            self.model = KNNBaseline(k=params.get("k", 40),
                                     min_k=params.get("min_k", 1),   # 最少的邻居个数
                                     sim_options=sim_options,
                                     bsl_options=bsl_options,
                                     verbose=params.get("verbose", True))
        elif self.module_type == "KNNbasic":
            # 最基础的KNN算法，可分为user - based KNN和item - based KNN
            self.model = KNNBasic(k=params.get("k", 40),
                                  min_k=params.get("min_k", 1),
                                  sim_options=sim_options,
                                  verbose=params.get("verbose", True))
        elif self.module_type == "SVD":
            self.model = SVD(n_factors=params.get("n_factors", 100),
                             n_epochs=params.get("n_epochs", 20),
                             init_mean=params.get("init_mean", 0),
                             init_std_dev=params.get("init_std_dev", 0.1),
                             lr_all=params.get("lr_all", 0.005),
                             reg_all=params.get("reg_all", 0.02),
                             lr_bu=params.get("lr_bu", None),
                             lr_bi=params.get("lr_bi", None),
                             lr_pu=params.get("lr_pu", None),
                             lr_qi=params.get("lr_qi", None),
                             reg_bu=params.get("reg_bu", None),
                             reg_bi=params.get("reg_bi", None),
                             reg_pu=params.get("reg_pu", None),
                             reg_qi=params.get("reg_qi", None),
                             random_state=params.get("random_state", None),
                             verbose=params.get("verbose", False)
                             )
            """
            n_factors –因素数。默认值为100。
            n_epochs – SGD过程的迭代次数。默认值为 20。
            偏见（bool）–是否使用基线（或偏见）。请参阅上面的注释。默认值为True。
            init_mean –因子向量初始化的正态分布平均值。默认值为0。
            init_std_dev –因子向量初始化的正态分布的标准偏差。默认值为0.1。
            lr_all –所有参数的学习率。默认值为0.005。
            reg_all –所有参数的正则项。默认值为 0.02。
            lr_bu –的学习率bu。lr_all如果设置优先 。默认值为None。
            lr_bi –的学习率bi。lr_all如果设置优先 。默认值为None。
            lr_pu –的学习率pu。lr_all如果设置优先 。默认值为None。
            lr_qi –的学习率qi。lr_all如果设置优先 。默认值为None。
            reg_bu –的正则化术语bu。reg_all如果设置优先。默认值为None。
            reg_bi –的正则化术语bi。reg_all如果设置优先。默认值为None。
            reg_pu –的正则化术语pu。reg_all如果设置优先。默认值为None。
            reg_qi –的正则化术语qi。reg_all如果设置优先。默认值为None。
            random_state（int，numpy中的RandomState实例或None）–确定将用于初始化的RNG。
            如果为int，random_state则将用作新RNG的种子。通过多次调用进行相同的初始化非常有用 fit()。
            如果是RandomState实例，则将该实例用作RNG。如果为None，则使用numpy中的当前RNG。默认值为 None。
            详细 –如果True，则打印当前纪元。默认值为False。
            """
        elif self.module_type == "SVDpp":
            self.model = SVDpp(n_factors=params.get("n_factors", 100),
                               n_epochs=params.get("n_epochs", 20),
                               init_mean=params.get("init_mean", 0),
                               init_std_dev=params.get("init_std_dev", 0.1),
                               lr_all=params.get("lr_all", 0.005),
                               reg_all=params.get("reg_all", 0.02),
                               lr_bu=params.get("lr_bu", None),
                               lr_bi=params.get("lr_bi", None),
                               lr_pu=params.get("lr_pu", None),
                               lr_qi=params.get("lr_qi", None),
                               reg_bu=params.get("reg_bu", None),
                               reg_bi=params.get("reg_bi", None),
                               reg_pu=params.get("reg_pu", None),
                               reg_qi=params.get("reg_qi", None),
                               random_state=params.get("random_state", None),
                               verbose=params.get("verbose", False))
            """
            n_factors –因素数。默认值为20。
            n_epochs – SGD过程的迭代次数。默认值为
            20。
            init_mean –因子向量初始化的正态分布平均值。默认值为0。
            init_std_dev –因子向量初始化的正态分布的标准偏差。默认值为0
            .1。
            lr_all –所有参数的学习率。默认值为0
            .007。
            reg_all –所有参数的正则项。默认值为
            0.02。
            lr_bu –的学习率bu。lr_all如果设置优先 。默认值为None。
            lr_bi –的学习率bi。lr_all如果设置优先 。默认值为None。
            lr_pu –的学习率pu。lr_all如果设置优先 。默认值为None。
            lr_qi –的学习率qi。lr_all如果设置优先 。默认值为None。
            lr_yj –的学习率yj。lr_all如果设置优先 。默认值为None。
            reg_bu –的正则化术语bu。reg_all如果设置优先。默认值为None。
            reg_bi –的正则化术语bi。reg_all如果设置优先。默认值为None。
            reg_pu –的正则化术语pu。reg_all如果设置优先。默认值为None。
            reg_qi –的正则化术语qi。reg_all如果设置优先。默认值为None。
            reg_yj –的正则化术语yj。reg_all如果设置优先。默认值为None。
            random_state（int，numpy中的RandomState实例或None）–确定将用于初始化的RNG。如果为int，random_state则将用作新RNG的种子。通过多次调用进行相同的初始化非常有用
            fit()。如果是RandomState实例，则将该实例用作RNG。如果为None，则使用numpy中的当前RNG。默认值为
            None。
            详细 –如果True，则打印当前纪元。默认值为False。
            """
        elif self.module_type == "NMF":
            # 非负矩阵分解，即要求p矩阵和q矩阵都是正的
            self.model = NMF(n_factors=params.get("n_factors", 100),
                             n_epochs=params.get("n_epochs", 20),
                             init_mean=params.get("init_mean", 0),
                             init_std_dev=params.get("init_std_dev", 0.1),
                             lr_all=params.get("lr_all", 0.005),
                             reg_all=params.get("reg_all", 0.02),
                             lr_bu=params.get("lr_bu", None),
                             lr_bi=params.get("lr_bi", None),
                             lr_pu=params.get("lr_pu", None),
                             lr_qi=params.get("lr_qi", None),
                             reg_bu=params.get("reg_bu", None),
                             reg_bi=params.get("reg_bi", None),
                             reg_pu=params.get("reg_pu", None),
                             reg_qi=params.get("reg_qi", None),
                             random_state=params.get("random_state", None),
                             verbose=params.get("verbose", False))

            """
            n_factors –因素数。默认值为15。
            n_epochs – SGD过程的迭代次数。默认值为 50。
            偏见（bool）–是否使用基线（或偏见）。默认值为 False。
            reg_pu –用户的正则化术语λu。默认值为 0.06。
            reg_qi –项目的正规化术语λi。默认值为 0.06。
            reg_bu –的正则化术语bu。仅与偏置版本相关。默认值为0.02。
            reg_bi –的正则化术语bi。仅与偏置版本相关。默认值为0.02。
            lr_bu –的学习率bu。仅与偏置版本相关。默认值为0.005。
            lr_bi –的学习率bi。仅与偏置版本相关。默认值为0.005。
            init_low –因子的随机初始化的下限。必须大于0以确保非负因素。默认值为 0。
            init_high –因子的随机初始化的上限。默认值为1。
            random_state（int，numpy中的RandomState实例或None）–确定将用于初始化的RNG。
            如果为int，random_state则将用作新RNG的种子。通过多次调用进行相同的初始化非常有用 fit()。
            如果是RandomState实例，则将该实例用作RNG。如果为None，则使用numpy中的当前RNG。默认值为 None。
            详细 –如果True，则打印当前纪元。默认值为False。
            """
        elif self.module_type == "SlopeOne":
            self.model = SlopeOne(**params)

        elif self.module_type == "cc":
            # 基于聚类的协同过滤
            self.model = CoClustering(n_cltr_u=params.get("n_cltr_u", 3),
                                      n_cltr_i=params.get("n_cltr_i", 3),
                                      n_epochs=params.get("n_epochs", 20),
                                      random_state=params.get("random_state", None),
                                      verbose=params.get("verbose",False)
                                      )
            """
            n_cltr_u（int）–用户集群的数量。默认值为3。
            n_cltr_i（int）–项目集群的数量。默认值为3。
            n_epochs（int）–优化循环的迭代次数。默认值为 20。
            random_state（int，numpy中的RandomState实例或None）–确定将用于初始化的RNG。
            如果为int，random_state则将用作新RNG的种子。通过多次调用进行相同的初始化非常有用 fit()。
            如果是RandomState实例，则将该实例用作RNG。如果为None，则使用numpy中的当前RNG。默认值为 None。
            详细（bool）–如果为True，则将打印当前纪元。默认值为 False。
            """

        elif self.module_type == "BaselineOnly":
            # 不考虑用户的偏好
            self.model = BaselineOnly(bsl_options=bsl_options, verbose=True)

        elif self.module_type == "Np":
            # 该算法即随机预测算法，假设测试集的评分满足正态分布，然后生成正态分布的随机数进行预测，
            self.model = NormalPredictor()

    def fit(self, trainset):
        self.model.fit(trainset=trainset)
        # 计算相似度
        # 具体的计算对象跟sim_options中参数有关
        # 相似度矩阵，计算相似度矩阵的方式取决于sim_options算法创建时候所传递的参数，返回相似度矩阵
        self.sim = self.model.compute_similarities()
        # 计算用户和项目的基线，这个方法只能适用于Pearson相似度或者BaselineOnly算法，
        # 返回一个包含用户相似度和用户相似度的元组
        self.bu, self.bi = self.model.compute_baselines()
        return self

    def test(self, testset, verson=False):
        predictions = self.model.test(testset=testset, verbose=verson)
        return predictions

    def onekey_transform(self, trainset, testset, verbose=False):
        predictions = self.model.fit(trainset=trainset).test(testset=testset, verbose=verbose)
        return predictions

    def predict(self, uid, iid, r_ui, verbose=False):
        assert isinstance(uid, str) or isinstance(uid, int)
        assert isinstance(iid, str) or isinstance(iid, str)
        if isinstance(uid, int):
            uid = str(uid)
        if isinstance(iid, int):
            iid = str(iid)
        evluations = self.model.predict(uid=uid, iid=iid, r_ui=r_ui, verbose=verbose)
        return evluations

    def get_neighbors(self, iid, k):
        """
        :param iid:  iid 表示对应的uid 或itemid， 具体的跟 sim_options 中参数有关
        :param k:  k代表最近的k个邻居
        :return:
        """
        return self.model.get_neighbors(iid=iid, k=k)

    @staticmethod
    def metric(predictions, verbose=True, metric_type="rmse"):
        assert metric_type in {"mse", "fcp", "mae", "rmse"}
        if metric_type == "mse":
            metric = accuracy.mse(predictions=predictions, verbose=verbose)
        elif metric_type == "fcp":
            metric = accuracy.fcp(predictions=predictions, verbose=verbose)
        elif metric_type == "mae":
            metric = accuracy.mae(predictions=predictions, verbose=verbose)
        else:
            metric = accuracy.rmse(predictions=predictions, verbose=verbose)
        return metric

    def _estimate(self, trainset, uid, iid, top_k=10):   #一般不使用
        """
        :param trainset:
        :param uid:  均使用的inner_id
        :param iid:  使用的内部的id
        :param top_k:
        :return:
        """
        if not (trainset.knows_user(uid=uid) and trainset.knows_item(iid=iid)):
            raise PredictionImpossible('User and/or item is unkown.')
        neighbors = [(vid, self.sim[uid, vid]) for (vid, r) in trainset.ir[iid]]
        # 计算u和v之间的相似性，其中v描述了所有其他用户，他们也对项目I进行了评级。
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)  # 降序
        # 相似度排序操作
        for v, sim_uv in neighbors[:top_k]:
            print('user {0:} with sim {1:1.2f}'.format(v, sim_uv))

    # #  推荐单个的列表信息
    # def recommender_single(self, trainset, ids, top_k=10):
    #     """
    #     :param trainset:
    #     ur，user评分列表（item_inner_id，rating）的字典，键是用户的inner_id
    #     ir，item评分列表（user_inner_id，rating）的字典，键是item的inner_id
    #     :param ids:  使用的内部的id， inner_id
    #     :param top_k:
    #     :return:
    #     """
    #     if not trainset.knows_user(uid=ids):  # 用户中不存在当前的id
    #         raise PredictionImpossible('User is unkown.')
    #     if not trainset.knows_item(iid=ids):   # 物品中不存在当前的id
    #         raise PredictionImpossible("Iterm is unkown")
    #     neighbors = self.model.get_neighbors(iid=ids, k=top_k)  # 获取邻居
    #     innerid_2_rawid(trainset=trainset,inner_id=ids,data_type="")

    # def recommender(self, trainset, ids, top_k=10,verbose=True):
    #     # 遍历每个用户 以及所有物品产生推荐列表
    #     for inner_user_id in range(self._get_n_u_i(object_tytpe="user", verbose=False)):
    #         if verbose:
    #             print("开始处理用户：{}".format(inner_user_id))
    #             top_k_list =[]
    #             count = 0
    #             #

    def model_save(self, out_file, predictions=None, verbose=0):   # 保存模型
        """
        :param out_file: 保存的位置
        :param predictions:  用来保存的预测
        :param verbose: 0， 1
        :return:
        algo 存储的算法
        """
        dump(file_name=out_file, predictions=predictions, algo=self.model,verbose=verbose)

    # 获取用户的数量或item 的数量
    def _get_n_u_i(self, object_tytpe, verbose=True):
        assert object_tytpe in {"user", "item"}
        if object_tytpe == "user":
            n_u_i = self.model.trainset.n_users
            if verbose:
                print("总的用户数是：%d" % n_u_i)
        else:
            n_u_i = self.model.trainset.n_items
            if verbose:
                print("总的用户数是：%d" % n_u_i)
        return n_u_i

    def _get_some_arrtibute(self, attribute_type):
        assert attribute_type in {"n_ratings", "rating_scale", "global_mean",
                                  "all_items", "all_user", "all_ratings"}
        if attribute_type == "n_ratings":
            res = self.model.trainset.n_ratings
        elif attribute_type == "rating_scale":
            res = self.model.trainset.rating_scale
        elif attribute_type == "global_mean":
            res = self.model.trainset.global_mean
        elif attribute_type == "all_items":
            res = self.model.trainset.all_items()   # 返回所有item 的内部id
        elif attribute_type == "all_user":
            res = self.model.trainset.all_users()   # 返回所有user 的内部id
        elif attribute_type == "all_ratings":
            res = self.model.trainset.all_ratings()  # 返回一个(uid, iid, rating)的元组
        else:
            ValueError("未知属性，请输出正确的属性")
        return res


def rawid_2_innerid(trainset, raw_id, data_type="user"):
    # 数据行 raw_id 实际的电影ID,    转成内部电影ID inner_id, iid
    inner_id = None
    if data_type == "item":
        inner_id = trainset.to_inner_iid(riid=raw_id)
    elif data_type == "user":
        inner_id = trainset.to_inner_uid(ruid=raw_id)
    else:
        ValueError("data_type 错误")
    return inner_id


# 将内部id 转成原始id, 内部id 即为矩阵内部的序号
def innerid_2_rawid(trainset, inner_id, data_type):
    raw_id = None
    if data_type == "item":
        raw_id = trainset.to_raw_iid(iiid=inner_id)
    elif data_type == "user":
        raw_id = trainset.to_raw_uid(iuid=inner_id)
    else:
        ValueError("data_type 错误")
    return raw_id


# 模型加载
def cf_model_load(file_path):
    """
    :param file_path: # 保存数据的位置
    :return:
    """
    # 假设里面保存了预测的数值，则返回 （prediction, algo）的元组，与保存的格式相对应
    return load(file_name=file_path)


def load_data(file_dict, dataformat):  # 加载数据
    if dataformat == "builtin":
        data = Dataset.load_builtin(name=file_dict["name"], prompt=True)
    elif dataformat == "file":
        reader = Reader(line_format=file_dict["line_format"], sep=file_dict.get("sep", None),
                        rating_scale=file_dict.get("rating_scale", (1, 5)),
                        skip_lines=file_dict.get("skip_lines", 0))
        data = Dataset.load_from_file(file_path=file_dict["file_path"], reader=reader)
    elif dataformat == "dataframe":
        reader = Reader(rating_scale=file_dict.get("rating_scale", (1, 5)))
        data = Dataset.load_from_df(df=file_dict["df"][file_dict["header"]], reader=reader)
    elif dataformat == "folds":   # 已经进行k折交叉验证
        files_dir = os.path.expanduser(file_dict["file_dir"])
        reader = Reader(name=file_dict["name"])
        train_file = files_dir + file_dict["train_name"]
        test_file = files_dir + file_dict["test_name"]
        folds_files = [(train_file % i, test_file % i) for i in file_dict["file_num"]]
        print(folds_files)
        data = Dataset.load_from_folds(folds_files=folds_files, reader=reader)
    else:
        ValueError("dataframe 超出了可处理的文件的类型范围")
    return data


# 数据转换, 转成内部的数据
def data_split(data, data_type="all", test_size=0.2, random_state=None, shuffle=True):
    if data_type == "all":  # 使用整个数据集
        trainset = data.build_full_trainset()
        testset = None
    else:   # 按比例进行数据分割
        trainset, testset = train_test_split(data=data, test_size=test_size, random_state=random_state,
                                             shuffle=shuffle)
    return trainset, testset


if __name__ == "__main__":
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv("./ml-100k/ml-100k/u.data", sep="\t", names=header)
    print(df)
    file_dicts = {"name": "ml-100k", "line_format": "user item rating timestamp",
                  "sep": "\t", "df": df, "file_path": "./ml-100k/ml-100k/u.data",
                  "train_name": "u%d.base", "test_name": "u%d.test",
                  "file_num": [1, 2, 3, 4, 5], "header": ['user_id', 'item_id', 'rating'],
                  "file_dir": "./ml-100k/ml-100k/"}
    # 测试几种加载数据的方式
    # print(load_data(dataformat="builtin", file_dict=file_dicts))
    # print(load_data(dataformat="file", file_dict=file_dicts))
    print(load_data(dataformat="dataframe", file_dict=file_dicts))
    # print(load_data(dataformat="folds", file_dict=file_dicts))
