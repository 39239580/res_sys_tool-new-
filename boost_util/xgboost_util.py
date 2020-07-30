import matplotlib.pyplot as plt
# from sklearn import datasets  #  加载数据集
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, explained_variance_score
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from xgboost import XGBClassifier  # 使用的scikit——learn  相关的API接口
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree, to_graphviz
import xgboost as xgb
import numpy as np


from scipy.sparse import csr_matrix
from  multiprocessing  import cpu_count
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix,f1_score,log_loss
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn_util.plot_custom_util import plot_importance_v1

# import xgboost as xgb
# from sklearn.preprocessing import OneHotEncoder  # 独热编码的包

#  需要多数据进行拆分
# 数据切割
# x_train,x_test,y_train,y_test =train_test_split(
#         data, target,test_size=0.3,random_sate =33  # 对数据进行分割
#         )
"""
xgboost 算法接口  https://xgboost.readthedocs.io/en/latest/parameter.html
"""


class Xgboost(object):
    def __init__(self, task="cla", module_type="performance", compute_task="cpu",**params):
        """
        :param task:
        :param module_type:
        :param compute_task:
        :param params:
        """
        assert task in ["cla", "reg"]
        assert module_type in ["debug", "performance", "balance"]
        assert compute_task in ["cpu","gpu"]


        self.task = task
        self.module_type = module_type   # 模块
        if self.module_type == "debug":
            params["n_jos"] = 1
        elif self.module_type == "performance":
            params["n_jos"] =cpu_count()   # cpu核心数
        else:   # 性能模式
            params["n_jos"] = cpu_count()//2
        self.compute_task = compute_task

        if self.compute_task== "gpu":  # 使用gpu
            params["tree_method"] = "gpu_hist"
        else:  # 默认cpu
            params["tree_method"] = "hist"  # 使用的cpu


        if self.task == "reg":   # 做回归任务
            self.model = XGBRegressor(learning_rate=params.get("learning_rate", 0.3),
                                      n_estimators=params.get("n_estimators", 100),  # 树的个数100,即代数
                                      max_depth=params.get("max_depth", 6),  # 树的深度
                                      min_child_weight=params.get("min_child_weight", 1),  # 叶子节点最小权重
                                      n_jobs=params.get("n_jos",None),  # 线程数
                                      gamma=params.get("gamma", 0),  # 惩罚项中叶子节点个数前的参数
                                      reg_lambda=params.get("lambda", 1),  # lambda
                                      reg_alpha=params.get("alpha", 0),
                                      tree_method=params.get("tree_method", "auto"),
                                      subsample=params.get("subsample", 1),  # 随机选择100%样本建立决策树
                                      colsample_bytree=1,  # 随机选择80%特征建立决策树
                                      objective=params.get("objective", "reg:squarederror"),  # 指定损失函数
                                      # num_class=params.get("num_class", 2),  # 不指定即为2分类
                                      booster=params.get("booster", "gbtree"),  # 使用的提升器
                                      scale_pos_weight=1,  # 解决样本不平衡问题
                                      random_state=27,  # 随机数
                                      )

        else:   # 做的分类任务
            self.model = XGBClassifier(learning_rate=params.get("learning_rate", 0.3),
                                       n_estimators=params.get("n_estimators", 100),  # 树的个数100,即代数
                                       max_depth=params.get("max_depth", 6),  # 树的深度
                                       min_child_weight=params.get("min_child_weight", 1),  # 叶子节点最小权重
                                       n_jobs=params.get("n_jos",None),  # 线程数
                                       gamma=params.get("gamma", 0),  # 惩罚项中叶子节点个数前的参数
                                       reg_lambda=params.get("lambda", 1),  # lambda
                                       reg_alpha=params.get("alpha", 0),
                                       tree_method=params.get("tree_method", "auto"),   # 树方法， 默认为auto
                                       subsample=params.get("subsample", 1),  # 随机选择100%样本建立决策树
                                       colsample_bytree=1,  # 随机选择80%特征建立决策树
                                       objective=params.get("objective", "multi:softmax"),
                                       # 指定损失函数   # 'binary:logistic   二分类交叉上

                                       # num_class=params.get("num_class", 2),  # 不指定即为2分类
                                       booster=params.get("booster", "gbtree"),  # 使用的提升器
                                       scale_pos_weight=1,  # 解决样本不平衡问题
                                       random_state=27,  # 随机数
                                       )
        """
        目标函数类型
        具体查看  https://xgboost.readthedocs.io/en/latest/parameter.html
        obejctive:  默认  reg:squarederror:
        reg:squarederror:  #回归平方误差
        reg:squaredlogerror  # 上述误差上取对数
        reg:logistic logistic regression
        reg:logistic    逻辑回归
        binary:logistic    逻辑回归二分类， 输出为概率值
        binary:logitraw    逻辑回归 2分类，输出为logits之前的得分
        binary:hinge   用于二元分类的铰链损失。这使得预测为0或1，而不是产生概率。
        multi:softmax:  多分类，需要指定num_class的类别
        multi:softprob:  输出为概率  ndata*nclass 的矩阵，即，每行数据为分属类别的概率
        """

    def train(self, x_train, y_train=None, sample_weight=None,base_margin=None, eval_set=None,eval_metric=None,
              early_stopping_rounds=None, verbose=True, sample_weight_eval_set=None):
        # print(self.model)
        """
        :param x_train:     回归中，使用特征矩阵，  array
        :param y_train:      标签  array
        :param eval_metric
        :return:
        """
        # 默认开启过早停止

        # eval_metric in ["rmse","rmsle","mae","logloss","error","error@t", "merror","mlogloss","auc","aucpr",
        #                 "ndcg","map","ndcg@n", "map@n","ndcg-", "map-", "ndcg@n-", "map@n-","poisson-nloglik",
        #                 "gamma-nloglik","cox-nloglik","gamma-deviance","tweedie-nloglik","aft-nloglik"]
        # eval_metric   参数可为字符串， 也可以是列表字符串的形式

        if eval_metric:   # 若需要使用评估模型模式，
            assert eval_set   # 要确保   测试集是存在的。

        self.model.fit(X=x_train, y=y_train, sample_weight=sample_weight, base_margin=base_margin,
                       eval_set=eval_set, eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds,
                       verbose=verbose,sample_weight_eval_set=sample_weight_eval_set)


        # early_stopping_rounds=10  过早停止的条件  # 默认使用值为10
        # verbose=True  # 是否开启冗余

    def plot_loss(self):   # 绘制loss
        result =self.model.evals_result()  #获取模型结果
        epochs = len(result["validation_0"]["rmse"])
        x_axis = range(0, epochs)
        # 绘制loss曲线图
        figure, ax = plt.subplots()
        ax.plot(x_axis, result["validation_0"]["rmse"], label="Train")
        ax.plot(x_axis, result["validation_1"]["rmse"], label="Test")
        ax.legend()
        plt.ylabel("loss")
        plt.title("Xgboost Log Loss")
        plt.show()

    def predict(self, x_test):
        """
        :param x_test:  #使用np.array、scipy.sparse  用于预测
        :return:
        """
        my_pred = self.model.predict(data=x_test,output_margin=False,
                                     validate_features=True,base_margin=None)
        return my_pred

    def plt_importance(self,figure_path=None, ifsave=True): # 绘制重要性特征
        """
        :param figure_path:  图片保存路径
        :param ifsave:  是否保存图片
        :return:
        """
        # 绘制特征重要性
        fig, ax = plt.subplots(figsize=(15, 15))
        plot_importance(self.model,
                        height=0.5,
                        ax=ax,
                        max_num_features=64)  # 最多绘制64个特征
        if ifsave:
            if not figure_path:
                plt.savefig("../model/XGBboost_model/Xgboost_featute_importance_before.png")
            else:
                plt.savefig(figure_path)
        plt.show()  # 显示图片

    def _plt_importance_v1(self,columns_name,figure_path=None, ifsave=True): # 绘制重要性特征，使用实际的列名进行替换
        fig, ax = plt.subplots(figsize=(15, 15))
        plot_importance_v1(self.model, model_name="xgb",columns_name=columns_name,
                           height=0.5,
                           ax=ax,
                           max_num_features=64)  # 最多绘制64个特征
        if ifsave:
            if not figure_path:
                plt.savefig("../model/XGBboost_model/Xgboost_featute_importance_after.png")
            else:
                plt.savefig(figure_path)
        plt.show()  # 显示图片

    def plt_tree(self,num_tree):   # 绘制树
        """
        :param num_tree:  指定目标树的序号
        :return:
        """
        plot_tree(booster=self.model,num_trees=num_tree)

    def plot_graphviz(self, num_tree):   # 进行绘制graphviz
        to_graphviz(self.model, num_trees=num_tree)


    # 获取重要特征
    def get_importance(self):
        return self.model.feature_importances_

    # 评估函数
    def evaluate(self, y_test, my_pred,  evalue_fun="mse"):
        if evalue_fun == "acc":   # 准确率    分类指标
            result = accuracy_score(y_true=y_test, y_pred=my_pred)
            print("accuarcy:%.2f" % (result * 100.0))
        elif evalue_fun == "auc":   # auc 值   分类指标
            result = roc_auc_score(y_true=y_test, y_score=my_pred)
            print("auc:%.2f" %(result))
        elif evalue_fun == "mae":  # 回归指标， 平均绝对误差
            result = mean_absolute_error(y_true=y_test, y_pred=my_pred)
            print("mae:%.2f" %(result))
        elif evalue_fun == "median_ae":   # 种植绝对误差  回归指标
            result = median_absolute_error(y_true=y_test, y_pred=my_pred)
            print("median_ae:%.2f" %(result))
        elif evalue_fun =="r2_score":   # R平方值   回归指标
            result = r2_score(y_true=y_test, y_pred=my_pred)
            print("r2_score:%.2f" %(result))
        elif evalue_fun =="evs":   # 回归反差，    回归指标
            result = explained_variance_score(y_true=y_test, y_pred=my_pred)
            print("explained_variance_score:%.2f"%(result))
        elif evalue_fun =="aps":  #  分类指标， 根据预测得分计算平均精度(AP)
            result = average_precision_score(y_true=y_test, y_score=my_pred, average="maco", sample_weight=None)
            print("average_precision_score:%.2f"%(result))
        elif evalue_fun =="bsl":
            result = brier_score_loss(y_true=y_test, y_prob=my_pred, sample_weight=None, pos_label=None)
            print("brier_score_loss:%.2f"%(result))
        elif evalue_fun =="cmt": #计算混淆矩阵来评估分类的准确性   分类指标
            result = confusion_matrix(y_true=y_test, y_pred=my_pred, labels=None, sample_weight=None)
            print("confusion_matrix:%.2f"%(result))
        elif evalue_fun =="f1_score":  # f1 得分， 分类指标
            result = f1_score(y_true=y_test, y_pred=my_pred, labels=None, pos_label=1, average="binary", sample_weight=None) #F1值
            print("f1_score:%.2f"%(result))
        elif evalue_fun =="log_loss": # 交叉熵孙绍， 分类指标
            result = log_loss(y_true=y_test, y_pred=my_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)
            print("log_loss:%.2f"%(result))
        elif evalue_fun =="precision_score":   # 查准率   分类指标
            result = precision_score(y_true=y_test, y_pred=my_pred,labels=None, pos_label=1, average="binary")
            print("precision_score:%.2f"%(result))
        elif evalue_fun =="recall_score":  # 查全绿   分类指标
            result = recall_score(y_true=y_test, y_pred=my_pred, labels=None, pos_label=1,average="binary", sample_weight=None)
            print("recall_score:%.2f"%(result))
        elif evalue_fun =="roc_auc_score": # 计算 roc 曲线下面的面积就是AUC值，  分类指标
            result  = roc_auc_score(y_true=y_test, y_score=my_pred, average="macro", sample_weight=None)
            print("roc_auc_score:%.2f"%(result))
        elif evalue_fun =="roc_curve":  # 计算PROC曲线的横轴坐标  分类指标
            fpr, tpr, thresholds= roc_curve(y_true=y_test,y_score=my_pred,pos_label=None, sample_weight=None, drop_intermediate=True)
            result =( fpr, tpr, thresholds)
        else:   # mse 参数   均方差， 回归指标
            result = mean_squared_error(y_true=y_test, y_pred=my_pred)
            print("mse：%.2f" % (result))
        return result

    def save_model(self, save_params):  # 模型保存
        self.model.save_model(fname=save_params.get("fname", "../model/XGBboost_model/XGboostmodel.model")   # 保存的文件路径名字
                              # format=save_params.get("format", "cbm"),  # 保存的数据格式
                              # pool=save_params.get("pool", None)  #  训练使用的数据   模型保存成json格式，无需使用pool
                              )




def load_model_from_disk(model_path="../model/XGBboostmodel.model"):   # 加载模型
    models=xgb.Booster(model_file=model_path)
                                   # format=load_param.get("format", 'cbm'))
    return models


# 加载数据集成Dmatrix 数据格式
def load_data_format_DMatrix(file_type, file_path=None, dataset=None, label=None,
                             row=None, col=None):
    """
    :param file_type:  原始文件类型
    :param file_path:  文件路径
    :param dataset:  数据对象
    :param label:  标签
    :param row:
    :return:
    """
    # assert (file_type=="numpy" and dataset!=None and label!=None)
    if file_type == "csv":
        data = xgb.DMatrix(file_path+"？format=csv&label_column=0")
    elif file_type == "array":  # numpy 2D 阵列
        data = xgb.DMatrix(data=dataset,label=label)
    elif file_type =="dataFrame":   #  dataframe数据格式
        data = xgb.DMatrix(data=dataset, label=label)
    elif file_type =="scipy_sparse_array": # 稀疏阵列
        csr = csr_matrix((dataset, (row, col)))
        data = xgb.DMatrix(csr)
    else:  # svm
        data = xgb.DMatrix(file_path)
    return data


# 将 plot_importance_feature 中的f0  f1 转成对应 的列名
def feature_importance2columns(feature_importance_array, original_columns):
    feature_index = np.argsort(-feature_importance_array)
    return np.array(original_columns)[feature_index].tolist()


