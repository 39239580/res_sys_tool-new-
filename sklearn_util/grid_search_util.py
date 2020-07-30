from sklearn.model_selection import GridSearchCV  # 网格搜索的相关包
from sklearn.tree import DecisionTreeRegressor  # 决策回归树
from sklearn.ensemble import GradientBoostingRegressor  # 梯度提升回归树
from sklearn.linear_model import LinearRegression  # 线性回归树
from sklearn.neural_network import MLPRegressor  # MLP 回归树
from sklearn.ensemble import AdaBoostRegressor  # Ada 集成回归树
from sklearn.ensemble import BaggingRegressor  # Bagging 包集成树
from sklearn.ensemble import ExtraTreesRegressor  # 扩展树回归
from sklearn.ensemble import RandomForestRegressor  # 随机森林  回归树
from sklearn.svm import LinearSVR  # 近线性支持向量机
from sklearn.svm import NuSVR  # 非线性支持向量机
from sklearn.svm import SVR  # 支持向量机
from xgboost import XGBRegressor  # xgboost 分类回归树
from multiprocessing import cpu_count  # cpu核心个数
from catboost import CatBoostRegressor
import numpy as np

param_grid = dict()  # 网格参数


#  查找最佳模型参数
def search_best_model_parm(model_type, param_dict, x, y, cal_type="debug", scoring=None, cv=None, refit=True,
                           verbose=0, return_train_score=False):  #
    """
    :param model_type:   # 模型类型  # 除了超参数(需要寻找参数的最优的取值)以外的参数
    :param param_dict:   # 需要优化的参数的取值，值为字典或者列表 ， 需要寻优 的超参数
    :param cal_type:  #  debug   调试模型， 性能模型， 均衡模型
    :return:
    """
    assert model_type in ["DT", "GB", "LR", "MLP", "Ada", "Bag", "Ext", "RF", "LSVR", "NuSVR", "SVR", "XGB", "CB"]
    # ValueError("model_type is error")
    # 决策树回归
    assert cal_type in ["debug", "performance", "equilibrium"]  # 分别对应为 调试模型， 均衡模型， 以及相应的均衡模型

    model_dict = {"DT": DecisionTreeRegressor(),  # 决策树
                  "GB": GradientBoostingRegressor(),  # BGDT
                  "LR": LinearRegression(),  # 线性回归器
                  "mlp": MLPRegressor(),  # MLP回归树
                  "Ada": AdaBoostRegressor(),  # Ada提升术
                  "Bag": BaggingRegressor(),  # 带窗回归树
                  "Ext": ExtraTreesRegressor(),  # 扩展树回归
                  "RF": RandomForestRegressor(),  # 随机森林回归树
                  "LSVR": LinearSVR(),  # 线性SVR
                  "NuSVR": NuSVR(),  # 非线性SVR
                  "SVR": SVR(),  # SVR
                  "XGB": XGBRegressor(),  # XGB回归树
                  "CB": CatBoostRegressor(logging_level="Silent")}  # CatBoost 相关算法, 使用静默模式

    if cal_type == "debug":  # 调试模型
        n_jobs = 1
    elif cal_type == "performance":  # 性能模型
        n_jobs = cpu_count()
    else:  # 均衡模型
        n_jobs = cpu_count() // 2

    grid = GridSearchCV(estimator=model_dict.get(model_type, "XGB"), param_grid=param_dict,
                        scoring=scoring, n_jobs=n_jobs, pre_dispatch=n_jobs, iid=False,
                        cv=cv, refit=refit, verbose=verbose, error_score=np.nan, return_train_score=return_train_score)
    """
    estimator  :  使用的分类器， 并且传入除需要确定最佳的参数之外的参数，每个分类器都需要一个scoring参数， 或者score方法
    param_grid : 最要被优化的参数的取值， 值为字典或列表, param_grid = param_test1,  如：param_test1 = {"n_estimators":range(10,71,10)}   
    scoring :  默认为None,   str   ，  列表/元组或字典。    
    每个 评估器件，scoring 中需要指定一个， 若评估器内没指定， scoring 需要指定， 当scoring为None 时，  使用评估器中默认的score 函数

    n_jobs :  默认为None,  1, 代表单线程，   -1 为多线程  
    pre_dispatch :   n_jobs   并行执行期间调度的作业数
    iid :  False. bool 型参数，    True 是， 将每个测试集的样本进行加权。
    cv : 默认为None, None 为使用默认的5折，   整数的时候，指定合适的折数， 或者使用cv_split
    refit :   使用找到的最佳参数重新拟合评估器 ， 默认为TRUE
    verbose :  显示打印信息， 0 不显示， 1 显示打印进度条
    error_score :  拟合过程中，若出错，使用这个数值进行填充 一般使用nan
    return_train_score:  bool 型， 默认为False,  不输出 训练分数 
    # 一般使用到   estimator, param_grid,scoring, n_jobs, cv, verbose

    """

    grid_result = grid.fit(X=x, y=y)  # 训练之后的数据
    return grid_result


# 获取模型中的相关属性
def get_best_model_attribute(grid_model, attribute_type):
    assert attribute_type in ["cv_result", "best_estimator", "best_score", "best_params",
                              "best_index", "score", "n_splits", "grid_score", "refit_time"]
    if attribute_type == "cv_result":
        return grid_model.cv_results_  # 数组型的字典
    elif attribute_type == "best_estimator":  # 铜鼓搜索选择的估计器
        return grid_model.best_estimator_
    elif attribute_type == "best_score":
        return grid_model.best_score_  # best_estimator 的分数  best_estimator_ 的平均交叉验证得分
        # 提供优化过程期间观察到的最好的评分
    elif attribute_type == "best_params":  # 在保存数据上给出最佳结果的参数设置， 已经取得的最佳结果的参数组合
        return grid_model.best_params_
    elif attribute_type == "best_index":  # 对应于最佳候选参数设置的索引（cv_results数组），
        return grid_model.best_index_
    elif attribute_type == "score":  #
        return grid_model.score_
    elif attribute_type == "n_splits":  # 交叉验证折数
        return grid_model.n_splits_
    elif attribute_type == "grid_score":
        return grid_model.grid_scores_  # 网格的分数值  给出不同参数情况下的评估结果
    else:  # refit_time
        return grid_model.refit_time  # 用于整个数据集重新拟合最佳模型的秒数

# GridSearchCV(x)

class GridSearch(object):
    def __init__(self,estimator, param_grid, scoring=None,
                 n_jobs=None, iid=False, refit=True, cv=None,
                 verbose=0, pre_dispatch=None,
                 error_score=np.nan, return_train_score=False):
        
        self.gridsearch = GridSearchCV(estimator, param_grid, scoring=scoring,
                                       n_jobs=n_jobs, iid=iid, refit=refit, cv=cv,
                                       verbose=verbose, pre_dispatch=pre_dispatch,
                                       error_score=np.nan, return_train_score=return_train_score)
    
    def fit(self, x, y=None):
        return self.randomsearch.fit(X=x, y=y)
    
    def transform(self, x):
        return self.randomsearch.transform(x=x)

    
    def predict(self, x):
        return self.randomsearch.predict(x=x)

    def predict_log_proba(self, x):
        return self.randomsearch.predict_log_proba(X=x)
    
    def predict_proba(self, x):
        return self.randomsearch.predict_proba(X=x)
    
    def inverse_transform(self, xt):
        return self.randomsearch.inverse_transform(Xt=xt)
    
    def decision_function(self, x):  # refit=True下才支持decision_function
        return self.randomsearch.decision_function(X=x)
    
    def set_params(self,params):
        self.randomsearch.set_params(params)
    
    def get_params(self, deep=True):
        return self.randomsearch.get_params(deep=deep)
    
    def get_score(self, x, y=None):
        return self.randomsearch.score(X=x,y=y)
        
    def get_attribute(self, attribute_name):
        assert attribute_name in ["cv_result", "best_estimator", "best_score", "best_params",
                                  "best_index", "score", "n_splits", "grid_score", "refit_time"]
        if attribute_name == "cv_result":
            return self.randomsearch.cv_results_
        elif attribute_name == "best_estimator":
            return self.randomsearch.best_estimator_
        elif attribute_name == "best_score":
            return self.randomsearch.best_score_
        elif attribute_name == "best_params":
            return self.randomsearch.best_params_
        elif attribute_name == "best_index":
            return self.randomsearch.best_index_
        elif attribute_name =="score":
            return self.randomsearch.scorer_
        elif attribute_name =="n_split":
            return self.randomsearch.n_splits_
        elif attribute_name =="refit-time":
            return self.randomsearch.refit_time_
        else:
            ValueError("输入的属性名称有误, 请输入正确的属性名称")


