from sklearn.model_selection import RandomizedSearchCV
import numpy as np
# from sklearn.datasets import load_iris
# from scipy.stats import uniform
# from sklearn.linear_model import LogisticRegression

# 随即搜索
class RandomSearch(object):
    def __init__(self,estimator, param_distributions, n_iter=10, scoring=None, n_jobs=None, iid=False, 
                 refit=True,cv=None, verbose=0, pre_dispatch=None,
                 random_state=None, error_score=np.nan,
                 return_train_score=False):
        """
        estimator  :  使用的分类器， 并且传入除需要确定最佳的参数之外的参数，每个分类器都需要一个scoring参数， 或者score方法
        param_distributions : 最要被优化的参数的取值， 值为字典或列表, param_grid = param_test1,  如：param_test1 = {"n_estimators":range(10,71,10)}   
          
        每个 评估器件，scoring 中需要指定一个， 若评估器内没指定， scoring 需要指定， 当scoring为None 时，  使用评估器中默认的score 函数
        n_iter:   int 默认为 10
        scoring :  默认为None,   str   ，  列表/元组或字典。  
        n_jobs :  默认为None,int,  1, 代表单线程，   -1 为多线程  
        iid :  False. bool 型参数，    True 是， 将每个测试集的样本进行加权。
        refit :   使用找到的最佳参数重新拟合评估器 ， 默认为TRUE
        cv : 默认为None, None 为使用默认的5折，   整数的时候，指定合适的折数， 或者使用cv_split
        verbose :  显示打印信息， 0 不显示， 1 显示打印进度条
        pre_dispatch :   n_jobs   并行执行期间调度的作业数   "2*n_jobs"  or int
        error_score :  拟合过程中，若出错，使用这个数值进行填充 一般使用nan
        return_train_score:  bool 型， 默认为False,  不输出 训练分数 
        # 一般使用到   estimator, param_grid,scoring, n_jobs, cv, verbose

        """
        self.randomsearch = RandomizedSearchCV(self, estimator=estimator,
                                               param_distributions=param_distributions, 
                                               n_iter=n_iter,
                                               scoring=scoring, 
                                               n_jobs=n_jobs, 
                                               iid=iid, 
                                               refit=refit,
                                               cv=cv, verbose=verbose, 
                                               pre_dispatch=pre_dispatch,
                                               random_state=random_state, 
                                               error_score=error_score,
                                               return_train_score=return_train_score)
    
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
        elif attribute_name =="scorer":
            return self.randomsearch.scorer_
        elif attribute_name =="n_split":
            return self.randomsearch.n_splits_
        elif attribute_name =="refit-time":
            return self.randomsearch.refit_time_
        else:
            ValueError("输入的属性名称有误, 请输入正确的属性名称")

    