from sklearn.model_selection import cross_val_score   # 交叉验证得分
from sklearn.tree import DecisionTreeRegressor   # 决策回归树
from sklearn.ensemble import GradientBoostingRegressor  # 梯度提升回归树
from sklearn.model_selection import KFold  # k折交叉
from sklearn.linear_model import LinearRegression   # 线性回归树
from sklearn.neural_network import MLPRegressor  # MLP 回归树
from sklearn.ensemble import AdaBoostRegressor  # Ada 集成回归树
from sklearn.ensemble import BaggingRegressor   # Bagging 包集成树
from sklearn.ensemble import ExtraTreesRegressor  # 扩展树回归
from sklearn.ensemble import RandomForestRegressor   # 随机森林  回归树
from sklearn.svm import LinearSVR   # 近线性支持向量机
from sklearn.svm import NuSVR  # 非线性支持向量机
from sklearn.svm import SVR   #  支持向量机
from xgboost import XGBRegressor  # xgboost 分类回归树
from sklearn.model_selection import ShuffleSplit  #
import pandas as pd
from sklearn.model_selection import cross_validate
from catboost import CatBoostRegressor


pd.set_option("display.max_columns", None)
def model_compare( X, y, n_split=5, shuffle= True, model_type="DT",kf=None, is_disp_score=True):
    """
    :param X:   特征矩阵
    :param y:   label 值
    :param n_split:  急着交叉的参数
    :param shuffle:   是否将数据进行打乱
    :param model_type:  # 选择相关模型进行操作
    :param kf:  # 交叉验证的参数
    :param is_disp_score:  #  是否直接显示
    :return:
    """
    # 确保正确的取值范围
    assert model_type in ["DT", "GB", "LR", "MLP", "Ada", "Bag", "Ext", "RF","LSVR", "NuSVR", "SVR", "XGB","CB"]
    # ValueError("model_type is error")
    # 决策树回归
    if model_type =="DT":  #
        alg_model = DecisionTreeRegressor()
    #  梯度提升树
    elif model_type =="GB":
        alg_model = GradientBoostingRegressor()
    elif model_type =="LR":
        alg_model  = LinearRegression()
    elif model_type =="MLP":
        alg_model =MLPRegressor(max_iter=10000)
    elif model_type == "Ada":
        alg_model =AdaBoostRegressor()
    elif model_type == "Bag":
        alg_model = BaggingRegressor()
    elif model_type == "Ext":
        alg_model = ExtraTreesRegressor()
    elif model_type == "RF":
        alg_model =RandomForestRegressor()
    elif model_type == "LSVR":
        alg_model = LinearSVR(max_iter=10e+8)
    elif model_type == "NuSVR":
        alg_model  = NuSVR()
    elif model_type == "SVR":
        alg_model = SVR()
    elif model_type == "CB":
        alg_model = CatBoostRegressor(logging_level="Silent")  # 进行静默， 不输出很多信息
    else:
        alg_model = XGBRegressor()
    if not kf:
        kf = KFold(n_splits=n_split, shuffle=shuffle)  # 交叉验证对象
    score_ndarray = cross_val_score(estimator=alg_model, X=X, y=y, cv=kf )  # 分数
    cv_results = cross_validate(estimator=alg_model, X=X, y=y,cv=kf,return_train_score=True)

    if is_disp_score:
        print(score_ndarray)
        print("模型名称：%s"%model_type)
        print("模型最佳得分均值：%.6f"%(score_ndarray.mean()))   # 数据得分 均值
    return  alg_model, score_ndarray,cv_results

# 模型全面比较
def model_compare_comparison(estimator_list, X, y, shuffle=True,
                             n_split=6, train_size=0.7,test_size=0.2, random_state=27, df_columns=None,
                             is_show=True, sorted_list=None):
    """
    :param estimator_list:   相关回归器名称的 list
    :param X:
    :param y：
    :shuffle：   是否开启打乱
    :param n_split:   # 交叉的折数
    :param train_size:   训练集的大小
    :param test_size:   测试集的大小
    :param random_state:  随机种子
    :param df_columns:  默认为None, 使用默认的None
    :param is_show: 默认为True
    :param sorted_list:  使用相关的 排序的列名 来进行操作
    :return:
    """
    assert  not df_columns
    # 交叉验证集对象
    cv_split= ShuffleSplit(n_splits=n_split, train_size=train_size, test_size=test_size, random_state=random_state)
    if not df_columns:
        df_columns = ['Name', 'Parameters', 'Train Accuracy Mean', 'Test Accuracy Mean',
                      'Test Accuracy Std', 'Comsumed Time']
    df = pd.DataFrame(columns=df_columns)

    row_index = 0
    for estimators in estimator_list:
        print(estimators)
        estimator,_,cv_results=model_compare(X=X, y=y, shuffle=shuffle, model_type=estimators, kf=cv_split, is_disp_score=True)
        # print(estimator)
        # print(_)
        # print(cv_results)
        # raise

        df.loc[row_index, 'Name'] = estimator.__class__.__name__
        df.loc[row_index, 'Parameters'] = str(estimator.get_params())
        # print(cv_results)
        # print(cv_results["train_score"])
        # print(df)
        # raise
        df.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_score'].mean()
        df.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_score'].mean()
        df.loc[row_index, 'Test Accuracy Std'] = cv_results['test_score'].std()
        df.loc[row_index, 'Comsumed Time'] = cv_results['fit_time'].mean()
        # print(row_index, estimator.__class__.__name__)
        # print(cv_results['test_score'])
        row_index += 1

    if not sorted_list:
        sorted_list = "Test Accuracy Mean"
    df = df.sort_values(by=sorted_list, ascending=False)  # 按照测试集进行排序， 降序排列
    if is_show:
        print(df)
    return df
