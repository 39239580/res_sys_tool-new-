import catboost as cb
from multiprocessing import cpu_count
from catboost.core import EFstrType
from catboost import Pool,cv
import matplotlib.pyplot as plt
from sklearn_util.plot_custom_util import plot_importance_v1


class Catboost(object):
    def __init__(self, task_type, module_type, compute_task, **params):
        """
        :param task_type:   # 任务类型  cal 或 reg
        :param module_type:
        :param compute_task:
        :param params:
        """
        assert  task_type in ["cla", "reg"]   # 两种类型
        assert  module_type in ["balance", "debug", "performance"]   # 三种 性能模型
        assert  compute_task in ["GPU", "CPU"]

        self.task_type = task_type  # cal   或使用reg
        self.module_type = module_type  # 模块
        if self.module_type == "debug":
            params["thread_count"] = 1
        elif self.module_type == "performance": # 性能模型
            params["thread_count"] = cpu_count()  # cpu核心数
        else:  # 均衡模型
            params["thread_count"] = cpu_count() // 2

        #通用参数
        # learning_rate(eta) = automatically
        # depth(max_depth) = 6: 树的深度
        # l2_leaf_reg(reg_lambda) = 3  L2正则化系数
        # n_estimators(num_boost_round)(num_trees=1000) = 1000: 解决ml问题的树的最大数量  基分类器的数量
        # one_hot_max_size = 2: 对于某些变量进行one - hot编码

        # loss_function = “Logloss”
        # loss_function in ["Logloss","RMSE","MAE","CrossEntropy","MultiClass", "MultiClassOneVsAll"] 或使用自定义函数

        # custom_metric = None  自定义指标
        # custom_metric in ["RMSE","Logloss","MAE","CrossEntropy","Recall","Precision","F1","Accuracy","AUC","R2"]

        # eval_metric = Optimized objective 优化目标
        # eval_metric in ["RMSE","Logloss","MAE","CrossEntropy","Recall","Precision","F1","Accuracy","AUC","R2"]

        # nan_mode = None：处理NAN的方法
        # nan_mode in ["Forbidden","Min","Max"]

        # leaf_estimation_method = None：迭代求解的方法，梯度和牛顿
        # leaf_estimation_method in ["Newton","Gradient"]

        # random_seed = None: 训练时候的随机种子

        # 性能参数
        # thread_count = -1：训练时所用的cpu / gpu核数
        # used_ram_limit = None：CTR问题，计算时的内存限制
        # gpu_ram_part = None：GPU内存限制
        # 处理单元设置
        # task_type = CPU：训练的器件
        # devices = None：训练的GPU设备ID
        # counter_calc_method = None,
        # leaf_estimation_iterations = None,
        # use_best_model = None,
        # verbose = None,
        # model_size_reg = None,
        # rsm = None,
        # logging_level = None,
        # metric_period = None,
        # ctr_leaf_count_limit = None,
        # store_all_simple_ctr = None,
        # max_ctr_complexity = None,
        # has_time = None,
        # classes_count = None,
        # class_weights = None,
        # random_strength = None,
        # name = None,
        # ignored_features = None,
        # train_dir = None,
        # custom_loss = None,
        # bagging_temperature = None
        # border_count = None
        # feature_border_type = None,
        # save_snapshot = None,
        # snapshot_file = None,
        # fold_len_multiplier = None,
        # allow_writing_files = None,
        # final_ctr_computation_mode = None,
        # approx_on_full_history = None,
        # boosting_type = None,
        # simple_ctr = None,
        # combinations_ctr = None,
        # per_feature_ctr = None,
        # device_config = None,
        # bootstrap_type = None,
        # subsample = None,
        # colsample_bylevel = None,
        # random_state = None,
        # objective = None,
        # max_bin = None,
        # scale_pos_weight = None,
        # gpu_cat_features_storage = None,
        # data_partition = None
        self.compute_task = compute_task

        if self.compute_task =="gpu":  #
            params["task_type"] = "GPU"
        else:
            params["task_type"] = "CPU"



        if self.task_type =="reg":   #  做回归任务
            """  
            # 使用相关的成本函数， RMSE, MultiRMSE, MAE, Quantile, LogLinQuantile, Poisson, MAPE, Lq or custom objective object"
            """
            self.model =cb.CatBoostRegressor(iterations=None,
                                             learning_rate=params.get("leaning_rate",None),  #  学习率
                                             depth=params.get("depth",None),   # 深度
                                             l2_leaf_reg=params.get("l2_leaf_reg",None), #l2 正则
                                             model_size_reg=params.get("model_size_reg",None),
                                             rsm=params.get("rms",None),  #
                                             loss_function=params.get("loss_function",'RMSE'),  # 损失函数值
                                             border_count=params.get("border_count",None), # 边界树
                                             feature_border_type=params.get("feature_border_type",None),
                                             per_float_feature_quantization=params.get("per_float_feature_quantization",None),
                                             input_borders=params.get("input_borders",None),
                                             output_borders=params.get("output_borders",None),
                                             fold_permutation_block=params.get("fold_permutation_block",None),
                                             od_pval=params.get("od_pval",None),
                                             od_wait=params.get("od_wait",None),
                                             od_type=params.get("od_type",None),
                                             nan_mode=params.get("nan_mode",None),
                                             counter_calc_method=params.get("counter_calc_method",None),
                                             leaf_estimation_iterations=params.get("leaf_estimation_iterations",None),
                                             leaf_estimation_method=params.get("leaf_estimation_method", None),# 叶子及分类器方法
                                             thread_count=params.get("thread_count",None), # 线程数
                                             random_seed=params.get("random_seed",None),# 随机种子
                                             use_best_model=params.get("use_best_model",None),
                                             best_model_min_trees=params.get("best_model_min_trees",None), # 最好模型最小数
                                             verbose=params.get("verbose",None),
                                             silent=params.get("silent",None),
                                             logging_level=params.get("logging_level",None),
                                             metric_period=params.get("metric_period",None),
                                             ctr_leaf_count_limit=params.get("ctr_leaf_count_limit",None),
                                             store_all_simple_ctr=params.get("store_all_simple_ctr",None),
                                             max_ctr_complexity=params.get("max_ctr_complexity",None),
                                             has_time=params.get("has_time",None),
                                             allow_const_label=params.get("allow_const_label",None),
                                             one_hot_max_size=params.get("one_hot_max_size",None),
                                             random_strength=params.get("random_strength",None),
                                             name=params.get("name",None),
                                             ignored_features=params.get("ignored_features",None),
                                             train_dir=params.get("train_dir",None),
                                             custom_metric=params.get("custom_metric",None),
                                             eval_metric=params.get("eval_metric",None),
                                             bagging_temperature=params.get("bagging_temperature",None),
                                             save_snapshot=params.get("save_snapshot",None),
                                             snapshot_file=params.get("snapshot_file",None),
                                             snapshot_interval=params.get("snapshot_interval",None),
                                             fold_len_multiplier=params.get("fold_len_multiplier",None),
                                             used_ram_limit=params.get("used_ram_limit",None),
                                             gpu_ram_part=params.get("gpu_ram_part",None),
                                             pinned_memory_size=params.get("pinned_memory_size",None),
                                             allow_writing_files=params.get("allow_writing_files",None),
                                             final_ctr_computation_mode=params.get("final_ctr_computation_mode",None),
                                             approx_on_full_history=params.get("final_ctr_computation_mode",None),
                                             boosting_type=params.get("boosting_type",None),
                                             simple_ctr=params.get("simple_ctr",None),
                                             combinations_ctr=params.get("combinations_ctr",None),
                                             per_feature_ctr=params.get("per_feature_ctr",None),
                                             ctr_target_border_count=params.get("ctr_target_border_count",None),
                                             task_type=params.get("task_type",None),  # cpu 或GPU
                                             device_config=params.get("device_config",None),
                                             devices=params.get("devices",None), # 训练的gpu设备ID
                                             bootstrap_type=params.get("bootstrap_type",None),
                                             subsample=params.get("subsample",None),
                                             sampling_unit=params.get("sampling_unit",None),
                                             dev_score_calc_obj_block_size=params.get("dev_score_calc_obj_block_size",None),
                                             max_depth=params.get("max_depth",None),  # 最大树的深度，默认为6 ==depth
                                             n_estimators=params.get("n_estimators",None), # 基分类器的数量，
                                             # 决ml伪命题的树的最大数量，默认值为1000，==num_boost_round, ==num_trees=1000
                                             num_boost_round=params.get("num_boost_round",None), # 提升轮数树
                                             num_trees=params.get("num_trees",None), # 树数量
                                             colsample_bylevel=params.get("colsample_bylevel",None),
                                             random_state=params.get("random_state",None),  # 随机种子
                                             reg_lambda=params.get("reg_lambda",None), # 正则化参数lambda
                                             objective=params.get("objective",None), # 目标函数
                                             eta=params.get("eta",None),
                                             max_bin=params.get("max_bin",None),
                                             gpu_cat_features_storage=params.get("gpu_cat_features_storage",None),
                                             data_partition=params.get("data_partition",None),
                                             metadata=params.get("metadata",None),
                                             early_stopping_rounds=params.get("early_stopping_rounds",None), # 过早停止代数
                                             cat_features=params.get("cat_features",None),
                                             grow_policy=params.get("grow_policy",None),
                                             min_data_in_leaf=params.get("min_data_in_leaf",None), # 叶子中的最小数
                                             min_child_samples=params.get("min_child_samples",None), # 最小子样本
                                             max_leaves=params.get("max_leaves",None), # 最大叶子数
                                             num_leaves=params.get("num_leaves",None), # 叶子数量
                                             score_function=params.get("score_function",None), # 得分函数
                                             leaf_estimation_backtracking=params.get("leaf_estimation_backtracking",None),
                                             ctr_history_unit=params.get("ctr_history_unit",None),
                                             monotone_constraints=params.get("monotone_constraints",None),
                                             feature_weights=params.get("feature_weights",None),  # 特征全面直接拍卖行
                                             penalties_coefficient=params.get("penalties_coefficient",None),
                                             first_feature_use_penalties=params.get("first_feature_use_penalties",None),
                                             model_shrink_rate=params.get("model_shrink_rate",None),
                                             model_shrink_mode=params.get("model_shrink_mode",None),
                                             langevin=params.get("langevin",None),
                                             diffusion_temperature=params.get("diffusion_temperature",None),
                                             boost_from_average=params.get("boost_from_average",None)
                                             )

        else:  # 做胡分类任务
            self.model=cb.CatBoostClassifier(iterations=None, # 迭代数， 通用参数
                                             learning_rate=params.get("leaning_rate", None),  #学习率，通用参数
                                             depth=params.get("depth", None),# 树的深度，
                                             l2_leaf_reg=params.get("l2_leaf_reg", None), # l2 正则化参数
                                             model_size_reg=params.get("model_size_reg", None),
                                             rsm=params.get("rms", None),
                                             loss_function=params.get("loss_function", None),
                                             border_count=params.get("border_count", None),
                                             feature_border_type=params.get("feature_border_type", None),
                                             per_float_feature_quantization=params.get("per_float_feature_quantization", None),
                                             input_borders=params.get("input_borders", None),
                                             output_borders=params.get("output_borders", None),
                                             fold_permutation_block=params.get("fold_permutation_block", None),
                                             od_pval=params.get("od_pval", None),
                                             od_wait=params.get("od_wait", None),
                                             od_type=params.get("od_type", None),
                                             nan_mode=params.get("nan_mode", None),
                                             counter_calc_method=params.get("counter_calc_method", None),
                                             leaf_estimation_iterations=params.get("leaf_estimation_iterations", None),
                                             leaf_estimation_method=params.get("leaf_estimation_method", None),
                                             thread_count=params.get("thread_count", None),  # 性能参数，使用-1时，
                                             # 使用过最大的cpu核心数进行训练
                                             random_seed=params.get("random_seed", None),
                                             use_best_model=params.get("use_best_model", None),
                                             # best_model_min_trees=params.get("best_model_min_trees", None),
                                             verbose=params.get("verbose", None),
                                             # silent=params.get("silent", None),
                                             logging_level=params.get("logging_level", None),
                                             metric_period=params.get("metric_period", None),
                                             ctr_leaf_count_limit=params.get("ctr_leaf_count_limit", None),
                                             store_all_simple_ctr=params.get("store_all_simple_ctr", None),
                                             max_ctr_complexity=params.get("max_ctr_complexity", None),
                                             has_time=params.get("has_time", None),
                                             allow_const_label=params.get("allow_const_label", None),
                                             one_hot_max_size=params.get("one_hot_max_size", None), # one_hot编码的最大尺寸
                                             random_strength=params.get("random_strength", None),
                                             name=params.get("name", None),
                                             ignored_features=params.get("ignored_features", None),
                                             train_dir=params.get("train_dir", None),
                                             custom_loss=params.get("custom_loss",None),
                                             custom_metric=params.get("custom_metric", None),
                                             eval_metric=params.get("eval_metric", None),
                                             bagging_temperature=params.get("bagging_temperature", None),
                                             save_snapshot=params.get("save_snapshot", None),
                                             snapshot_file=params.get("snapshot_file", None),
                                             snapshot_interval=params.get("snapshot_interval", None),
                                             fold_len_multiplier=params.get("fold_len_multiplier", None),
                                             used_ram_limit=params.get("used_ram_limit", None), #CTR问题，
                                             # 计算时的内存限制 性能参数
                                             gpu_ram_part=params.get("gpu_ram_part", None), # 性能参数， GPU显存限制
                                             # pinned_memory_size=params.get("pinned_memory_size", None),
                                             allow_writing_files=params.get("allow_writing_files", None),
                                             final_ctr_computation_mode=params.get("final_ctr_computation_mode", None),
                                             approx_on_full_history=params.get("final_ctr_computation_mode", None),
                                             boosting_type=params.get("boosting_type", None),
                                             simple_ctr=params.get("simple_ctr", None),
                                             combinations_ctr=params.get("combinations_ctr", None),
                                             per_feature_ctr=params.get("per_feature_ctr", None),
                                             # ctr_target_border_count=params.get("ctr_target_border_count", None),
                                             task_type=params.get("task_type", None),
                                             device_config=params.get("device_config", None),
                                             devices=params.get("devices", None),
                                             bootstrap_type=params.get("bootstrap_type", None),
                                             subsample=params.get("subsample", None),
                                             sampling_unit=params.get("sampling_unit", None),
                                             dev_score_calc_obj_block_size=params.get("dev_score_calc_obj_block_size", None),
                                             max_depth=params.get("max_depth", None),
                                             n_estimators=params.get("n_estimators", None),
                                             num_boost_round=params.get("num_boost_round", None),
                                             num_trees=params.get("num_trees", None),
                                             colsample_bylevel=params.get("colsample_bylevel", None),
                                             random_state=params.get("random_state", None),
                                             reg_lambda=params.get("reg_lambda", None), # 正则化参数， l2, ==l2_leaf_reg
                                             objective=params.get("objective", None),
                                             eta=params.get("eta", None),  # 使用自动的学习率 ==learning_rate
                                             max_bin=params.get("max_bin", None),
                                             scale_pos_weight=params.get("scale_pos_weight",None),
                                             gpu_cat_features_storage=params.get("gpu_cat_features_storage", None),
                                             data_partition=params.get("data_partition", None),
                                             metadata=params.get("metadata", None),
                                             early_stopping_rounds=params.get("early_stopping_rounds", None),
                                             cat_features=params.get("cat_features", None),
                                             grow_policy=params.get("grow_policy", None),
                                             min_data_in_leaf=params.get("min_data_in_leaf", None),
                                             min_child_samples=params.get("min_child_samples", None),
                                             max_leaves=params.get("max_leaves", None),
                                             num_leaves=params.get("num_leaves", None),
                                             score_function=params.get("score_function", None),
                                             leaf_estimation_backtracking=params.get("leaf_estimation_backtracking", None),
                                             ctr_history_unit=params.get("ctr_history_unit", None),
                                             monotone_constraints=params.get("monotone_constraints", None),
                                             feature_weights=params.get("feature_weights", None),
                                             penalties_coefficient=params.get("penalties_coefficient", None),
                                             first_feature_use_penalties=params.get("first_feature_use_penalties", None),
                                             model_shrink_rate=params.get("model_shrink_rate", None),
                                             model_shrink_mode=params.get("model_shrink_mode", None),
                                             langevin=params.get("langevin", None),
                                             diffusion_temperature=params.get("diffusion_temperature", None),
                                             boost_from_average=params.get("boost_from_average", None),
                                             text_features=params.get("text_features",None),
                                             tokenizers=params.get("tokenizers",None),
                                             dictionaries=params.get("dictionaries",None),
                                             feature_calcers=params.get("feature_calcers",None),
                                             text_processing=params.get("text_processing",None)
                                             )

    # 进行训练
    def train(self, X, y=None, cat_features=None, text_features=None, sample_weight=None, baseline=None,
              use_best_model=None,eval_set=None, verbose=None, logging_level=None, plot=False, column_description=None,
              verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None,
              save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None):
        """
        :param X: catboost.Pool， list， array, df, Series， Sparse.df  spmatrix
        :param y:  list  array, df  Series
        :param cat_features:  list, array 一个一维的分类列索引数组。 拿来做处理的类别特征
        :param text_features:  listm array 文本列的一维数组索引(指定为整数)或名称(指定为字符串)
        :param sample_weight: list， array, df， Series   样本权重， 默认值为1
        :param baseline:   list, array， 所有输入对象的公式值数组。训练从所有输入对象的这些值开始，而不是从零开始。
        :param use_best_model: bool型  有验证集，使用True, 否则使用False
        :param eval_set: catboost.Pool , catboost.Pool 的列表， tuple(x,y), tuple(x,y)列表。 string,
        数据集文件的路径， 数据集文件的路径列表   用于存放验证集 验证集
        :param verbose: bool/int 默认值使用1， False为静默模式不输出
        :param logging_level: 有取值slinet， Verbose, info, Debug  string型数据
        :param plot: 默认为False,  绘制评估值， 自定义loss函数值， 训练时间与剩余时间
        :param column_description: 列名描述
        :param verbose_eval:  是否输出
        :param metric_period: int 1   被设置为正数， 迭代计算目标和评估值的频率
        :param silent: Flase，  日志模式
        :param early_stopping_rounds: int, 默认为False 设置过拟合的预制
        :param save_snapshot: 是否保存快照，bool， None
        :param snapshot_file: 快照保存名字，
        :param snapshot_interval:int, 快照保存时间间隔600s
        :param init_model:  # None,   初始化模型，用于集成等操作
        :return:
        """
        if self.task_type =="cal":
            self.model.fit(X=X, y=y, cat_features=cat_features, text_features=text_features, sample_weight=sample_weight,
                           baseline=baseline, use_best_model=use_best_model, eval_set=eval_set, verbose=verbose,
                           logging_level=logging_level, plot=plot, column_description=column_description,
                           verbose_eval=verbose_eval, metric_period=metric_period, silent=silent,
                           early_stopping_rounds=early_stopping_rounds,
                           save_snapshot=save_snapshot, snapshot_file=snapshot_file,
                           snapshot_interval=snapshot_interval, init_model=init_model)
        else:
            self.model.fit(X=X, y=y, cat_features=cat_features, sample_weight=sample_weight,baseline=baseline,
                           use_best_model=use_best_model, eval_set=eval_set, verbose=verbose,
                           logging_level=logging_level, plot=plot, column_description=column_description,
                           verbose_eval=verbose_eval, metric_period=metric_period, silent=silent,
                           early_stopping_rounds=early_stopping_rounds,
                           save_snapshot=save_snapshot, snapshot_file=snapshot_file,
                           snapshot_interval=snapshot_interval, init_model=init_model)

    # 进行预测
    def predict(self,X,prediction_type="Class", ntree_start=0, ntree_end=0, thread_count=-1, verbose=None):
        """
        :param X:
        :param prediction_type:
        :param ntree_start:
        :param ntree_end:
        :param thread_count:
        :param verbose:
        :return:
        """
        if self.task_type =="cal":   # 做 分类任务
            predict = self.model.predict(data=X,prediction_type=prediction_type,ntree_start=ntree_start, ntree_end=ntree_end,
                                        thread_count=thread_count,verbose=verbose)
        else:
            predict = self.model.predict(data=X,prediction_type=None, ntree_start=ntree_start, ntree_end=ntree_end,
                                         thread_count=thread_count,verbose=verbose)
        return predict


    # 绘制 树
    def plot_cattree(self,tree_idx, pool):
        self.model.plot_tree(tree_idx=tree_idx, pool=pool)

    # 获取每份验证集上的最好性能计算
    def get_best_scores(self):
        return self.model.get_best_score()

    # 最后一个验证集上评估指标或损失函数的最佳结果。
    def get_best_iteration(self):
        return self.model.get_best_iteration()

    # 所有的训练参数
    def get_all_param(self):
        return self.model.get_all_params()

    # 获取边界  返回浮点型特征的索引
    def get_borders_fn(self):
        return self.model.get_borders()

    # 获取评估结果
    def get_evals_result(self):
        """
        :return:  仅仅输出计算评估值
        """
        # 使用默认的参数的时候， 不输出以下几种指标的值
        # """PFound, yETIRank 的值，
        # """
        return self.model.get_evals_result()

    def get_object_impotance(self, pool, train_pool, top_size=-1, type='Average', update_method='SinglePoint',
            importance_values_sign='All', thread_count=-1, verbose=False, ostr_type=None):
        """
        :param pool:   数据类型为Pool   要求用参数
        :param train_pool:    Pool 数据类型，    要求使用参数
        :param top_size:   int   默认值-1 无限制
        :param type:  类型， 默认值使用“Average”  来自输入数据集中的每个对象的训练数据集中的对象得分的平均值。
         还可取值"PerObject",输入数据集中每个对象在训练数据集中的得分。
        :param update_method: 默认取值"SinglePoint","TopKLeaves", "ALLPoints", “top"
        SinglePoint  最快但是最不准的方法. TopKLeaves指定叶子的数量。值越高，计算越精确，速度越慢。ALLPoint 最慢且精度最高
        :param importance_values_sign: string  "Positive"， "Negative", "ALL", 默认使用全部样本
        :param thread_count:  线程数  int   默认-1
        :param verbose:  是否打印显示， 可选择1或 True
        :param ostr_type:   #不用管可以忽略
        :return:
        """
        return self.model.get_object_importance(pool=pool, train_pool=train_pool, top_size=top_size,
                                                type=type, update_method=update_method,
                                                importance_values_sign=importance_values_sign,
                                                thread_count=thread_count, verbose=verbose, ostr_type=ostr_type)



    # 获取重要性特征
    def get_feature_importance_(self,type="FeatureImportance"):
        """
        :param type: "PredictionValuesChange"  计算每个特征的得分
                     "LossFunctionChange"   根据loss计算每个特征值的得分
                     "FeatureImportance"
                     "ShapValues"
                     "ShapInteractionValues"
                     "Interaction"
                     "PredictionDiff"
        :return:
        """
        assert type in ["PredictionValuesChange","LossFunctionChange","FeatureImportance","ShapValues",
                        "ShapInteractionValues","Interaction","PredictionDiff"]

        original_feature_name = self.model.feature_names_   # 原始的列名


        importan_feature = self.model.get_feature_importance(data=None, type=EFstrType.FeatureImportance,
                                                             prettified=False,thread_count=-1, verbose=False,
                                                             fstr_type=None, shap_mode="Auto",
                                                             interaction_indices=None, shap_calc_type="Regular")
        return importan_feature, original_feature_name

    # 保存模型
    def save_models(self, model_path):
        self.model.save_model(model_path)

    # 加载模型
    def load_models(self, name_path):
        return self.model.load_model(fname=name_path)

    #  绘制预测的结果
    def plot_predictions(self, data, features_to_change, plot=True, plot_file=None):
        """
        :param data: array, df, SparseDataFrame, spmatrix , catbool.pool 需要绘制的数据
        :param features_to_change:   整形列表   string ,  整形与字符串型的结合，   数值特征的列表，以改变预测值。
        :param plot: Bool型数据， 是否进行显示，
        :param plot_file:  字符串型  绘图的标题
        :return:
        """
        all_predictions, figs = self.model.plot_predictions(data=data, features_to_change=features_to_change,
                                                            plot=plot, plot_file=plot_file)
        return all_predictions, figs

    # 计算 特征统计
    def calc_feature_statistic(self,data,target=None, feature=None, prediction_type=None, cat_feature_values=None,
                               plot=True, max_cat_features_on_plot=10, thread_count=-1, plot_file=None):
        """
        :param data:   数组，  df .  SparseDataFrame， 等数据格式
        :param target:  numpy   Series
        :param feature:  字符串 int或他们组合的整数列表，
        :param prediction_type: string   ,
        :param cat_feature_values:  list， 数组， Series数据格式
        :param plot:  bool
        :param max_cat_features_on_plot:  int， 在一个图表上要输出的分类特征的不同值的最大数目。
        :param thread_count:  int ，    线程数
        :param plot_file:   绘制 要将图表保存到的输出文件的名称。
        :return:
        """

        result = self.model.calc_feature_statistics(data=data, target=target, feature=feature,
                                                    prediction_type=prediction_type,
                                                    cat_feature_values=cat_feature_values,
                                                    plot=plot, max_cat_features_on_plot=max_cat_features_on_plot,
                                                    thread_count=thread_count, plot_file=plot_file)

        return result

    # 评估指标
    def eval_mertrics_fn(self,data, metrics, ntree_start=0, ntree_end=0, eval_period=1,
                         thread_count=-1, tmp_dir=None, plot=False):
        """
        :param data:  catboost.Pool
        :param metrics: 字符串列表 ["logloss", "AUC"] 等等
        支持的评估类型名称为["RMSE","Logloss","MAE","CrossEntropy","Quantile","LogLinQuantile","Lq","MultiRMSE","MultiClass",
        "MultiClassOneVsAll","MAPE","Poisson","PairLogit","PairLogitPairwise","QueryRMSE","QuerySoftMax","Tweedie",
        "SMAPE","Recall","Precision","F1","TotalF1","Accuracy","BalancedAccuracy","BalancedErrorRate","Kappa","WKappa",
        "LogLikelihoodOfPrediction", "AUC","R2","FairLoss","NumErrors","MCC","BrierScore","HingeLoss","HammingLoss"
        "ZeroOneLoss","MSLE","MedianAbsoluteError","Huber","Expectile","MultiRMSE","PairAccuracy","AverageGain","PFound"
        "NDCG","DCG","FilteredDCG","NormalizedGini","PrecisionAt","RecallAt","MAP","CtrFactor",
        :param ntree_start:  默认为0
        :param ntree_end:  默认为0
        :param eval_period:  区间，假设 start=0,end=N,eval_period=2, 树的范围[0, 2), [0, 4), ... , [0, N)
        :param thread_count:  # 使用的进程数  默认为-1 yu 训练中保持一致
        :param tmp_dir:
        :param plot:
        :return:
        """
        self.model.eval_metrics(data=data,metrics=metrics,ntree_start=ntree_start,ntree_end=ntree_end,
                                eval_period=eval_period,thread_count=thread_count,tmp_dir=tmp_dir,plot=plot)

    #  进行网格搜索
    def grid_search_fn(self,param_grid, X, y=None, cv=3, partition_random_seed=0,
                    calc_cv_statistics=True, search_by_train_test_split=True,
                    refit=True, shuffle=True, stratified=None, train_size=0.8, verbose=True, plot=False):
        """
        :param param_grid:  搜索的使用的超级参数，  字典或字典列表
        :param X:   X的值   使用数组, DataFrame  或者使用catboost.Pool
        :param y:   y的值   使用数组, DataFrame  或者使用catboost.Pool
        :param cv:  cv 的折数 交叉验证的折数， 默认使用3折，
        :param partition_random_seed:   默认值为0  整数。 做为种子值
        :param calc_cv_statistics:  决定质量是否参与评估  当search_by_train_test_split为True时才起作用，用于交叉验证选取最好的参数
        :param search_by_train_test_split: True 开启训练和测试集
        :param refit: 使用最好的参数进行整个数据集的拟合
        :param shuffle: 是否开启打乱
        :param stratified:  # 支持分层抽样， 默认为None
        :param train_size:  默认0.8
        :param verbose:  默认为True 等价于 verbose=1. 否则没信息输出
        :param plot:   是否进行数据输出， 搜索参数是使用
        :return:
        """
        gride_search_result = self.model.grid_search(param_grid=param_grid, X=X, y=y, cv=cv,
                                                     partition_random_seed=partition_random_seed,
                                                     calc_cv_statistics=calc_cv_statistics,
                                                     search_by_train_test_split=search_by_train_test_split,
                                                     refit=refit, shuffle=shuffle,stratified=stratified,
                                                     train_size=train_size,verbose=verbose, plot=plot)
        return  gride_search_result

    # 将训练好的两个模型进行对比操作
    def compare(self, othermodel, data, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None):
        """
        :param othermodel:   其他训练好的模型
        :param data: Pool 类型的数据,
        :param metrics:  字符串类型的评估列表   支持类型与val_mertrics_fn 一致
        :param ntree_start:  开始的树  int
        :param ntree_end:     结束的树   int
        :param eval_period:   int,
        :param thread_count:   线程数
        :param tmp_dir:   string   中间结果的临时存放
        :return:
        """
        self.model.compare(model= othermodel, data=data, metrics=metrics, ntree_start=ntree_start,
                           ntree_end=ntree_end, eval_period=eval_period, thread_count=thread_count, tmp_dir=tmp_dir)

    # 检查模型是否被训练过
    def is_fitted(self):
        return self.model.is_fitted()

    def plot_tree(self, tree_idx, pool=None):
        """
        :param tree_idx:  模型中树的索引
        :param pool: 默认值 None
        :return:
        """
        self.model.plot_tree(tree_idx=tree_idx,pool=pool)

    def random_search(self,param_distributions, X, y=None, cv=3, n_iter=10, partition_random_seed=0,
                          calc_cv_statistics=True, search_by_train_test_split=True, refit=True,
                          shuffle=True, stratified=None, train_size=0.8, verbose=True, plot=False):
        """
        :param param_distributions:  字典，使用的超参数
        :param X: X的值   使用数组, DataFrame  或者使用catboost.Pool
        :param y: y的值   使用数组, DataFrame  或者使用catboost.Pool
        :param cv: cv 的折数 交叉验证的折数， 默认使用3折，
        :param n_iter:
        :param partition_random_seed: 默认值为0  整数。 做为种子值
        :param calc_cv_statistics: 决定质量是否参与评估  当search_by_train_test_split为True时才起作用，用于交叉验证选取最好的参数
        :param search_by_train_test_split: True 开启训练和测试集
        :param refit:使用最好的参数进行整个数据集的拟合
        :param shuffle: 是否开启打乱
        :param stratified:  # 支持分层抽样， 默认为None
        :param train_size: 默认0.8
        :param verbose:  默认为True 等价于 verbose=1. 否则没信息输出
        :param plot: 是否进行数据输出， 搜索参数是使用
        :return:
        """
        return self.model.randomized_search(param_distributions=param_distributions, X=X, y=y, cv=cv, n_iter=n_iter,
                                            partition_random_seed=partition_random_seed, calc_cv_statistics=calc_cv_statistics,
                                            search_by_train_test_split=search_by_train_test_split, refit=refit,
                                            shuffle=shuffle, stratified=stratified, train_size=train_size,
                                            verbose=verbose, plot=plot)

    # 使用特征名
    def set_feature_names(self,feature_names):
        """
        :param feature_names: 数组，或使用list
        :return:
        """
        self.model.set_feature_names(feature_names=feature_names)

    # 设置相关参数
    def set_params(self,**params):
        """
        :param params:
        :return:
        """
        self.model.set_params(**params)


    # def plot_importance(self):
    #     fig, ax = plt.subplots(figsize=(15, 15))   # 有问题的  对象
    #     plot_importance(self.model,
    #                     height=0.5,
    #                     ax=ax,
    #                     max_num_features=64)  # 最多绘制64个特征
    #     plt.show()  # 显示图片

    # # 重要性进行绘图操作
    def plot_importance(self,figure_path=None, ifsave=True):
        feat_, feat_name_ = self.get_feature_importance_()
        plt.figure(figsize=(15,15))
        plt.barh(feat_name_, feat_, height=0.5)
        if ifsave:
            if not figure_path:
                plt.savefig("../model/Catboost_model/catboost_featute_importance_before.png")
            else:
                plt.savefig(figure_path)
        plt.show()

    def _plot_importance_v1(self,columns_name, figure_path=None, ifsave=True):
        fig, ax = plt.subplots(figsize=(15, 15))
        plot_importance_v1(self.model,model_name="cb",columns_name=columns_name,
                           height=0.5,
                           ax=ax,
                           max_num_features=64)  # 最多绘制64个特征
        if ifsave:
            if not figure_path:
                plt.savefig("../model/Catboost_model/catboost_featute_importance_after.png")
            else:
                plt.savefig(figure_path)

        plt.show()  # 显示图片


# 使用池进行训练
def pool_format(data,label=None,cat_features=None,text_features=None,column_description=None,
                pairs=None, delimiter='\t',has_header=False,weight=None,group_id=None,group_weight=None,
                subgroup_id=None,  pairs_weight=None, baseline=None, feature_names=None,thread_count=-1):
    """
    :param data:  列表或数组， 或df,series, 或使用featuresDatam ，string等数据格式
    :param label:   训练集的label 值，列表， 数组， df, series， 为可选项， 默认为None， 如果不为空，给出1维或2维的浮点型数组
    :param cat_features:  种类特征，
    :param text_features: 文本特征
    :param column_description:
    :param pairs:
    :param delimiter:
    :param has_header:
    :param weight:
    :param group_id:
    :param group_weight:
    :param subgroup_id:
    :param pairs_weight:
    :param baseline:
    :param feature_names:
    :param thread_count:
    :return:
    """
    return Pool(data,label=label,cat_features=cat_features,text_features=text_features,
                column_description=column_description,pairs=pairs, delimiter='\t',has_header=has_header,
                weight=weight,group_id=group_id,group_weight=group_weight,subgroup_id=subgroup_id,
                pairs_weight=pairs_weight, baseline=baseline, feature_names=feature_names,thread_count=thread_count)

# 进行交叉验证操作， 可以单独使用， 也可以使用cross_vallid 来进行操作，     若果是选择最优模型类型, 使用 cross_valid， 如果
#
def cv_fn(pool=None, params=None, dtrain=None, iterations=None, num_boost_round=None,
       fold_count=None, nfold=None, inverted=False, partition_random_seed=0, seed=None,
       shuffle=True, logging_level=None, stratified=None, as_pandas=True, metric_period=None,
       verbose=None, verbose_eval=None, plot=False, early_stopping_rounds=None,
       save_snapshot=None, snapshot_file=None, snapshot_interval=None, folds=None, type='Classical'):
    """
    :param pool:  pool 类型的数据
    :param params:  参数字典
    :param dtrain: Pool 类型数组，或者 tuple(x,y)组成的元组
    :param iterations: int  默认值10000. 树的最大棵树
    :param num_boost_round: 与iteration 等价
    :param fold_count:  这是 默认为3折
    :param nfold: 别名， 也是fold_count 相同的值，
    :param inverted:  False. j间隔
    :param partition_random_seed: 别名seed,  0
    :param seed:
    :param shuffle: 是否打乱， shuffle, 默认为True
    :param logging_level:  日志格式，
    :param stratified:  执行分层采样， None
    :param as_pandas: True  返回的数据类型， 默认返回df格式的数据
    :param metric_period:1 计算目标和指标值的迭代频率
    :param verbose: 是否开启打印， 别名verbose_eval    False
    :param verbose_eval:
    :param plot: # 绘制相关效果图   False
    :param early_stopping_rounds:  False  是否开启过拟合操作
    :param save_snapshot: 保存快照
    :param snapshot_file: 保存快照的名字
    :param snapshot_interval: 快照的间隔
    :param folds: None
    :param type:
    :return:
    """
    scores=cv(pool=pool, params=params, dtrain=dtrain, iterations=iterations, num_boost_round=num_boost_round,
              fold_count=fold_count, nfold=nfold, inverted=inverted, partition_random_seed=partition_random_seed,
              seed=seed, shuffle=shuffle, logging_level=logging_level, stratified=stratified, as_pandas=as_pandas,
              metric_period=metric_period, verbose=verbose, verbose_eval=verbose_eval, plot=plot,
              early_stopping_rounds=early_stopping_rounds,save_snapshot=save_snapshot, snapshot_file=snapshot_file,
              snapshot_interval=snapshot_interval, folds=folds, type=type)
    return scores

# 从磁盘中直接加载 模型文件
def load_cb_model_from_disk(model_path,model_type="reg"):
    """
    :param model_path: 模型存放的路径
    :param model_type: 模型的类型
    :return:
    """
    assert model_type in ["reg","class"]
    if model_type =="reg":
        model = cb.CatBoostRegressor(logging_level="silent")
    else:
        model = cb.CatBoostClassifier(logging_level="silent")
    return model.load_model(fname=model_path)
