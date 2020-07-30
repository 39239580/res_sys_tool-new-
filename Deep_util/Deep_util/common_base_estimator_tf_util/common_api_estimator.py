# import tensorflow as tf
# tf.estimator.Estimator 构建 tensorflow 模型

#  tensorflow最底层  tensorflow kernel   tensorflow distributed execution engine
# low-level-api    python  java c++ go
# mid-level api   layers Datasets  Metrics
# High-level api   Estimators
# Estimator 封装了操作： 训练、评估、 预测、导出以共使用

# ---------- 数据集------------
# 通过tf.data 模块， 构建输入管道，讲数据传送到模型中， tf.data模块返回的是 DataSet 对象， 每个Dataset 对象包含feature_dict, labels）对。

#   -------------------定义特征列----------------------------
# 特征列作为原始数据和Estimator 之间的媒介。 要创建特征列，需要调用tf.feature_column模块的函数
# tf.feature_column   主要分成两个部分
# 其中
# tf.feature_column 主要有两个主类别，和一个混合类别
# --------------------种类列categorical column--------------------
# tf.feature_column.categorical_column_with_identity 分类识别列，将每个分桶表示一个唯一的整数， 模型可以在分类表示列中学习每个类别各自的权重
# tf.feature_column.categorical_column_with_vocabulary_file 分类词汇列， 讲字符串表示为独热矢量，根据文件中的词汇将每个字符串映射到一个整数
# tf.feature_column.categorical_column_with_vocabulary_list  分类词汇列，将字符串表示为独热矢量， 根据明确词汇表将每个字符串映射到一个整数。
# tf.feature_column.categorical_column_with_hash_bucket  经过hash处理的列 讲类别数量非常大的特征，模型会计算输入的哈希值，
# 然后使用模运算符将其置于其中一个hash_bucket 类别中
# tf.feature_column.crossed_column # 特征组合列
# --------------------Dense Columns-----------------------------
# tf.feature_column.numeric_column  数值列 将默认数据类型tf.float32的数值制定为模型输入
# tf.feature_column.indicator_column   指标列
# tf.feature_column.embedding_column  嵌入列， 讲分类列视为输入
# --------------------------混合类别-----------------------------
# tf.feature_column.bucketized_column 分桶列，将数字列根据数值范围分为不同的类别（为模型中加入非线性特征， 提高模型的表达能力）

# ----------------------------estimator 创建模型 ----------------------------
# 预创建的Estimator 是tf.estimator.Estimator基类的子类，自定义的Estimator 是 tf.estimator.Estimator 的实例。两者的区别在于
# 预创建的Estimator 已经有模型函数，而自定义的Estimator 需要自己编写模型函数
# 预创建的Estimators 有 分类操作有DNNClassifier  LinearClassifier  DNNLinearCombinedClassifier
# 预创建的Estimators 有 回归操作有DNNRegressor  LinearRegressor  DNNLinearCombinedRegressor

# -----------------------预创建的Estimators----------------------------------------
# 提供了三个预创建的分类器Estimator(Estimator代表一个完整的模型)
# tf.estimator.DNNClassifier   多类别分类的深度模型
# tf.estimator.LinearClassifier   基于线性模型的分类器
# tf.estimator.DNNLinearCombinedClassifier  宽度和深度模型

# -----------------------------------自定义的estimator------------------------------
# 定义模型函数， 模型参数具有一下参数
# def my_model_fn(features, labels, mode, params，config):
# features, labels 是从输入函数中返回的特征和标签批次
# model 表示调用程序是请求新联、预测还是评估，  tf.estimator.ModeKeys
# params 是调用程序将params传递给Estimator的构造函数，转而传递给model_fn
# classifier = tf.estimator.Estimator(model_fn=my_model,params={"feature_columns":my_feature_columns,
#                                                               "hidden_units":[10,10],  两层均为10个节点的隐含层
#                                                               "n_classes":3,   # 种类的类别为3
#                                                               })
# 模型-输入层： 将特征字典和feature_columns转换为模型的输入
# 模型隐含层：  tf.layers 提供所有类型的隐藏层， 包括卷积、 池化层、 丢弃层
# 模型输出层： tf.layers.dense 定义输出层。  使用 tf.nn.softmax 将分数转换成概率

# -----------------------------------------config--------------------------------------
# 源代码
# class RunConfig(object):
#   """This class specifies the configurations for an `Estimator` run."""
#
#   def __init__(self,
#                model_dir=None,
#                tf_random_seed=None,
#                save_summary_steps=100,
#                save_checkpoints_steps=_USE_DEFAULT,
#                save_checkpoints_secs=_USE_DEFAULT,
#                session_config=None,
#                keep_checkpoint_max=5,
#                keep_checkpoint_every_n_hours=10000,
#                log_step_count_steps=100,
#                train_distribute=None,
#                device_fn=None,
#                protocol=None,
#                eval_distribute=None,
#                experimental_distribute=None,
#                experimental_max_worker_delay_secs=None,
#                session_creation_timeout_secs=7200):
#
#     model_dir: 指定存储模型参数，graph等的路径
#     save_summary_steps: 每隔多少step就存一次Summaries，不知道summary是啥
#     save_checkpoints_steps:每隔多少个step就存一次checkpoint
#     save_checkpoints_secs: 每隔多少秒就存一次checkpoint，不可以和save_checkpoints_steps同时指定。如果二者都不指定，则使用默认值，即每600秒存一次。如果二者都设置为None，则不存checkpoints。
#
#     注意上面三个**save-**参数会控制保存checkpoints（模型结构和参数）和event文件（用于tensorboard），如果你都不想保存，那么你需要将这三个参数都置为FALSE
#
#     keep_checkpoint_max：指定最多保留多少个checkpoints，也就是说当超出指定数量后会将旧的checkpoint删除。当设置为None或0时，则保留所有checkpoints。
#     keep_checkpoint_every_n_hours：
#     log_step_count_steps:该参数的作用是,(相对于总的step数而言)指定每隔多少step就记录一次训练过程中loss的值，同时也会记录global steps/s，通过这个也可以得到模型训练的速度快慢。（天啦，终于找到这个参数了。。。。之前用TPU测模型速度，每次都得等好久才输出一次global steps/s的数据。。。蓝瘦香菇）
#
#   后面这些参数与分布式有关，以后有时间再慢慢了解。
#
#     train_distribute
#     device_fn
#     protocol
#     eval_distribute
#     experimental_distribute
#     experimental_max_worker_delay_secs


# -------------------------模型训练、评估和预测---------------------------
#     Estimator方法             Esimator模式
#     train()                  ModeKeys.TRAIN
#     evaluate()               ModeKeys.EVAL
#     predict()                ModeKeys.PREDICT

# ----------------------------------模型训练----------------------------------------
# classifier.train(
#     input_fn = lambda:irir_data.train_input_fn(train_x, train_y, args.batch_size),
#     max_steps = args.train_steps
# )
# Estimators   会调用函数模型并将model设置为ModeKeys.TRAIN
# input_fn: 输入数据， 将input_fn调用封装在lambda 中 以获取参数， 提供一个不采用任何参数的输入函数
# max_steps:模型训练的最多步数
# 在my_model_fn 中，定义损失函数和优化损失函数的方法
# 计算损失    train 个 eval 模型中
#     loss = tf.losses.sparse_softmax_corss_entropy(labels=labels, logits=logits)  # 多类别分类
#     配置训练玄幻， 用在训练模型中
#     采用随机梯度下降法优化损失函数，学习速率为0.001
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#         train_op = optimizer.minimize(
#             loss=loss,
#             global_step=tf.train.get_global_step()
#         )
#         return tf.estimator.EstimatorsSpec(mode=mode, loss=loss, train_op=train_op)

# -------------------------------------模型评估----------------------------------
# eval_result = classifier.evalute(
#     input_fn = lambda: iris_data.eval_input_fn(test_x, test_y, args.batch_size)
# )
# print("\nTest set accuary: {accuracy:0.3f}\n".format(**eval_result))

# Estimator 会调用模型函数并讲mode设置为ModeKeys.EVAL. 模型函数必须返回一个包含模型损失 和一个或多个指标(可选)的
# tf.estimator.EstimatorSpec
# 使用 tf.estrics 计算常用指标
# 添加评估指标 for EVAL mode  准确率指标
# eval_metric_ops ={"accuracy": tf.metrics.accuary(labels=labels, predictions=predictions["classes"])}
# return tf.esitimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# ------------------------------------模型预测-------------------------------------
# predictions = classifier.predict(input_fn=lambda :
                                # iris_data.eval_input_fn(predict_x, labels=None, batch_size=arg.batch_size)

# 调用Estimator的predict方法， 则model_fn 会收到mode=ModeKeys.PREDICT，模型函数返回一个包含预测的tf.estimator.EstimatorSpec
#
# predictions = {
#     # 生成预测， 预测和评估模型
#     "classes": tf.argmax(input = logits, axis=1),
#     # 添加softmax_tensor  到计算图，被用于预测，
#     # 对数似然
#     "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
#     # 生成特征向量图片
# }
# if mode == tf.estimator.ModeKeys.PREDICT:
#     return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

# predictions 存储的是三个键值对：
# classes: 存储的是模型对此样本预测的最有可能的类别id;
# probabilities: 存储的是样本属于各个类别的概率值
# features: 存储的是样本的特征向量(倒数第二层)每10分钟(600)

# -------------------------------------模型保存和恢复-------------------------------------------
# Estimator 自动将模型信息写入磁盘： 检查点，训练期间所创建的模型版本， 事件文件，包含TensorBoard 用于创建可视化图表的信息。
# 在Estimator 的构造函数model_dir参数中定义模型保存路径
# 模型保存：
# Data Files---->input Function----> Estimator ----> train()  ----> Checkpoint
#                                              ----> evalute()
#                                              ----> predict()
# 第一次调用train。   第一次调用，将会检查点和事件文件添加到model_dir目录中。
# 默认情况下，Estimator 将按照一下时间安排讲检查点保存到model_dir 中: 每10分钟（600秒）写入一个检查点；
# 在train方法开始(第一次迭代)和完成(最后一次迭代)时，写入一个检查点，在目录中保留5个最近写入的检查点。
# 通过 tf.estimator.RunConfig   对默认保存事件更改：
# my_checkpointing_config = tf.estimator.RunConfig(
#     save_checjpoints_secs = 20*60,
#     keep_checkpoints_max = 10,
# )
#
# classifier = tf.estimator.DNNClassifier(
#     feature_columns = my_feature_columns,
#     hidden_units = [10,10],
#     n_classes=3,
#     model_dir ="models/iris",
#     config = my_checkpointing_config)
# ------------------------模型恢复----------------------------
#                                              train     <------
# Data Files--->inputFunction --->Estimator    evaluate  <------checkpoint
                                             # predict   <------

# 第一次调用estimator的train方法时，将第一个检查点保存到model_dir中， 随后每次调用Estimator 的train, evaluate
# 或predict方法是，都会：Estimator 运行model_fn构建模型图
# Estimator 根据最近写入的检查点中存储的数据来初始化新模型的权重
# 通过检查点恢复模型的状态仅在模型和检查点兼容时可行。 假设训练好的为2个隐藏层为10个节点的网络，从10改成 20时， 模型会不兼容并报错
import tensorflow as tf
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops import nn
# from tensorflow.python.ops import init_ops


class CommonEstimator(object):
    def __init__(self, model_type, task_type, params):
        self.model_type = model_type
        self.task_type = task_type
        self.params = params
        self.create_estimator()  # 自带的相关estimator

    def create_estimator(self):
        # 系统自带的estimator
        """
        optimizer   可使用的参数为  'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'
        """
        if self.task_type in {"class", "classifier", "CLASS", "cla", None}:
            if self.model_type in {"DNN", "dnn", "deep", "DEEP", "Deep"}:
                self.model = tf.estimator.DNNClassifier(
                    hidden_units=self.params["hidden_units"],
                    feature_columns=self.params["feature_columns"],
                    model_dir=self.params.get("model_dir", None),
                    n_classes=self.params.get("n_classes", 2),
                    weight_column=self.params.get("weight_column", None),
                    label_vocabulary=self.params.get("label_vocabulary", None),
                    optimizer=self.params.get("optimizer", 'Adagrad'),
                    activation_fn=self.params.get("activation_fn", nn.relu),
                    dropout=self.params.get("dropout", None),
                    # input_layer_partitioner=self.params.get("input_layer_partitioner",None),
                    config=self.params.get("config", None),
                    warm_start_from=self.params.get("warm_start_from", None),
                    loss_reduction=self.params.get("loss_reduction", losses.Reduction.SUM),
                    batch_norm=self.params.get("batch_norm", False)
                                            )

            elif self.model_type in {"wide_deep", "wide and deep", "Wide_Deep", "wide&deep"}:
                self.model = tf.estimator.DNNLinearCombinedClassifier(
                    model_dir=self.params.get("model_dir", None),
                    linear_feature_columns=self.params.get("linear_feature_columns", None),
                    linear_optimizer=self.params.get("linear_optimizer", 'Ftrl'),
                    dnn_feature_columns=self.params.get("dnn_feature_columns", None),
                    dnn_optimizer=self.params.get("dnn_optimizer", 'Adagrad'),
                    dnn_hidden_units=self.params.get("dnn_hidden_units", None),
                    dnn_activation_fn=self.params.get("dnn_activation_fn", nn.relu),
                    dnn_dropout=self.params.get("dnn_dropout", None),
                    n_classes=self.params.get("n_classes", 2),
                    weight_column=self.params.get("weight_column", None),
                    label_vocabulary=self.params.get("label_vocabulary", None),
                    # input_layer_partitioner=self.params.get("input_layer_partitioner",None),
                    config=self.params.get("config", None),
                    warm_start_from=self.params.get("warm_start_from", None),
                    loss_reduction=self.params.get("loss_reduction", losses.Reduction.SUM),
                    batch_norm=self.params.get("batch_norm", False),
                    linear_sparse_combiner=self.params.get("linear_sparse_combiner", 'sum')
                )

            elif self.model_type in {"wide", "Wide", "WIDE"}:
                self.model = tf.estimator.LinearClassifier(
                    feature_columns=self.params["feature_columns"],
                    model_dir=self.params.get("model_dir", None),
                    n_classes=self.params.get("n_classes", 2),
                    weight_column=self.params.get("weight_column", None),
                    label_vocabulary=self.params.get("label_vocabulary", None),
                    optimizer=self.params.get("optimizer", 'Ftrl'),
                    config=self.params.get("config", None),
                    # partitioner=self.params.get("partitioner",None),
                    warm_start_from=self.params.get("warm_start_from", None),
                    loss_reduction=self.params.get("loss_reduction", losses.Reduction.SUM),
                    sparse_combiner=self.params.get("sparse_combiner", 'sum')
                )

            else:
                raise ValueError("模型类型设定有误")

        elif self.task_type in {"regression", "ReGression", "Regre", "regre"}:
            if self.model_type in {"DNN", "dnn", "deep", "DEEP", "Deep"}:
                self.model = tf.estimator.DNNRegressor(
                    hidden_units=self.params["hidden_units"],
                    feature_columns=self.params["feature_columns"],
                    model_dir=self.params.get("model_dir", None),
                    label_dimension=self.params.get("label_dimension", 1),
                    weight_column=self.params.get("weight_column", None),
                    optimizer=self.params.get("optimizer", 'Adagrad'),
                    activation_fn=self.params.get("activation_fn", nn.relu),
                    dropout=self.params.get("dropout", None),
                    # input_layer_partitioner=self.params.get("input_layer_partitioner",None),
                    config=self.params.get("config", None),
                    warm_start_from=self.params.get("warm_start_from", None),
                    loss_reduction=self.params.get("loss_reduction", losses.Reduction.SUM),
                    batch_norm=self.params.get("batch_norm", False)
                )

            elif self.model_type in {"wide_deep", "wide and deep", "Wide_Deep", "wide&deep"}:
                self.model = tf.estimator.DNNLinearCombinedRegressor(
                    model_dir=self.params.get("model_dir", None),
                    linear_feature_columns=self.params.get("linear_feature_columns", None),
                    linear_optimizer=self.params.get("linear_optimizer", 'Ftrl'),
                    dnn_feature_columns=self.params.get("dnn_feature_columns", None),
                    dnn_optimizer=self.params.get("dnn_optimizer", 'Adagrad'),
                    dnn_hidden_units=self.params.get("dnn_hidden_units", None),
                    dnn_activation_fn=self.params.get("dnn_activation_fn", nn.relu),
                    dnn_dropout=self.params.get("dnn_dropout", None),
                    label_dimension=self.params.get("label_dimension", 1),
                    weight_column=self.params.get("weight_column", None),
                    # input_layer_partitioner=self.params.get("input_layer_partitioner",None),
                    config=self.params.get("config", None),
                    warm_start_from=self.params.get("warm_start_from", None),
                    loss_reduction=self.params.get("loss_reduction", losses.Reduction.SUM),
                    batch_norm=self.params.get("batch_norm", False),
                    linear_sparse_combiner=self.params.get("linear_sparse_combiner", 'sum')
                )

            elif self.model_type in {"wide", "Wide", "WIDE"}:
                self.model = tf.estimator.LinearRegressor(
                    feature_columns=self.params["feature_columns"],
                    model_dir=self.params.get("model_dir", None),
                    label_dimension=self.params.get("label_dimension", 1),
                    weight_column=self.params.get("weight_column", None),
                    optimizer=self.params.optimizer.get("optimizer", 'Ftrl'),
                    config=self.params.optimizer.get("config", None),
                    # partitioner=self.params.optimizer.get("partitioner",None),
                    warm_start_from=self.params.get("warm_start_from", None),
                    loss_reduction=self.params.get("loss_reduction", losses.Reduction.SUM),
                    sparse_combiner=self.params.get("sparse_combiner", 'sum')
                )

            else:
                raise ValueError("模型类型设定有误")

        else:
            raise  ValueError("任务类型设定有误差")

    def fit(self, input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None):
        self.model.train(input_fn=input_fn,
                         hooks=hooks,
                         steps=steps,
                         max_steps=max_steps,
                         saving_listeners=saving_listeners)

    def predict(self, input_fn, predict_keys=None, hooks=None, checkpoint_path=None, yield_single_examples=True):
        return self.model.predict(input_fn=input_fn,
                                  predict_keys=predict_keys,
                                  hooks=hooks,
                                  checkpoint_path=checkpoint_path,
                                  yield_single_examples=yield_single_examples)

    def evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None, name=None):
        self.model.evaluate(input_fn=input_fn,
                            steps=steps,
                            hooks=hooks,
                            checkpoint_path=checkpoint_path,
                            name=name)


def custom_runconfig(config_params):   # 自定义相关配置函数
    _USE_DEFAULT = object()
    config = tf.compat.v1.estimator.RunConfig(
        model_dir=config_params.get("model_dir", None),  # 模型保存路径
        tf_random_seed=config_params.get("tf_random_seed", None),  # 随机种子
        save_summary_steps=config_params.get("save_summary_steps",100),  #　保存摘要间隔
        save_checkpoints_steps=config_params.get("save_checkpoints_steps", _USE_DEFAULT),  #　每隔多少步保存检查点，
        save_checkpoints_secs=config_params.get("save_checkpoints_secs", _USE_DEFAULT), # 每隔多少秒保存检查点
        # 若save_checkpoints_steps和save_checkpoints_secs均未设置，则默认600秒，　若均设置为None，则禁用此功能, 两个设置均不能同时设置
        session_config=config_params.get("session_config", None),  # 用于设置ConfigProto  或None
        keep_checkpoint_max=config_params.get("keep_checkpoint_max", 5),  # 默认保存的模型个数
        keep_checkpoint_every_n_hours=config_params.get("keep_checkpoint_every_n_hours", 10000),  # 10000禁用此功能,每隔n小时进行保存
        log_step_count_steps=config_params.get("log_step_count_steps",100),  # 每多少步数输出日志
        train_distribute=config_params.get("train_distribute",None), # 可使用　tf.distribute.Strategy,
        # 指定后，　Estimator会根据策略在评估过程中分发用户的模型，首选experimental_distribute.train_distribute
        device_fn=config_params.get("device_fn", None), # 如果为None，则默认为tf.train.replica_device_setter,
        # None　使用默认的设备。，　非None，　设置的设备会覆盖estimator
        protocol=config_params.get("protocol", None), # None, grpc　　启动服务器的协议。
        eval_distribute=config_params.get("eval_distribute", None),
        experimental_distribute=config_params.get("experimental_distribute", None),  #
        # 一个可选 tf.contrib.distribute.DistributeConfig对象，用于指定与DistributionStrategy相关的配置。该train_distribute和
        # eval_distribute可以作为参数传递RunConfig或设置 experimental_distribute，但不能同时使用。
        experimental_max_worker_delay_secs=config_params.get("experimental_max_worker_delay_secs",None), # 一个可选的整数，
        # 指定worker在启动之前应等待的最长时间, 默认下最多延迟60秒
        session_creation_timeout_secs=config_params.get("session_creation_timeout_secs", 7200)
        # worker应等待的最大时间（通过初始化或恢复会话时）可通过MonitoredTrainingSession使用。
        # 默认值为7200秒，但用户可能希望设置一个较低的值，以更快地检测变量 / 会话（重新）初始化的问题。
    )
    return config

