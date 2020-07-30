import tensorflow as tf
from tensorflow.python.ops import init_ops

# -------------------------------demo的文档地址---------------------------------------------
# https://blog.csdn.net/amao1998/article/details/80202777
# def data_process():
#     ds = tf.data.TextLineDataset('../IRIS/iris_training.csv').skip(1)
#     # 指定csv文件的列名
#     COLUMNS = ['SepalLength', 'SepalWidth',
#                'PetalLength', 'PetalWidth',
#                'label']
#     # 指定csv各个字段的默认值
#     FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]
#     def _parse_line(line):
#         # 将每一行解析到对应的字段当中
#         fields = tf.io.decode_csv(line, FIELD_DEFAULTS)
#         # 将字段值与列名组成一个字典
#         features = dict(zip(COLUMNS,fields))
#         #将特征值与标记列分开
#         label = features.pop('label')
#         return features, label
#     ds = ds.map(_parse_line)
#     def train_func(ds):
#         dataset = ds.shuffle(1000).repeat().batch(100)
#         return dataset.make_one_shot_iterator().get_next()
#     print(train_func(ds))
#
#     feature_columns = [
#         tf.feature_column.numeric_column(name)
#         for name in COLUMNS[:-1]]
#     print(feature_columns)
#     return feature_columns

# import os
import pandas as pd

def data_process():
    FUTURES = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Species']
    SPECIES = ['Setosa', 'Versicolor', 'Virginica']


    train = pd.read_csv("./IRIS/iris_training.csv", names=FUTURES, header=0)
    train_x, train_y = train, train.pop('Species')

    test = pd.read_csv("./IRIS/iris_training.csv", names=FUTURES, header=0)
    test_x, test_y = test, test.pop('Species')
    feature_columns = []
    for key in train_x.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))
    return FUTURES, SPECIES, train_x, train_y, test_x, test_y, feature_columns

#针对训练的喂食函数
def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size) #每次随机调整数据顺序
    return dataset.make_one_shot_iterator().get_next()

#针对测试的喂食函数
def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    inputs=(features,labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
#    return dataset
    return dataset.make_one_shot_iterator().get_next()





# -----------------------------------自定义estimator  demo------------------------------------------------
def custom_estimator(model_fn, model_dir, model_config, params):
    #  用户自定义
    classifier = tf.estimator.Estimator(
        model_fn= model_fn,
        model_dir=model_dir,
        config=model_config,
        params= params,
        warm_start_from=None)
    return classifier

# 自定estimator
def custom_model(features, labels, mode, params):
    # ----------------------------网络结构----------------------------------
    # 输入层  暂时只支持 DNN  操作， 后续，根据自己需要进行网络结构修改
    net = tf.feature_column.input_layer(features=features,
                                        feature_columns=params["feature_columns"]
    )
    # 中间层
    for unit in params["hidden_units"]:
        net = tf.layers.dense(net, unit, activation=tf.nn.relu)

    # 定义输出层
    logits  = tf.layers.dense(inputs= net,
                              units=params["n_classes"],
                              activation=None,
                              )
    # 判断是否是预测
    predicted_classes = tf.argmax(logits, 1)
    #  后续的训练，测试与预测

    if mode == tf.estimator.ModeKeys.PREDICT:    # 要先放在最前面
        predictions = {
            "class_ids" : predicted_classes[:, tf.newaxis],
            "probabilities": tf.nn.softmax(logits),
            "logits": logits
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # 计算代价：
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # 计算精确度
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name="acc_op")
    metrics = {"accuracy": accuracy}

    # 判断是否是评估
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=metrics
        )

    # 判断是否是训练模式
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

# 定义训练的相关配置参数
def run_config_fn(params):
    return tf.estimator.RunConfig(keep_checkpoint_max=params.get("keep_checkpoint_max",5), # 保存的最多模型个数
                                  log_step_count_steps=params.get("log_step_count_steps",50)  # 训练50轮进行打印
                                  )

def train(model,train_x, train_y , batch_size, steps):
    model.train(input_fn = lambda : train_input_fn(train_x, train_y,batch_size), steps=steps)

def eval(model, test_x, test_y, batch_size):
    eval_result = model.predict(input_fn = lambda : eval_input_fn(test_x, test_y, batch_size))
    print(eval_result)
    return eval_result
def predict(model,epoches, batch_size,SPECIES):
    for i in range(epoches):
        print('\nPlease enter features: SepalLength,SepalWidth,PetalLength,PetalWidth')
        a, b, c, d = map(float, input().split(','))  # 捕获用户输入的数字
        predict_x = {
            'SepalLength': [a],
            'SepalWidth': [b],
            'PetalLength': [c],
            'PetalWidth': [d],
        }
        # 进行预测
        predictions = model.predict(
            input_fn=lambda: eval_input_fn(predict_x,
                                           labels=[0, ],
                                           batch_size=batch_size))

        # 预测结果是数组，尽管实际我们只有一个
        for pred_dict in predictions:
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]
            print(SPECIES[class_id], 100 * probability)


def run_app():
    # 定义超参数
    # model_params = {"learning_rate":0.001}
    config_param = {}
    run_config = run_config_fn(config_param)
    FUTURES, SPECIES, train_x, train_y, test_x, test_y, feature_columns = data_process()
    batch_size = 100

    # 自定义的模型
    model_dir = "./IRIS/iris.model"
    estimator_params ={'feature_columns': feature_columns,
                       'hidden_units': [10, 10],
                       'n_classes': 3}
    classifier = custom_estimator(model_fn=custom_model, model_dir=model_dir, model_config=run_config, params=estimator_params)
    train(model=classifier, train_x=train_x, train_y=train_y, batch_size=batch_size, steps=2000)
    eval(model=classifier,test_x=test_x, test_y=test_y, batch_size=batch_size)
    # predict(model=classifier, epoches=100, batch_size=batch_size,SPECIES=SPECIES)


if __name__ == "__main__":
    run_app()
