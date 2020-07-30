import tensorflow as tf
from Deep_util.estimator_deep_util.XDeepFM_estimator import XDeepFMEstimator
from tensorflow import estimator
import time


FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer("embedding_size", 16, "Emebedding_size")  # 变量名称，默认值，用法描述
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.001, "Learing_rate")
tf.compat.v1.flags.DEFINE_float("dropout", 0.5, "Dropout rate")
tf.compat.v1.flags.DEFINE_string("task_type", "train", "Task type in {train, infer, eval, export}")
tf.compat.v1.flags.DEFINE_integer("num_epochs", 5, "Number of epochs")
tf.compat.v1.flags.DEFINE_string("deep_layers", '100,100', "deep layers")
tf.compat.v1.flags.DEFINE_string("cross_layers", "20,10,10", "cross layers")

tf.compat.v1.flags.DEFINE_string("train_path", "./criteo_data/train/", "Data path")
tf.compat.v1.flags.DEFINE_integer("train_parts", 150, "Tfrecord counts")
tf.compat.v1.flags.DEFINE_integer("eval_parts", 10, "Eval tfrecord")

tf.compat.v1.flags.DEFINE_string("test_path", "./criteo_data/test/", "Test path")
tf.compat.v1.flags.DEFINE_integer("test_parts", 15, "Tfrecord counts")

tf.compat.v1.flags.DEFINE_string("export_path", "./export/", "Model export path")
tf.compat.v1.flags.DEFINE_integer("batch_size", 256, "Number of batch_size")
tf.compat.v1.flags.DEFINE_integer("log_steps", 50, "Log_step_count_steps")
tf.compat.v1.flags.DEFINE_integer("eval_steps", 200, "Eval_steps")

tf.compat.v1.flags.DEFINE_integer("save_checkpoints_steps", 2000, "save_checkpoints_steps")
tf.compat.v1.flags.DEFINE_boolean("mirror", True, "Mirrored Strategy")

con_feature = ["_c{0}".format(i) for i in range(0, 14)]  # 连续特征
cat_feature = ["_c{0}".format(i) for i in range(14, 40)]  # 种类特征

# 对于连续型特征　不设定默认值，　使用均值进行填充
feature_description = {k: tf.compat.v1.FixedLenFeature(dtype=tf.compat.v1.float32, shape=1) for k in con_feature}
feature_description.update({k: tf.compat.v1.FixedLenFeature(dtype=tf.compat.v1.string, shape=1, default_value="NULL")
                            for k in cat_feature})


def build_feature_columns(embedding_size):
    con_feature.remove("_c0")
    linear_feature_columns = []
    embeddsing_feature_columns = []
    c1 = [0.0, 1.0, 2.0, 3.0, 5.0, 12.0]
    c2 = [0.0, 1.0, 2.0, 4.0, 10.0, 28.0, 76.0, 301.0]
    c3 = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 16.0, 24.0, 54.0]
    c4 = [1.0, 2.0, 3.0, 5.0, 6.0, 9.0, 13.0, 20.0]
    c5 = [20.0, 155.0, 1087.0, 1612.0, 2936.0, 5064.0, 8622.0, 16966.0, 39157.0]
    c6 = [3.0, 7.0, 13.0, 24.0, 36.0, 53.0, 85.0, 154.0, 411.0]
    c7 = [0.0, 1.0, 2.0, 4.0, 6.0, 10.0, 17.0, 43.0]
    c8 = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 17.0, 25.0, 37.0]
    c9 = [4.0, 8.0, 16.0, 28.0, 41.0, 63.0, 109.0, 147.0, 321.0]
    c10 = [0.0, 1.0, 2.0]
    c11 = [0.0, 1.0, 2.0, 3.0, 4.0, 8.0]
    c12 = [0.0, 1.0, 2.0]
    c13 = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 22.0]
    buckets_cont = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13]  # 桶
    # buckets_cat = [1460, 583, 10131226, 2202607, 305, 23, 12517, 633, 3, 93145, 5683, 8351592, 3194, 27, 14992, 5461305,
    #                10, 5652, 2172, 3, 7046546, 17, 15, 286180, 104, 142571]

    buckets_cat = [1460, 583, 100000, 100000, 305, 23, 12517, 633, 3, 93145, 5683, 100000, 3194, 27, 14992, 100000,
                   10, 5652, 2172, 3, 100000, 17, 15, 100000, 104, 100000]

    for i, j in zip(con_feature, buckets_cont):
        f_num = tf.compat.v1.feature_column.numeric_column(i, normalizer_fn=lambda x: tf.compat.v1.log(x+1.0))
        if i == "_c2":
            f_num = tf.compat.v1.feature_column.numeric_column(i, normalizer_fn=lambda x: tf.compat.v1.log(x+4.0))

        f_bucket = tf.compat.v1.feature_column.bucketized_column(f_num. j)
        f_embedding = tf.compat.v1.feature_column.embedding_column(f_bucket, embedding_size)

        linear_feature_columns.append(f_num)
        embeddsing_feature_columns.append(f_embedding)
    for i, j in zip(cat_feature, buckets_cat):
        f_cat = tf.compat.v1.feature_column.categorical_column_with_hash_bucket(key=i, hash_bucket_size=i)
        f_ind = tf.compat.v1.feature_column.indicator_column(f_cat)
        f_embedding = tf.compat.v1.feature_column.embedding_column(f_cat, embedding_size)

        linear_feature_columns.append(f_ind)
        embeddsing_feature_columns.append(f_embedding)
    return linear_feature_columns, embeddsing_feature_columns


def _parse_examples(serial_exmp):
    features = tf.compat.v1.parse_single_example(serial_exmp, features=feature_description)
    labels = features.pop("_c0")
    return features, labels


def default_params(distribute_strategy, linear_feature_columns, embedding_feature_columns):
    config = estimator.RunConfig(
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=5,
        log_step_count_steps=FLAGS.log_steps,
        save_summary_steps=200,
        train_distribute=distribute_strategy,
        eval_distribute=distribute_strategy
    )
    model_params = {
        "linear_feature_columns": linear_feature_columns,
        "embedding_feature_columns": embedding_feature_columns,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "dropout": FLAGS.dropout,
        "deep_layers": FLAGS.deep_layers,
        "cross_layers": FLAGS.cross_layers
    }
    return config, model_params


def input_fn(filename, batch_size, num_epoches=-1, need_shuffle=False):
    """
    :param filename:
    :param batch_size:
    :param num_epoches:
    :param need_shuffle:
    :return:  函数构造的数据为一下的一种
    输出是一个对象， tf.data.Dataset对象，输出为一个元组 （feature, labels）
    （feature, labels）元组， feature 是张量或字符串字典， label也是
    """
    dataset = tf.compat.v1.data.TFRecordDataset(filenames=filename)   # 制作自己的数对象
    dataset = dataset.map(_parse_examples, num_parallel_calls=4).batch(batch_size=batch_size)
    if need_shuffle:
        dataset = dataset.shuffle(buff_size=100)
    dataset = dataset.prefetch(buffer_size=100).repeat(num_epoches)
    return dataset


def main():
    data_dir = FLAGS.train_path
    data_files = []
    for i in range(FLAGS.train_parts):
        data_files.append(data_dir + "part-r{:0>5}".format(i))
    
    train_files = data_files[:-FLAGS.eval_parts]  # 既可用于　训练集
    eval_file = data_files[-FLAGS.eval_parts:]  # 可以用于　测试集
    
    test_files = []
    for i in range(FLAGS.test_parts):
        test_files.append(FLAGS.test_path + "part-r-{:0>5}".format(i))
    
    linear_feature_columns, embedding_feature_columns = build_feature_columns(FLAGS.embedding_size)
    distribute_strategy = None
    if FLAGS.mirror:
        distribute_strategy = tf.compat.v1.distribute.MirroredStrategy()   # 
    
    config, model_params = default_params(distribute_strategy=distribute_strategy,
                                          linear_feature_columns=linear_feature_columns,
                                          embedding_feature_columns=embedding_feature_columns)
    
    model_dir = "./models/"
    XDeepFM = XDeepFMEstimator(config=config, model_params=model_params, model_dir=model_dir)
    if FLAGS.task_type == "train":
        start = time.time()
        XDeepFM.train(filename=train_files, batch_size=FLAGS.batch_size, input_fn=input_fn,
                      num_epoches=FLAGS.num_epoches, need_shuffle=True,
                      hooks=None, steps=None, max_steps=None, saving_listeners=None)

        XDeepFM.evaluate(filename=eval_file, batch_size=FLAGS.batch_size, input_fn=input_fn,
                         num_epoches=-1, need_shuffle=False, steps=FLAGS.eval_steps)

        end = time.time()
        tf.compat.v1.logging.info("Training times :%.2f seconds"%(end-start))

    elif FLAGS.task_type == "eval":
        XDeepFM.evaluate(filename=eval_file, batch_size=FLAGS.batch_size, input_fn=input_fn,
                         num_epoches=1, steps=FLAGS.eval_steps*10)
    elif FLAGS.task_type == "predict":
        p = XDeepFM.predict(filename=eval_file, batch_size=FLAGS.batch_size, input_fn=input_fn,
                            num_epoches=1
                            )
        tf.compat.v1.logging.info("done predict")


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.app.run(main)
