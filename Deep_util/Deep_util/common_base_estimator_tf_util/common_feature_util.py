import tensorflow as tf
from tensorflow.compat.v1.feature_column import categorical_column_with_hash_bucket # 经过hash处理的列，将类别数量非常大的特征，　模型会计算他的哈细致
from tensorflow.compat.v1.feature_column import categorical_column_with_vocabulary_list # 分类词汇列，将字符串表示独热矢量，根据文件中的词汇表将每个字符串映射到一个整数
from tensorflow.compat.v1.feature_column import categorical_column_with_vocabulary_file # 分类词汇列，将字符串表示独热矢量，根据文件中的词汇将每个字符串映射到一个整数
from tensorflow.compat.v1.feature_column import categorical_column_with_identity  # 分类识别列，将每个分桶表示一个唯一的整数，　模型可在分类表示列中学习每个类别各自的权重
from tensorflow.compat.v1.feature_column import crossed_column  # 特征组合列

# Dense Columns 中API
from tensorflow.compat.v1.feature_column import numeric_column   # 数值列，将默认数据类型tf.float32的数值制定为模型的输入, 实数或者数值特征
from tensorflow.compat.v1.feature_column import embedding_column   # 嵌入列，　将分类列视为输入
from tensorflow.compat.v1.feature_column import indicator_column   # 指标列  转成one_hot
# 混合类　进行分桶割列，　将数字列根据数值范围分成不同的列别，　为模型中加入飞线性特征，　提高模型的表达能力
from tensorflow.compat.v1.feature_column import bucketized_column

from tensorflow.compat.v1.feature_column import make_parse_example_spec  #　根据输入的feature_columns创建解析规范字典

# 相比上述的API, 下列的api主要的不同来自于　序列
from tensorflow.compat.v1.feature_column import sequence_categorical_column_with_hash_bucket
from tensorflow.compat.v1.feature_column import sequence_categorical_column_with_identity
from tensorflow.compat.v1.feature_column import sequence_categorical_column_with_vocabulary_file
from tensorflow.compat.v1.feature_column import sequence_categorical_column_with_vocabulary_list
from tensorflow.compat.v1.feature_column import sequence_numeric_column
# 共享　embedding_vec向量
from tensorflow.compat.v1.feature_column import shared_embedding_columns
# from tensorflow.compat.v1.feature_column import shared_embedding_columns
# 带有加权的　categorical_column
from tensorflow.compat.v1.feature_column import weighted_categorical_column
from tensorflow.python.framework import dtypes
# 所有的categorical 特征均需要将其转成Dense columns 类型，才能传入到模型当中，　　　　
# categorical  经过indicator columns  才能转成　Dense columns
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '１'   #　将所有的显卡的info 信息干掉
def common_feature_enginee(feature_task, params):
    if feature_task == "categorical_identity":
        feature = categorical_column_with_identity(key=params["key"],
                                                   num_buckets=params["num_buckets"],
                                                   default_value=params.get("default_value",None)
                                                   )
        # API 使用　num_buckets为取值范围　[0, num_buckets], default_value  默认值
        """
        将输入值本身作为分类ID，　序号编码
        """


    elif feature_task == "categorical_vocab_file":
        feature = categorical_column_with_vocabulary_file(key=params["key"],
                                                          vocabulary_file=params["vocabulary_file"],
                                                          vocabulary_size=params.get("vocabulary_size",None),
                                                          num_oov_buckets=params.get("num_oov_buckets",0),
                                                          default_value=params.get("default_value",None),
                                                          dtype=dtypes.string)
        """
        直接从磁盘中直接读取
        输入为字符串或者整数格式，并且，有一个词汇词汇表文件将每个值映射到整个id， 使用此选项. vocabulary_size, num_oov_buckets
        只能指定其中一个，　　不能同时使用。　对于输入字典　features,   features[key]是tensor 或　SparseTensor。　假设为tensor, 
        缺失值　int　使用-1代替，　string使用''代替。num_oov_buckets为　输入位于词汇字典中的映射根据字典映射，超过词典范围的，映射到
        num_oov_buckets　中
        """


    elif feature_task == "categorical_vocab_list":
        feature = categorical_column_with_vocabulary_list(key=params["key"],
                                                          vocabulary_list=params["vocabulary_list"],
                                                          dtype=params.get("dtype",None),
                                                          default_value=params.get("default_value",-1),
                                                          num_oov_buckets=params.get("num_oov_buckets",0))
        """
        从内存中。　
        输入为字符串或者整数格式，并且内存中的词汇表每个值映射到整个id，使用此选项，　default_value，　num_oov_buckets
        只能指定一个，　不能同时使用。　对于输入字典　features,   features[key]是tensor 或　SparseTensor。　假设为tensor, 
        缺失值　int　使用-1代替，　string使用''代替。num_oov_buckets为　输入位于词汇字典中的映射根据字典映射，超过词典范围的，映射到
        num_oov_buckets　中
        """

    elif feature_task == "categorical_hash":
        feature = categorical_column_with_hash_bucket(key=params["key"],
                                                      hash_bucket_size=params["hash_bucket_size"],
                                                      dtype=dtypes.string)
        """
        稀疏特征为字符串或整数格式，　通过hash处理后，　将输入映射到有限数量的桶中
        """

    elif feature_task == "categorical_weight":
        feature = weighted_categorical_column(categorical_column=params["categorical_column"],
                                              weight_feature_key=params["weight_feature_key"],
                                              dtype=dtypes.float32)
        """
        加入了加权处理，每个稀疏输入有相同的id和值的时候，使用此选项，如将文本文档表示为单词频率的集合，则可提供２个并行的稀疏输入
        """


    elif feature_task == "crossed_column":  # 交叉分类特征列
        feature = crossed_column(keys=params["keys"],
                                 hash_bucket_size=params["hash_bucket_size"],
                                 hash_key=params.get("hash_key", None))
        """
        执行交叉分类要素的列。交叉特征将会根据hash_bucket_size进行哈希处理。
        第一个稀疏tensor
        shape = [2, 2]
        {
            [0, 0]: "a"　　　[0, 0]代表的是坐标
            [1, 0]: "b"
            [1, 1]: "c"
        }
        第二个稀疏tensor
        shape = [2, 1]
        {
            [0, 0]: "d"
            [1, 0]: "e"
        }
        交叉
        shape = [2, 2]
        {
            [0, 0]: Hash64("d", Hash64("a")) % hash_bucket_size
            [1, 0]: Hash64("e", Hash64("b")) % hash_bucket_size
            [1, 1]: Hash64("e", Hash64("c")) % hash_bucket_size
        }
        """

    elif feature_task == "numeric_columns":   #　实数或数值型特征
        feature = numeric_column(key=params["key"],
                                 shape=params.get("shape",(1,)),
                                 default_value=params.get("default_value",None),
                                 dtype=dtypes.float32,
                                 normalizer_fn=params.get("normalizer_fn", None))

    elif feature_task == "embedding_column":   #  embedding 编码
        feature = embedding_column(categorical_column=params["categorical_column"],
                                   dimension=params["dimension"],
                                   combiner=params.get("combiner",'mean'),
                                   initializer=params.get("initializer",None),
                                   ckpt_to_load_from=params.get("ckpt_to_load_from",None),
                                   tensor_name_in_ckpt=params.get("tensor_name_in_ckpt",None),
                                   max_norm=params.get("max_norm",None),
                                   trainable=params.get("trainable", True))
        """
        DNN模型的特征输入用的比较多
        """

    elif feature_task == "indicator_column":  # one_hot编码　　
        feature = indicator_column(categorical_column=params["categorical_column"])
        """
        给定categorical　columns　多热点表示
        DNN魔性，　indicator_column可用于包装任何内容，如果buckets数字非常大的时候，考虑使用embedding_column
        """

    elif feature_task == "bucketized_column":   # 分桶编码
        feature = bucketized_column(source_column=params["source_column"],
                                    boundaries=params["boundaries"])
        """
        boundaries 包含左边界，不包含右边界。
        boundaries=[0.,1.,2.]  生成的桶(-inf, 0.), [0., 1.), [1.,2.) 和[2., +inf)
        假设输入的是
        boundaries　＝[0,10,100]
        input tensor = [[-5, 10000] [150, 10], [5, 100]]
        那么输出的为
        output =[[0, 3]   # 得到的是桶的序号
                 [3, 2]
                 [1, 3]]
        """

    elif feature_task == "make_parse_example_spec":  # 根据feature_columns输入创建解析规范字典
        feature = make_parse_example_spec(feature_columns=params["feature_columns"])
        # 返回字典

    elif feature_task == "shared_embedding":
        feature = shared_embedding_columns(categorical_columns=params["categorical_columns"],
                                           dimension=params["dimension"],
                                           combiner=params.get("combiner", 'mean'),
                                           initializer=params.get("initializer",None),
                                           shared_embedding_collection_name=params.get("shared_embedding_collection_name",None),
                                           ckpt_to_load_from=params.get("ckpt_to_load_from",None),
                                           tensor_name_in_ckpt=params.get("tensor_name_in_ckpt",None),
                                           max_norm=params.get("max_norm",None),
                                           trainable=params.get("trainable",True)
                                           )
        """
        从sparse, categorical输入转成密集型向量的列表，　与embedding_column相似，　不同在于，　会生成共享权重
        """

    elif feature_task == "sequence_numeric":
        feature = sequence_numeric_column(key=params["key"],
                                          shape=params.get("shape",(1,)),
                                          default_value=params.get("default_value",0),
                                          dtype=dtypes.float32,
                                          normalizer_fn=params.get("normalizer_fn", None))

    elif feature_task == "sequence_categorical_vocab_list":
        feature = sequence_categorical_column_with_vocabulary_list(key=params["key"],
                                                                   vocabulary_list=params["vocabulary_list"],
                                                                   dtype=params.get("dtype", None),
                                                                   default_value=params.get("default_value", -1),
                                                                   num_oov_buckets=params.get("num_oov_buckets", 0)
                                                                   )

    elif feature_task == "sequence_categorical_vocab_file":
        feature = sequence_categorical_column_with_vocabulary_file(key=params["key"],
                                                                   vocabulary_file=params["vocabulary_file"],
                                                                   vocabulary_size=params.get("vocabulary_size", None),
                                                                   num_oov_buckets=params.get("num_oov_buckets", 0),
                                                                   default_value=params.get("default_value", None),
                                                                   dtype=dtypes.string)

    elif feature_task == "sequence_categorical_identity":
        feature = sequence_categorical_column_with_identity(key=params["key"],
                                                            num_buckets=params["num_buckets"],
                                                            default_value=params.get("default_value",None)
                                                            )

    elif feature_task == "sequence_categorical_hash":
        feature = sequence_categorical_column_with_hash_bucket(key=params["key"],
                                                               hash_bucket_size=params["hash_bucket_size"],
                                                               dtype=dtypes.string)

    else:
        raise  ValueError("feature_task is error!")

    return feature




# －－－－－－－－－－－－－－－－－－－测试部分－－－－－－－－－－－－－－－－－－－－－－－－－－－
def debug_fn():
    # 手动创建简单的数据集
    dataset = tf.compat.v1.data.Dataset.from_tensor_slices(({"age":[1,2,3],
                                                             "relationship":["wife", "husband", "unmarried"]},

                                                            [0, 1, 0]))
    dataset=dataset.batch(3)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    features, label = next_element

    # 创建类别特征
    age = common_feature_enginee("categorical_identity",{"key":"age","num_buckets":5, "default_value":0})
    relationship = common_feature_enginee("categorical_vocab_list", {"key":"relationship",
                                                                     "vocabulary_list":
                                                                         ["husband", "Not-in-family","wife","Own-child",
                                                                          "unmarried","other-relative"]})
    # 由于后面使用　tf.feature_column.input_layer 来查看结果，它只接受ｄｅｎｓｅ特征
    # 因此先使用　indicator_column　将embedding_columns 转成ｄｅｎｓｅ
    age_one_hot =common_feature_enginee("indicator_column",{"categorical_column":age})
    relationship_one_hot = common_feature_enginee("indicator_column",{"categorical_column":relationship})

    age_one_hot_dense = tf.compat.v1.feature_column.input_layer(features=features, feature_columns=age_one_hot)
    relationship_one_hot_dense = tf.compat.v1.feature_column.input_layer(features=features,
                                                                         feature_columns=relationship_one_hot)
    concat_one_hot_dense = \
        tf.compat.v1.feature_column.input_layer(
            features=features, feature_columns=[age_one_hot_dense, relationship_one_hot_dense])

    with tf.compat.v1.Session() as sess:
        # 使用　table_initializeer()   初始化　Graph中所有的LookUpTable
        sess.run(tf.compat.v1.tables_initializer())
        a, b, c = sess.run([age_one_hot_dense, relationship_one_hot_dense, concat_one_hot_dense])
        print("age_one_hot:\n{}".format(a))
        print("relationship_one_hot:\n".format(b))
        print("concat_one_hot:\n".format(c))

debug_fn()


# -------------------------------------测试部分2---------------------------------------
# https://blog.csdn.net/Elenstone/article/details/105677345
# 功能说明　https://www.jianshu.com/p/c20f6656a3e4

