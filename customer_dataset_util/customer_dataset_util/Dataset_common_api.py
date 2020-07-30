import tensorflow as tf
from tensorflow import data
from tensorflow.compat.v1.data.experimental import AUTOTUNE
import itertools
import numpy as np
"""
tf.data.Dataset　API 用于构建高效的输入管道，　也是tensorflow 官方比较推荐的数据数据输入管道pipline
1. 根据输入数据创建源数据集合
2．应用数据集转换以处理数据。
3. 遍历数据集并处理元素
流行以迭代方式进行，因此整个数据集不需要放入内存中。　节约内存。
"""


def commonDataSet(data_type,
                  tensor=None,
                  record_bytes=None,
                  filename=None,
                  file_pattern=None,
                  generator=None,
                  output_types=None,
                  end_point=None,
                  step=1,
                  output_shapes=None,
                  args=None,
                  compression_type=None,
                  buffer_size=None,
                  num_parallel_reads=None,
                  header_bytes=None,
                  footer_bytes=None,
                  start_point=0,
                  datasets=None):
    assert data_type in {"file_list", "TFrecord", "binary", "tensor_slices", "tensor", "generator",
                         "range", "zip", "debug"}

    if data_type == "file_line":  # 文件中的行
        DataSet = data.TextLineDataset(filenames=filename,
                                       compression_type=compression_type,
                                       buffer_size=buffer_size,
                                       num_parallel_reads=num_parallel_reads
                                       )
        """
        filenames   使用的字符串tensor 或者　包含多个文件名
        compression_type　　可选项　""(不压缩)   ZLIB  GZIP  字符串　　　None
        buffer_size 可选，　整形，　表示要缓存额大小，　０　会导致根据亚索类型选择默认的缓冲值  None
        num_parallel_reads　　可选，　int  表示要病毒读取的文件数。　如果值大于１，　 None
        则以交错顺序输出并行读取的文件记录，　如果设为None ,则被顺序读取
        """

    elif data_type == "file_list":
        DataSet = data.Dataset.list_files(file_pattern=file_pattern,
                                          shuffle=None,
                                          seed=None)
        """
        file_pattern　　使用的文件列表　或字符串　"/path/*.txt"
        shuffle　　是否进行打乱操作  默认为打乱
        seed  随机种子数　　int
        """

    elif data_type == "TFrecord":  # 读取TFreacord 文件使用的　　tfrecord 文件列表
        DataSet = data.TFRecordDataset(filenames=filename,
                                       compression_type=compression_type,
                                       buffer_size=buffer_size,
                                       num_parallel_reads=num_parallel_reads)
        """
        filenames   使用的字符串tensor 或者　包含多个文件名
        compression_type　　可选项　""(不压缩)   ZLIB  GZIP  字符串　　　None
        buffer_size 可选，　整形，　表示要缓存额大小，　０　会导致根据亚索类型选择默认的缓冲值  None
        num_parallel_reads　　可选，　int  表示要并行读取的文件数。　如果值大于１，　 None
        则以交错顺序输出并行读取的文件记录，　如果设为None ,则被顺序读取
        """

    elif data_type == "binary":  # CIFAR10数据集使用的数据格式就是这种，　二进制文件
        DataSet = data.FixedLengthRecordDataset(filenames=filename,
                                                record_bytes=record_bytes,
                                                header_bytes=header_bytes,
                                                footer_bytes=footer_bytes,
                                                buffer_size=buffer_size,
                                                compression_type=compression_type,
                                                num_parallel_reads=num_parallel_reads)
        """
        filenames   使用的字符串tensor 或者　tf.data.Dataset中包含多个文件名
        record_bytes   tf.int64    数据类型
        header_bytes  表示文件开头要跳过的字节数，　可选
        footer_bytes　　表示文件末尾要忽略的字节数
        buffer_size 可选，　整形，　表示要缓存额大小，　０　会导致根据亚索类型选择默认的缓冲值  None
        compression_type　　可选项　""(不压缩)   ZLIB  GZIP  字符串　　　None       
        num_parallel_reads　　可选，　int  表示要病毒读取的文件数。　如果值大于１，　 None
        则以交错顺序输出并行读取的文件记录，　如果设为None ,则被顺序读取
        """

    elif data_type == "generator":
        DataSet = data.Dataset.from_generator(generator=generator,
                                              output_types=output_types,
                                              output_shapes=output_shapes,
                                              args=args,
                                              )
        """
        generator  , iter迭代对象，　若args未指定，怎，　generator不必带参数，否则，它必须于args 中的参数一样多
        output_types　数据类型
        output_shapes　尺寸　　可选
        args
        """

    elif data_type == "range":
        DataSet = data.Dataset.range(start_point, end_point, step)

    elif data_type == "zip":  # zip 操作, 对两个或者对个数据集进行合并操作
        DataSet = data.Dataset.zip(datasets=datasets)
        """
        dataset  必须是一个tuple   (datasetA, datasetB)
        """
    elif data_type == "tensor_slices":  # 张量 作用于　from_tensor 的区别，　from_tensor_slices的作用是可以将tensor进行切分
        # 切分第一个维，　如　tensor 使用的是　np.random.uniform(size=(5,2)).  　　即　划分为　５行２列的数据形状。　　５行代表５个样本
        # 2列为每个样本的维。
        DataSet = data.Dataset.from_tensor_slices(tensors=tensor)
        # 实际上　是将array作为一个tf.constants 保存到计算图中，　当array比较大的时候，导致计算图非常大，　给传输于保存带来不变。
        # 此时使用placeholder　取代这里的array.  并使用initializable iterator，　只在需要时，将array传进去，　
        # 这样可以避免大数组保存在图中
    elif data_type == "debug":
        DataSet = data.Dataset(...)

    else:
        DataSet = data.Dataset.from_tensors(tensors=tensor)

    return DataSet


def trainformation(DataSet, opreation_type, map_fn=None, batch_size=None, drop_remainder=None,
                   DataB=None, dataset_fn=None, start=None, filter_n=None, map_func=None, buffer_size=None,
                   initial_state=None, reduce_func=None, num_shards=None, index=None, count=None, size=None,
                   shift=None, stride=None, options=None, save_path=None,
                   num_parallel_calls=AUTOTUNE, seed=None,
                   reshuffle_each_iteration=None, cycle_length=AUTOTUNE,
                   block_length=1):

    assert opreation_type in {"map", "batch", "shuffle", "repeat", "cache", "concatenate", "apply",
                              "enumerate", "filter", "flat_map", "prefetch", "reduce",
                              "shard", "skip", "take", "unbatch", "window", "with_options", "interleave"}
    if opreation_type == "map":  # map 操作
        """
        map_fn
        num_parallel_calls  可选项　int32 表示要并行异步处理的数字元素，如果未制指定，则按顺序处理元素
        """
        return DataSet.map(map_fn, num_parallel_calls=num_parallel_calls)


    elif opreation_type == "batch":  # 设置批量大小
        """
        batch_size   设置批量大小
        drop_remainder　bool型参数，　可选，　设置为TRUE,当最后一批的数据比其他批数据量少的时候，进行删除掉。
        """
        # data.Dataset.batch()
        return DataSet.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    elif opreation_type == "shuffle":  # 打乱操作
        """
        batch_size　批大小
        seed　随机种子
        reshuffle_each_iteration 每次迭代都进行打乱，　布尔型　可选项，　默认为真
        """
        return DataSet.shuffle(buffer_size=batch_size, seed=seed,
                               reshuffle_each_iteration=reshuffle_each_iteration)

    elif opreation_type == "repeat":  # 重复次数
        """
        count 可选　　重复的次数，None 或　－１　为无限循环
        """
        return DataSet.repeat(count=count)

    elif opreation_type == "cache":  # 缓存操作
        if save_path:
            return DataSet.cache(save_path)   # 存放到硬盘
        else:
            return DataSet.cache()   # 存放至内存

    elif opreation_type == "concatenate":  # 两个数据集进行拼接
        return DataSet.concatenate(DataB)

    elif opreation_type == "apply":  # apply 操作，　
        return DataSet.apply(dataset_fn)

    elif opreation_type == "as_numpy_iterator":  # 返回一个numpy 迭代对象，　于其他函数配合使用, v2.2版本中可以使用 1.14版本用不了
        # 2.0版本也没有
        return DataSet.as_numpy_iterator()

    elif opreation_type == "enumerate":  # 枚举数据中的
        """
        start  整型，　代表枚举开始的元素
        """
        return DataSet.enumerate(start=start)

    elif opreation_type == "filter":  # 过滤函数
        """
        predicate filter_n  过滤函数
        """
        return DataSet.filter(predicate=filter_n)

    elif opreation_type == "flat_map":  # 数据集展平
        return DataSet.flat_map(map_func=map_func)

    elif opreation_type == "prefetch":  # 预提元素
        """
        buffer_size  预提元素的最大元素
        """
        return DataSet.prefetch(buffer_size=buffer_size)


    elif opreation_type == "reduce":  # reduce 操作
        """
        输出出单个元素
        initial_state  转换的初始化状态的元素
        reduce_func 　映射函数
        """
        return DataSet.reduce(initial_state=initial_state, reduce_func=reduce_func)


    elif opreation_type == "shard":  # 创建一个Data 长度为1/shard 的子数据集合等间隔数据集
        """
        num_shards   整数　　子数据集的大小
        index　　索引　第几个子数据集
        """
        return DataSet.shard(num_shards=num_shards, index=index)


    elif opreation_type == "skip":  # skip 操作
        """
        count  跳到地count元素
        """
        return DataSet.skip(count=count)


    elif opreation_type == "take":  # 获取前面多少个元素
        return DataSet.take(count=count)

    elif opreation_type == "unbatch":  # batch的反操作 2.0版本的功能
        return DataSet.unbatch()

    elif opreation_type == "window":  # 将数据划分成多个window
        """
        size  窗口的大小
        shift　偏移量可选项，　必须为正数
        stride　　滑动窗口，　必须为正数
        drop_remainder　如果最后一个窗口的数据，小于批的大小，对最后一个窗口的数据进行删除操作。默认设置为true
        """
        return DataSet.window(size=size, shift=shift, stride=stride, drop_remainder=drop_remainder)


    elif opreation_type == "with_options":  # 创建带选项的数据集
        return DataSet.with_options(options=options)

    elif opreation_type == "interleave":  # 同时处理许多输入文件
        """
        map_func
        cycle_length  可选，　同时处理的输入元素的数量，没制定，则从cpu中的得到该值
        block_length　可选，循环到另一个输出元素钱　，每个输入元素要生成的连续元素的数量
        num_parallel_calls　可选，并行参数书两
        """
        return DataSet.interleave(map_func=map_func, cycle_length=cycle_length,
                                  block_length=block_length, num_parallel_calls=num_parallel_calls
                                  )

    else:
        raise ValueError("方法不能使用,　请仔细检查")

def gen_iter(DataSet, iter_way="one_short"):
    assert iter_way in {"one_short", "another"}
    if iter_way == "one_short":
        return tf.compat.v1.data.make_one_shot_iterator(dataset=DataSet)  # 较简单的迭代方式，　只能从头读到尾部一次
        #  自动初始化操作, 不需要进行手动的初始化操作
    else:
        return tf.compat.v1.data.make_initializable_iterator(dataset=DataSet)  # 读取较大数据集的操作
        # 返回的iter 是没有经过初始化状态的，　在运行进行的操作为　　　

# initializable_iterator才能使用此次函数
def init_iter(iters):  # 应用在　sess 中  初始化迭代对象
    # print(type(self.iter))
    return iters.initializer

def get_next_element(iters):  # 获取得迭代其的下一个元素
    return iters.get_next()

def string_handle(iters):
    return iters.string_handle()

def version_waring(poweroff=True):
    if poweroff:
        # 关闭　eager 模型，避免tf版本不兼容导致无法运行
        tf.compat.v1.disable_eager_execution()
    else:
        # eager 模型

        tf.compat.v1.enable_eager_execution()

# ------------------------------------- 下列函数于placeholder 相搭配----------------------------------------
# 构建迭代器件
# 基于给定的一个handle创建一个未初始化的　迭代对象
def from_string_handle(string_handle, output_types, output_shapes=None, output_classes=None):
    iters = tf.compat.v1.data.Iterator.from_string_handle(
        string_handle,
        output_types,
        output_shapes=output_shapes,
        output_classes=output_classes)
    return iters


# 基于给定的一个structure创建一个未初始化的　迭代对象
def from_struct(output_types, output_shapes=None, shared_name=None, output_classes=None):
    iters = tf.compat.v1.data.Iterator.from_structure(
        output_types,
        output_shapes=output_shapes,
        shared_name=shared_name,
        output_classes=output_classes)
    return iters


def iter_init(iter, dataset):
    return iter.make_initializer(dataset=dataset)

# -------------------------------debug------------------------------------------
def filter_fn_(x):
    return tf.math.less_equal(x, 1)


def flat_map_(x):
    # CommonDataSet
    return commonDataSet(data_type="tensor_slices",tensor=[x+1])


def gen_():
    for i in itertools.count(1):
        yield (i, [1]*i)


def interleave_(x):
    return trainformation(DataSet=commonDataSet(data_type="tensor",tensor=[x]), opreation_type="repeat",count=6)


def map_fn_(x):
    return x+1


def reduce_fn1(x, _):
    return x+1


def reduce_fn2(x, y):
    return x+y


def debug1():
    # -----------------------------------concatenate------------------------------
    a = commonDataSet(data_type="range", start_point=1, end_point=4) # 变成了对象
    b = commonDataSet(data_type="range", start_point=4, end_point=8)
    a = trainformation(DataSet=a, opreation_type="concatenate", DataB=b)  # 所以需要写成属性的情况

    # -----------------------------------filter-----------------------------------
    d = commonDataSet(data_type="tensor_slices", tensor=[1,2,3])
    d = trainformation(DataSet=d, opreation_type="filter", filter_n=filter_fn_)

    # ------------------------------------flat_map------------------------------------
    e = commonDataSet(data_type="tensor_slices", tensor=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    e = trainformation(DataSet=e, opreation_type="flat_map", map_func=flat_map_)

    # ------------------------------------from_gennerator----------------------------
    tf.compat.v1.enable_eager_execution()
    f = commonDataSet(data_type="generator", generator=gen_, output_types=(tf.int64, tf.int64),
                      output_shapes=(tf.TensorShape([]), tf.TensorShape([None])))

    print(type(f.take(2)))

    for value in f.take(2):
        print(value)
    for value in trainformation(DataSet=f, opreation_type="take",count=2):
        print(value)
    # ----------------------------interleave------------------------------------
    a=commonDataSet(data_type="range", start_point=1, end_point=6)
    # a.Data.interleave(map_func=interleave1_,cycle_length=2, block_length=4)


    a = trainformation(DataSet=a, opreation_type="interleave", map_func=interleave_, cycle_length=2, block_length=4)
    print("ok")

    # -----------------------------------map ---------------------------------------------
    a = commonDataSet(data_type="range", start_point=1, end_point=6)
    a = trainformation(DataSet=a, opreation_type="map", map_fn=map_fn_)

    # -----------------------------------reduce-------------------------------------------
    g = commonDataSet(data_type="range", end_point=5)
    g = trainformation(DataSet=g, opreation_type="reduce", initial_state=np.int64(0), reduce_func=reduce_fn1)
    h = commonDataSet(data_type="range", end_point=5)
    h = trainformation(DataSet=h, opreation_type="reduce", initial_state=np.int64(0), reduce_func=reduce_fn2)
    print("ok---")


# --------------------------------------创建自己的数据集demo------------------------------------------
def demo1():
    version_waring()   #将eager 模式关闭，　　数据长度为５
    dataset=commonDataSet(data_type="tensor_slices", tensor=np.array([1.0,2.0,3.0,4.0,5.0]))
    # dataset=data.Dataset.from_tensor_slices(tensors=np.array([1.0,2.0,3.0,4.0,5.0]))
    # iter=tf.compat.v1.data.make_one_shot_iterator(dataset=dataset)
    dataset = gen_iter(DataSet=dataset, iter_way="one_short")
    one_element=get_next_element(iters=dataset)
    print("ok")
    with tf.compat.v1.Session() as sess:
        for i in range(5): #取出５个元素
            print(sess.run(one_element))

# -------------------------------------另一种创建数据的方法demo---------------------------------------
def demo2():
    version_waring()   #将eager 模式关闭　　　数据长度为５
    dataset = commonDataSet(data_type="tensor_slices", tensor=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    dataset = gen_iter(DataSet=dataset, iter_way="another")
    one_element = get_next_element(iters=dataset)
    print("ok")
    with tf.compat.v1.Session() as sess:
        sess.run(init_iter(dataset))
        for i in range(5):  # 取出５个元素
            print(sess.run(one_element))


# -------------------------------------另一种创建数据的方法demo---------------------------------------
# 数据两从头读到尾部，会报错，　自动读取数据的长度
def demo3():
    version_waring()   #将eager 模式关闭
    dataset = commonDataSet(data_type="tensor_slices", tensor=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    dataset = gen_iter(DataSet=dataset, iter_way="one_short")
    one_element = get_next_element(iters=dataset)
    print("ok")
    with tf.compat.v1.Session() as sess:
        try:
            while True:
                print(sess.run(one_element))
        except tf.compat.v1.errors.OutOfRangeError:
            print("end!")

# Eager模式中，　创建迭代对象
# -----------------------------------Eager模式-----------------------------------------------------
# 此模式下，不需要使用到sess, 暂时为调通，　不建议使用，　一般不使用eager模式

# def demo4():
#     version_waring(poweroff=False)
#     dataset=CommonDataSet(data_type="tensor_slices", tensor=np.array([1.0,2.0,3.0,4.0,5.0]))
#     for one_element in tf.compat.v1.data.Iterator(dataset):
#         print(one_element)


# ----------------------------------placeholder与--------------------------------
# make_initializable_iterator  搭配使用  可以使用比较大的数据集
def demo5():
    version_waring()
    limit  = tf.compat.v1.placeholder(dtype=tf.int32, shape=[])
    dataset = commonDataSet(data_type="tensor_slices", tensor=tf.compat.v1.range(start=0, limit=limit))
    dataset = gen_iter(DataSet=dataset, iter_way="another")
    next_elements=get_next_element(iters=dataset)
    with tf.compat.v1.Session() as sess:
        sess.run(init_iter(dataset), feed_dict={limit:10})
        for i in range(10):
            value = sess.run(next_elements)
            assert i == value

#   -----------------------------tf.compat.v1.data.Iterator.from_string_handle创建迭代对象-------------------------------
# def demo6():
#     version_waring()
#     dataset_train=CommonDataSet(data_type="debug")
#     dataset_train.gen_iter(iter_way="one_short")
#     dataset_test=CommonDataSet(data_type="debug")
#     dataset_test.gen_iter(iter_way="one_short")
#     handle = tf.compat.v1.placeholder(tf.string, shape=[])
#     iterator=from_string_handle(string_handle=handle, output_types=dataset_train.Data.output_types)
#     next_element = iterator.get_next()
#     with tf.compat.v1.Session() as sess:
#         train_iterator_handle = sess.run(dataset_train.string_handle())
#         test_iterator_handle = sess.run(dataset_test.string_handle())
#         loss = f(next_element)
#
#         train_loss = sess.run(loss, feed_dict={handle: train_iterator_handle})
#         test_loss = sess.run(loss, feed_dict={handle: test_iterator_handle})


# def filter_n_demo7(x):
#     return x%2==0
#
# def demo7():
#     iterator = from_struct(output_types=tf.int64, output_shapes=tf.TensorShape([]))
#     dataset_range = CommonDataSet(data_type="range", end_point=10)
#     range_initializer = iter_init(iter=iterator,dataset=dataset_range)
#     dataset_evens = dataset_range.trainformation(opreation_type="filter", filter_n=filter_n_demo7)
#     evens_initializer = iterator.make_initializer(dataset=dataset_evens)
#     prediction, loss= model_fn(iterator.get_next())
#     with tf.compat.v1.Session() as sess:
#         for _ in range(num_epochs):
#             sess.run(range_initializer)
#             while True:
#                 try:
#                     pred, loss_val = sess.run([prediction, loss])
#                 except tf.compat.v1.errors.OutOfRangeError:
#                     break
#
#             # 初始化迭代其
#             sess.run(evens_initializer)
#             while True:
#                 try:
#                     pred, loss_val = sess.run([prediction, loss])
#                 except tf.compat.v1.errors.OutOfRangeError:
#                     break

# debug1()
# demo1()
# demo2()
# demo3()
# # demo4()
# demo5()
