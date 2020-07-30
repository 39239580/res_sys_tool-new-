import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
from customer_dataset_util.Dataset_common_api import commonDataSet, trainformation, version_waring
from customer_dataset_util.Dataset_common_api import gen_iter, get_next_element
import os

# 构建心脏病数据集　　
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   #　将所有的显卡的info 信息干掉
pd.set_option("display.max_columns", None)  # 显示所有的列
file_path = "test.csv"
if os.path.exists(file_path):
    dataframe = pd.read_csv("test.csv",index_col=0)
else:
    URL = 'https://storage.googleapis.com/applied-dl/heart.csv'  # 文件地址
    dataframe = pd.read_csv(URL)   # 将数据读进　内存，　若数据过大，csv文件过大，使用tf.data 从磁盘中读取
    dataframe.to_csv("test.csv")

print(dataframe.head())
# 训练集与测试机，　验证机的比例分别为　0.6, 0.2, 0.2
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), "train_examples")
print(len(val), "validation examples")
print(len(test), "test　examples")
version_waring(poweroff=False)

#  -----------------------------------使用通用dataset来创建pipeline-----------------------
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    # 创建对象
    ds=commonDataSet(data_type="tensor_slices", tensor=(dict(dataframe), labels))

    if shuffle:
        ds=trainformation(DataSet=ds, opreation_type="shuffle", batch_size=len(dataframe))
    # print(type(ds))
    ds= trainformation(DataSet=ds, opreation_type="batch", batch_size=batch_size, drop_remainder=False)
    # print(type(ds))
    # print("ok")
    return ds

batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)  # 训练集批大小为５
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)  # 验证集
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)  #测试集


# 在上述创建的pipeline 的基础上  执行相应的数据调用
for feature_batch, label_batch in train_ds.take(2):  # 取出前两批
    print("Every feature:", list(feature_batch.keys()))
    print("A batch of ages", feature_batch["age"])
    print("A batch of targets", label_batch)

#  https://tensorflow.google.cn/tutorials/structured_data/feature_columns   参考路径

#　['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
#   age 年龄　  sex性别(1=男，0=女)     cp 胸腔疼痛类型(0,1,2,3,4)
#  trestbps 静态血压　　　chol　胆固醇　　　fbs　空腹血糖含量达到120mg/dl 1=是，　0=否
#  restecg　静态心电图(0,1,2)   　thalach　最大心率　　　　　exang　运动是否引发心绞痛(1=是，　0=否)
#  oldpeak　运动相对休息诱发ST段压低　　　运动峰st段位坡度
#  ca 用荧光染色的主要血管数量0-3  thal 地中海贫血　3=正常，　6=固定缺陷，　7=可逆转缺陷
# 数值化的特征有　age, trestbps, chol, thalach, oldpeak, slope, ca
# 种类特征有　sex, cp, fbs, restecg, exang, thal

# ---------------------------年龄处理　　分桶处理-------------------------------
# #18岁以下, 18-25, 25-30, 30-35, 35-40, 40-45, 45-50, 50-55, 55-60, 60-65, 65以上岁数
# age_buckets = feature_column.bucketized_column(source_column=age, boundaries=[18,25,30,35,40,45,50,55,60,65])
# # ---------------------------地中海贫血，　包括normal, fixed(固定), reversible(可逆转)---------------------------
# # 获取thal 字段原始数据
# thal = feature_column.categorical_column_with_vocabulary_list(key="thal", vocabulary_list=["fixed","normal", "reversible"])
# # 第一种为one_hot编码，　转换为one-hot编码
# thal_one_hot = feature_column.indicator_column(thal)
# # 第二种为hash编码的方式，　当特征取值范围非常大的时候，　可使用hash编码的方式
# # thal_hashed = feature_column.categorical_column_with_hash_bucket(key="thal", hash_bucket_size=1000)
# # 第三种为embedding的编码方式，　当特征取值范围非常大的时候
# # thal_embedding = feature_column.embedding_column(key=thal, dimension=8)
# # 假设有两个字段香瓜关联，需要整体表达，可使用feature crosses编码方式
# crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)

feature_columns = []
# ---------------------------根据字段名，　添加所需的数据列---------------------------------------------
for header in ["age", "trestbps", "chol", "thalach","oldpeak", "slope","ca"]:
    feature_columns.append(feature_column.numeric_column(key=header))

# 取出年龄数据
age = feature_column.numeric_column("age")
age_buckets = feature_column.bucketized_column(age, boundaries=[18,25,30,35,40,45,50,55,60,65])
# 数据段作为一个新参量添加到数据集
feature_columns.append(age_buckets)

# 获取thal字段原始数据
thal = feature_column.categorical_column_with_vocabulary_list(key="thal",
                                                              vocabulary_list=["fixed", "normal","reversible"])
# one-hot编码
thal_one_hot = feature_column.indicator_column(thal)
# 作为新的数据添加列
feature_columns.append(thal_one_hot)

# 将thal嵌入８维空间做向量化
thal_embedding = feature_column.embedding_column(categorical_column=thal, dimension=8)
feature_columns.append(thal_embedding)

#　把年龄段和thal字段作为关联属性加入新列
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

# 定义输入层
feature_layer = tf.compat.v1.keras.layers.DenseFeatures(feature_columns=feature_columns)
print(feature_layer)

# 定义完整模型
model = tf.keras.Sequential([feature_layer,
                             layers.Dense(128,activation="relu"),
                             layers.Dense(128, activation="relu"),
                             layers.Dense(1,activation="sigmoid")])

# 模型编译
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#　训练
model.fit(train_ds, validation_data=val_ds,epochs=5)
#
# 评估
test_loss, test_acc = model.evaluate(test_ds)

# 显示评估的正确率
print('===================\nTest accuracy:', test_acc)

