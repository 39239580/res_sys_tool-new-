# 构建宠物数据集
# 每行描述一只宠物，　每列描述一个属性，　使用此信息来预测宠物的收养速度

import numpy as np
import pandas as pd
import tensorflow as tf
from  tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os
from customer_dataset_util.Dataset_common_api import commonDataSet, trainformation


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'   #　将所有的显卡的info 信息干掉
pd.set_option("display.max_columns", None)
csv_file = "datasets/petfinder-mini/petfinder-mini.csv"
dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
if os.path.exists(csv_file):
    dataframe = pd.read_csv(csv_file)
else:
    tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
                            extract=True, cache_dir='.')
    dataframe = pd.read_csv(csv_file)
    # 数据集较小，　一次性读入内存中

print(dataframe.head(5))

# 原始数据集中的任务是用来预测宠物的收养速度，例如，在第一周，第一个月，　前三个月，
# 修改标签列，　０　表示未收养宠物，１　表示已收养宠物
# 原始数据中，４表示宠物未被收养
dataframe["target"] = np.where(dataframe["AdoptionSpeed"]==4,0,1)  #筛选出一部分数据，４，０，１
print(dataframe.head(5))
#　丢弃无用列
dataframe = dataframe.drop(columns=["AdoptionSpeed","Description"])
print(dataframe.head(5))

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), "train examples")
print(len(val), "validation examples")
print(len(test), "test examples")





#   创建输入管道,
def df_to_dataset(dataframe, shuffle= True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = commonDataSet(data_type="tensor_slices",tensor=(dict(dataframe), labels))
    if shuffle:
        ds = trainformation(DataSet=ds, opreation_type="shuffle",batch_size=len(dataframe))
    ds = trainformation(DataSet=ds, opreation_type="batch", batch_size=batch_size, drop_remainder=False)
    return ds

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in trainformation(train_ds,opreation_type="take", count=1):
    print("Every feature:", list(feature_batch.keys()))
    print("A batch of ages", feature_batch["Age"])
    print("A batch of targets", label_batch)

example_batch = next(iter(train_ds))[0]  # 一个批量的数据

def demo(feature_column):   # 显示相关数据　　＃　进行显示
    feature_layer = layers.DenseFeatures(feature_columns=feature_column)
    print(feature_layer(example_batch).numpy())


feature_columns = []
# ---------------------------------------数值列--------------------------------------------
for header in ["PhotoAmt", "Fee", "Age"]:
    feature_columns.append(feature_column.numeric_column(header))
# 测试
# photo_count = feature_column.numeric_column('PhotoAmt')
# demo(photo_count)

# --------------------------------------分桶列--------------------------------------------------
age = feature_column.numeric_column(key="Age")
age_buckets = feature_column.bucketized_column(source_column=age, boundaries=[1,2,3,4,5])
# 测试
# demo(age_buckets)
feature_columns.append(age_buckets)

# --------------------------------------种类列--------------------------------------------------
animal_type = feature_column.categorical_column_with_vocabulary_list(key="Type", vocabulary_list=["Cat", "Dog"])
animal_type_one_hot = feature_column.indicator_column(animal_type)
# demo(animal_type_one_hot)

print("ok")
# --------------------------------------序号列--------------------------------------------------
indicator_column_names = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                          'FurLength', 'Vaccinated', 'Sterilized', 'Health']
for col_name in indicator_column_names:
    categorical_column = feature_column.categorical_column_with_vocabulary_list(col_name, dataframe[col_name].unique())
    indicator_column = feature_column.indicator_column(categorical_column=categorical_column)
    feature_columns.append(indicator_column)

# print("ok")
# --------------------------------------------------向量列----------------------------------------
breed1 = feature_column.categorical_column_with_vocabulary_list(key="Breed1", vocabulary_list=dataframe.Breed1.unique())
breed1_embedding = feature_column.embedding_column(breed1, dimension=8)
feature_columns.append(breed1_embedding)
# 测试
breed1_hashed = feature_column.categorical_column_with_hash_bucket(
    'Breed1', hash_bucket_size=10)
demo(feature_column.indicator_column(breed1_hashed))

# ---------------------------------------------------交叉特征-------------------------------------
# 测试
# crossed_feature = feature_column.crossed_column([age_buckets, animal_type], hash_bucket_size=10)
# demo(feature_column.indicator_column(crossed_feature))
age_type_feature = feature_column.crossed_column([age_buckets, animal_type],hash_bucket_size=100)
feature_columns.append(feature_column.indicator_column(age_type_feature))
# print(feature_column)


# 定义输入层
feature_layer = tf.compat.v1.keras.layers.DenseFeatures(feature_columns=feature_columns)
print(feature_layer)

# 定义完整模型
model = tf.keras.Sequential([feature_layer,
                             layers.Dense(128,activation="relu"),
                             layers.Dense(128, activation="relu"),
                             layers.Dropout(0.1),
                             layers.Dense(1,activation="sigmoid")])

# 模型编译
model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])

#　训练
model.fit(train_ds, validation_data=val_ds,epochs=10)
#
# 评估
test_loss, test_acc = model.evaluate(test_ds)

# 显示评估的正确率
print('===================\nTest accuracy:', test_acc)

