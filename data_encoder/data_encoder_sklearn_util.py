from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder   # 标签编码
from sklearn.preprocessing import OneHotEncoder  # one_hot编码
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np


class DataProcess(object):  # 特征处理
    def __init__(self, process_type):
        self.process_type = process_type

        if self.process_type == "Binary":  # 二值化处理
            self.processmodule = Binarizer(copy=True, threshold=0.0)
            # 大于 threshold 的映射为1， 小于 threshold 的映射为0

        elif self.process_type == "MinMax":  # 归一化处理
            self.processmodule = MinMaxScaler(feature_range=(0, 1), copy=True)

        elif self.process_type == "Stand":  # 标准化处理
            self.processmodule = StandardScaler(copy=True, with_mean=True, with_std=True)

        elif self.process_type == "Normal":  # 正则化处理
            self.processmodule = Normalizer(copy=True, norm="l2")  # 可选择l1, max ,l2三种

        elif self.process_type == "MultiLabelBinar":   # 多标签二值化处理
            self.processmodule = MultiLabelBinarizer(sparse_output=False)  # 使用其他CRS格式使用True
        else:
            raise ValueError("please select a correct process_type")

    def fit_transform(self, data):
        return self.processmodule.fit_transform(data)

    def fit(self, data):
        self.processmodule.fit(data)

    def transform(self, data):
        self.processmodule.transform(data)

    def set_params(self, params):
        self.processmodule.set_params(**params)

    def get_params(self):
        return self.processmodule.get_params(deep=True)

    def get_classes(self):
        assert self.process_type in {"MultiLabelBinar"}
        return self.processmodule.classes_  # 输出相关的classs有哪些不同的值

    def invser_transform(self, data):
        assert self.process_type in {"MultiLabelBinar", "MinMax", "Stand"}
        return self.processmodule.inverse_transform(data)

    def get_max(self):  # 获取数组中所多有维度上的最大值与最小值
        assert self.process_type in {"MinMax", "Stand"}
        return self.processmodule.data_max_

    def get_min(self):
        assert self.process_type in {"MinMax", "Stand"}
        return self.processmodule.data_min_

    def partial_fit(self):
        # 使用最后的一个缩放函数来在线计算最大值与最小值
        assert self.process_type in {"MinMax", "Stand"}
        return self.processmodule.partial_fit()


class DataEncoder(object):  # 支持三种编码方式
    def __init__(self, encoder_type):
        assert encoder_type in {"one_hot", "label", "Ordinal"}
        self.encoder_type = encoder_type
        if self.encoder_type == "one_hot":  # 种类的编码
            self.encodermodule = OneHotEncoder(categories='auto', drop=None, sparse=True,
                                               dtype=np.float64, handle_unknown='error')
            # categories 可取 "auto" 或种类的列表
            # drop  可取 {‘first’, ‘if_binary’} None  或 array [i] 表示丢弃第i个
            # first 表示丢弃每个种类特征的第一个， 二进制
            # sparse  返回一个稀疏矩阵，否则返回一个数组
            # handle_unknown  {‘error’, ‘ignore’}, default=’error’
        elif self.encoder_type == "label":
            self.encodermodule = LabelEncoder()

        elif self.encoder_type == "Ordinal":  # 序号编码
            self.encodermodule = OrdinalEncoder(categories="auto", dtype=np.float64)
            # categories 用法与onehot 差不多
        else:
            raise ValueError("please select a correct encoder_type")

    def fit_transform(self, data):
        return self.encodermodule.fit_transform(data)

    def fit(self, data):
        self.encodermodule.fit(data)

    def transform(self, data):
        self.encodermodule.transform(data)

    def set_params(self, params):
        self.encodermodule.set_params(**params)

    def get_params(self):
        return self.encodermodule.get_params(deep=True)

    def inverse_transform(self, data):
        return self.encodermodule.inverse_transform(data)

    def get_classes(self):
        assert self.encoder_type in {"label"}
        return self.encodermodule.classes_

    def get_category(self):
        assert self.encoder_type in {"one_hot", "Ordinal"}
        return self.encodermodule.categories_  # 返回数组列表

    def get_feature_names(self, output_feature):  # 获取输出特征的特征名字
        assert self.encoder_type in {"one_hot"}
        return self.encodermodule.get_feature_names(output_feature)


if __name__ == "__main__":
    # data1 = [[1., -1.,  2.], [2.,  0.,  0.], [0.,  1., -1.]]  # 二值化
    # print(data1)
    # fn = DataProcess("Binary")
    # print(fn.fit_transform(data1))
    #
    # data2 = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]  # 极大极小值归一化
    # print(data2)
    # fn = DataProcess("MinMax")
    # yt = fn.fit_transform(data2)  # 按照列进行归一化处理
    # print(yt)
    # print(fn.invser_transform(yt))   # 再进行反归一
    # print(fn.get_max())
    # print(fn.get_min())
    #
    # #
    #
    # data3 = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]  # 正则化
    # print(data3)
    # fn = DataProcess("Normal")
    # yt = fn.fit_transform(data3)
    # print(yt)
    #
    # data4 = [[0, 0], [0, 0], [1, 1], [1, 1]]
    # print(data4)
    # fn = DataProcess("Stand")
    # yt = fn.fit_transform(data4)   # 标准化
    # print(yt)
    # print(fn.invser_transform(yt))

    # data5 = [(1, 2), (3, 4), (5,)]
    # print(data5)
    # fn = DataProcess("MultiLabelBinar")
    # yt = fn.fit_transform(data5)  # 多标签编码
    # print(yt)
    # print(fn.invser_transform(yt))  # 反归一

    # data6 = [['Male', 1], ['Female', 3], ['Female', 2]]   # one_hot编码
    # fn = DataEncoder("one_hot")
    # yt = fn.fit_transform(data6).toarray()  # 转成数组
    # print(yt)
    # print(fn.inverse_transform(yt))

    # data7 = ["paris", "paris", "tokyo", "amsterdam"]   # 标签编码
    # fn = DataEncoder("label_encoder")
    # yt = fn.fit_transform(data7)
    # print(yt)
    # print(fn.inverse_transform(yt))
    # print(fn.get_classes())

    # data8 = [1, 2, 2, 6]
    # fn = DataEncoder("label_encoder")
    # yt = fn.fit_transform(data8)
    # print(yt)
    # print(fn.inverse_transform(yt))
    # print(fn.get_classes())

    data9 = [['Male', 1], ['Female', 3], ['Female', 2]]
    fn = DataEncoder("label_encoder")
    yt = fn.fit_transform(data9)
    print(yt)
    print(fn.inverse_transform(yt))
    print(fn.get_classes())



