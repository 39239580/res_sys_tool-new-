import pandas as pd
import os

# path1 = os.getcwd() + "/ratings.dat"  # 获取当前路径下的.dat文件
# data = pd.read_table(path1, header=None, sep="::")  #
# print(data)
# data.to_csv("ratings.csv",index=False,header=["userId","movieId","rating","timestamp"])


def dat2csvfn(filename, header):
    path1 = os.getcwd() + "/"+filename+".dat"  # 获取当前路径下的.dat文件
    data = pd.read_table(path1, header=None, sep="::")  #
    print(data)
    data.to_csv(filename+".csv", index=False, header=header)

