import numpy as np
import pandas as pd
import pickle as pk


# 加载数据
def load_file(file_path, file_formate):
    assert file_formate in {"csv", "pk"}
    if file_formate == "csv":
        df = pd.read_csv(file_path, header=None,  sep="\t")
    else:
        with open(file_path, "rb") as f:
            df = pk.load(f)
    return df


# def libsvm2libffm(df):


file_path = "../deepFMtestdata/data/train.csv"
df=load_file(file_path=file_path, file_formate="csv")
print(df)