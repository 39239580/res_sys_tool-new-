import pandas as pd
import pickle as pk


# 通用的数据读取工具
def read_data_from_file(file_type, file_path,
                        encoding="utf-8", header=None,
                        index_col=None, names=None,
                        sep=","):  # 以某某方式进行读取

    if file_type == "csv":
        df = pd.read_csv(file_path, encoding=encoding, header=header, index_col=index_col, names=names, sep=sep)
    elif file_type == "xlsx":
        df = pd.read_excel(file_type, engine=encoding, header=header, index_col=index_col, name=names, sep=sep)
    else:
        ValueError("file_type must be csv or xlsx")
    return df


def load_pk(file_path):   # 加载相应的pk 文件
    with open(file_path, "rb") as f:
        file = pk.load(f)
    return file


def save_pk(file, file_path):  # 保存相应的pk 文件
    with open(file_path, "wb") as f:
        pk.dump(file, f)
