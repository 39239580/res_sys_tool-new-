# 局部敏感hash
# 通过使用numpy数组对大量高维数据进行快速哈希计算。
# 内置支持通过redis的持久性。
# 支持多个哈希索引。
# 内置支持通用距离 / 目标函数，用于对输出进行排序。

from lshash.lshash import LSHash

class LocalSensitiveHash(object):
    def __init__(self, hash_size, input_dim, num_of_hashtables=1, storage=None, matrices_filename=None, overwrite=False):
        """
        Attributes:
        :param hash_size:
            The length of the resulting binary hash in integer.E.g., 32 means the resulting binary hash will be 32 - bit long.

        :param input_dim:
            The dimension of the input vector.E.g., a grey - scale picture of 30x30 pixels will have an input dimension of 900.

        :param num_hashtables:
            (optional) The number of hash tables used for multiple lookups.

        :param storage_config:
            (optional) A dictionary of the form `{backend_name: config}` where `backend_name` is the either `dict` or `redis`,
            and `config` is the configuration used by the backend.
            For `redis`it should be in the format of`{"redis": {"host": hostname, "port": port_num}}`,
            where `hostname` is normally `localhost` and `port` is normally 6379.

        :param matrices_filename:
            (optional) Specify the path to the compressed numpy file endin with extension `.npz`, where the uniform random planes
            are stored, or to be stored if the file does not exist yet.

        :paramoverwrite:
            (optional) Whether to overwrite the matrices file if it already exist
        """
        self.hash_object = LSHash(hash_size=hash_size,   # 二进制hash  结果的长度
                                  input_dim=input_dim,   # 输入向量的维度
                                  num_of_hashtables=num_of_hashtables,   # 用于多次查找的哈希表的数目。可选项
                                  storage=storage, # (可选)指定用于索引存储的存储的名称。选项包括“redis”
                                  matrices_filename=matrices_filename,  # (可选)指定.npz文件的路径随机矩阵被存储, 如果文件不存在
                                  overwrite=overwrite)  # 如果matrices文件存在，是否对其进行覆盖， 可选项

    # 从给定的局部敏感hash实例中索引数据点
    def lsh_index(self, input_point, extra_data=None):
        """
        :param input_point:  为一个数组或远祖，大小为input_dim维度
        :param extra_data: 可选项，附加数据将与input_point一起添加。
        :return:
        """
        self.hash_object.index(input_point=input_point, extra_data=extra_data)

    # 根据给定的LSHash 实例检索一个数据点
    def lsh_query(self, query_point, num_results=None, distance_fun="euclidean"):
        assert distance_fun in {"hamming", "euclidean", "true_euclidean", "centred_euclidean", "cosine", "l1norm"}
        """
        :param query_point:  检索的数据殿是一个数组或元组，大小为input_dim
        :param num_results:  # (可选)按顺序返回的查询结果的数量。默认情况下，将返回所有结果。
        :param distance_fun: # （可选）排序距离函数用于排序候选集， 默认使用的欧氏距离
        距离可使用的参数
        ("hamming",   汉明距离
         "euclidean",  欧式距离
         "true_euclidean", 真欧式距离
         "centred_euclidean",  中心欧式距离
         "cosine",  余弦距离
         "l1norm") l1 正则化
        :return:
        """
        return self.hash_object.query(query_point=query_point, num_results=num_results, distance_func=distance_fun)


#  -----------------------------------test------------------------------------------------------
def test_lshash():
    lsh = LSHash(6, 8)   # 对于输入数据为8维的数据创建6位hash
    lsh.index([1, 2, 3, 4, 5, 6, 7, 8])
    lsh.index([2, 3, 4, 5, 6, 7, 8, 9])
    lsh.index([10, 12, 99, 1, 5, 31, 2, 3])
    print(lsh.query([1, 2, 3, 4, 5, 6, 7, 7]))


if __name__ == "__main__":
    test_lshash()
