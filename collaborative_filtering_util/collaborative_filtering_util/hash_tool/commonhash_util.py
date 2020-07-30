from datasketch import MinHash, WeightedMinHashLSH, LeanMinHash, MinHashLSH
from datasketch import WeightedMinHashLSHForest, MinHashLSHForest
from datasketch import WeightedMinHash, HyperLogLog, HyperLogLogPlusPlus
from datasketch import MinHashLSHEnsemble
from datasketch.hashfunc import sha1_hash32, sha1_hash64
import farmhash
import xxhash
import mmh3
import numpy as np


# 相关文档说明https://pypi.org/project/datasketch/
class CommonHash(object):
    def __init__(self, hash_type=None, bits=None,hash_func=None, params=None):
        self.hash_type = hash_type
        self.hash_func = hash_func
        self.hash_bits = bits
        self.hashfunc = sha1_hash32
        if self.hash_bits in {32, "32", None}:
            if self.hash_func == "mmh3":
                self.hashfunc = mmh3.hash
            elif self.hash_func == "farmhash":
                self.hashfunc = farmhash.hash32
            elif self.hash_func == "xxhash":
                self.hashfunc = xxhash.xxh32
            else:
                # "hash32","default":
                self.hashfunc = sha1_hash32

        elif self.hash_bits in {64, "64"}:
            if self.hash_func == "mmh3":
                self.hashfunc = mmh3.hash64
            elif self.hash_func == "farmhash":
                self.hashfunc = farmhash.hash64
            elif self.hash_func == "xxhash":
                self.hashfunc = xxhash.xxh64
            else:
                self.hashfunc = sha1_hash64

        elif self.hash_bits in {128, "128"}:
            if self.hash_func == "mmh3":
                self.hashfunc = mmh3.hash128
            elif self.hash_func == "farmhash":
                self.hashfunc = farmhash.hash128
            else:
                raise ValueError("请检查对应的hash函数类型与位数")

        else:
            raise ValueError("请检查对应的hash函数的位数")

        if not params:
            params = {}

        """
        若只用redis 作为存储截止
        配置
        storage_config={  
        'type': 'redis',
        'redis': {'host': 'localhost', 'port': 6379},
        }
                
        要顺序插入大量MinHash，建议使用插入会话。这样可以减少批量插入过程中的网络呼叫数量。
        data_list = [("m1", m1), ("m2", m2), ("m3", m3)]
        with lsh.insertion_session() as session:
            for key, minhash in data_list:
                session.insert(key, minhash)
        请注意，在打开插入会话期间查询LSH对象可能会导致不一致。
        
        MinHash LSH还支持Cassandra群集作为存储层。为您的LSH使用长期存储可解决应用程序需要不断更新LSH对象的所有用例（例如，
        当您使用MinHashLSH逐步对文档进行群集时）。
        Cassandra存储选项可以配置如下：
        
         storage_config={
        'type': 'cassandra',
        'cassandra': {
            'seeds': ['127.0.0.1'],
            'keyspace': 'lsh_test',
            'replication': {
                'class': 'SimpleStrategy',
                'replication_factor': '1',
            },
            'drop_keyspace': False,
            'drop_tables': False,
        }}
        参数Seeds指定可以联系以连接到Cassandra集群的种子节点列表。选项键空间和复制指定创建键空间（如果尚不存在）时要使用的参数。
        如果要强制创建表或键空间（因此要删除现有表或键空间），请将drop_tables和drop_keyspace选项设置为 True。
        像Redis副本一样，建议使用插入会话来减少批量插入期间的网络调用数量。
        
        +-----------------------连接到现有的最小哈希LSH-------------------------------------+ 
        如果您的LSH使用外部存储层（例如Redis），则可以跨多个进程共享它。有两种方法可以做到这一点：
        
        推荐的方法是使用“酸洗”。MinHash LSH对象是可序列化的，因此您可以调用pickle：
        
        import pickle
        
        # Create your LSH object
        lsh = ...
        # Serialize the LSH
        data = pickle.dumps(lsh)
        # Now you can pass it as an argument to a forked process or simply save it
        # in an external storage.
        
        # In a different process, deserialize the LSH
        lsh = pickle.loads(data)
        使用pickle，您可以保存有关LSH所需的所有知识，例如在一个位置中进行各种参数设置。
        另外，您可以在首次创建LSH时在存储配置中指定基本名称。例如：

        # For Redis.
        lsh = MinHashLSH(
            threshold=0.5, num_perm=128, storage_config={
                'type': 'redis',
                'basename': b'unique_name_6ac4fg',
                'redis': {'host': 'localhost', 'port': 6379},
            }
        )
        
         # For Cassandra.
         lsh = MinHashLSH(
            threashold=0.5, num_perm=128, storage_config={
                'type': 'cassandra',
                'basename': b'unique_name',
                'cassandra': {
                    'seeds': ['127.0.0.1'],
                    'keyspace': 'lsh_test',
                    'replication': {
                        'class': 'SimpleStrategy',
                        'replication_factor': '1',
                    },
                    'drop_keyspace': False,
                    'drop_tables': False,
                }
            }
        )
        的基名将用于生成在所述存储层中唯一地标识与该LSH相关联的数据键前缀。因此，如果使用相同的基名创建新的LSH对象，则将在与旧LSH关联的存储层中使用相同的基础数据。
        
        如果不指定basename，则MinHash LSH将生成一个随机字符串作为基本名称，并且极不可能发生冲突。
        
        更详细的使用见 文档 ：http://ekzhu.com/datasketch/lsh.html
        """

        if self.hash_type in {"minhash","MinHash"}:
            # 主要计算Jaccard 的相似度， 使用较小的固定存储空间来估计线性时间内任意大小的集合之间的jaccard 相似度
            self.hash = MinHash(num_perm=params.get("num_perm", 128),   # int可选项， 如果hashvalues值不是None,则被忽略。随机排列函数的数量
                                # 用来控制hash 的精度
                                seed=params.get("seed", 1),  # 随机种子 可选
                                hashfunc=self.hashfunc,   # 可选 使用的hash函数，将输入传递给update 方法。并返回一个可以用32位编码的整数
                                hashobj=params.get("hashobj", None),  # Deprecated.已经被hashfunc 代替
                                hashvalues=params.get("hashvalues", None),  # 可选 数组或列表
                                permutations=params.get("permutations", None))   # 置换函数参数， 可选，可使用另一个Minhash 的现有状态来指定此参数进行快速的初始化
        elif self.hash_type in {"weightedminhashlsh", "mhlsh", "WeightedMinHashLSH","wmhlsh","MinHashLSH"}:  # 加权的最小哈希局部敏感哈希
            #  WeightedMinHashLSH()   与 MinHashLSH 等价  。 加权jaccard 相似度 查询
            # 不支持top-k查询， 但minhashlshforest 支持top-k
            self.hash = MinHashLSH(threshold=params.get("threshold", 0.9),  # 杰卡德距离的阈值
                                   num_perm=params.get("num_perm", 128),  # 置换函数设定个数， 在加权minihash 上的 样本规模大小
                                   weights=params.get("weights", (0.5, 0.5)),  # 元组， 可选项， 优化jaccard阈值
                                   params=params.get("params",None),  # 元组，可选项， – bands 的数量与规模大小
                                   storage_config=params.get("storage_config",None),  # 存储配置
                                   prepickle=params.get("prepickle",None))   # 默认使用pk格式存储
        elif self.hash_type in {"leanminhash","lmh","LeanMinHash","LMH"}:
            # 相比MinHash 中，内存更小的哈希。
            self.hash = LeanMinHash(minhash=params.get("minhash",None),
                                    seed=params.get("seed",None),
                                    hashvalues=params.get("hashvalues",None))

        elif self.hash_type in {"MinHashLSHForest", "minhashlshforest","mhlshf", "MHLSHF"}:
            self.hash = MinHashLSHForest(num_perm=params.get("num_perm",128),
                                         l=params.get("l",8))

        elif self.hash_type in {"MinHashLSHEnsemble","MinHashLSHEnsemble","mhlshe","MHLSHE"}:
            # 使用新距离做的minhashlsh操作 ， 即使用Containment 中文简称为遏制
            self.hash = MinHashLSHEnsemble(threshold=params.get("threshold",0.9),
                                           num_perm=params.get("num_perm",128),
                                           num_part=params.get("num_part",16),   #
                                           m=params.get("m",8),
                                           weights=params.get("weights",(0.5,0.5)),
                                           storage_config=params.get("storage_config",None),
                                           prepickle=params.get("prepickle", None))

        elif self.hash_type in {"HyperLogLog", "hyperloglog","hll", "HLL"}:
            # 相关的接口与HyperLogLog 相同
            # HyperLogLog能够使用较小且固定的内存空间，单次估算数据集的基数（不同值的数量）
            self.hash = HyperLogLog(p=params.get("p",8),  #  与MinHash 中的数据相比较，num_perm  用于控制精度
                                    reg=params.get("reg",None),
                                    hashfunc=params.get("hashfunc", sha1_hash32),   # 内部使用的hash 算法
                                    hashobj=params.get("hashobj",None))  # 可选 数组或列表，  使用hashfunc 代替了

        elif self.hash_type in {"hyperloglogplusplus", "HyperLogLogPlusPlus", "HyperLogLog++", "hyperlogkog++",
                                "HLLPP", "hllpp","HLL++","hll++"}:
            # 相关的接口与HyperLogLog 相同
            self.hash = HyperLogLogPlusPlus(p=params.get("p",8),
                                            reg=params.get("reg",None),
                                            hashfunc=params.get("hashfunc",sha1_hash64),  # 使用的64位的hash 算法
                                            hashobj=params.get("hashobj",None))

        else:
            raise ValueError("请选择正确的函数函数对象")

    def update(self, b):
        assert self.hash_type in {"minhash", "MinHash", "leanminhash", "lmh", "LeanMinHash",
                                  "weightedminhashlsh","mhlsh","WeightedMinHashLSH","wmhlsh","MinHashLSH",
                                  "MinHashLSHEnsemble","minhashlshensemble","MHLSHE","mhlshe",
                                  "HyperLogLog", "hyperloglog","hll", "HLL","HyperLogLog", "hyperloglog","hll", "HLL"}
        # 用新值更新此MinHash
        self.hash.update(b=b)

    def jaccard_distance(self, other):
        # 杰卡德相似度
        # print(other.seed())
        # print(self.hash.seed)
        return self.hash.jaccard(other.hash)

    def count(self):
        # 由这个MinHash表示的集合的估计基数。
        return self.hash.count()

    def merge(self, other):
        # 两个Minhash合并。
        self.hash.merge(other=other.hash)

    def digest(self):
        # 导出hash值,Minhash 中返回数组
        return self.hash.digest()

    def is_emty(self):
        # minhash 中的状态，判断是否为空
        return self.hash.is_empty()

    def clear(self):
        # 重置所有hash
        self.hash.clear()

    def copy(self):
        # 返回当前minhash 的副本
        return self.hash.copy()

    def lens(self):
        # 返回哈希值的数量
        return self.hash.__len__()

    def eq(self, other):
        # 判断两个mInhash  的种子和哈希值是都相等
        return self.hash.__eq__(other=other)

    def union(self, lmhs):
        # 两个minhash 进行融合。 参数列表必须为2以上
        return self.hash.union(*lmhs)

    def seed(self):
        # print(self.hash.seed)
        return self.hash.seed

    def insert(self, hash_name, hash_value):
        self.hash.insert(hash_name, hash_value.hash)

    def query(self, hash_value):
        return self.hash.query(hash_value.hash)

    def query_forest(self, hash_value, k):
        return self.hash.query(minhash=hash_value.hash, k=k)

    def query_ensemble(self, hash_value, size):
        return self.hash.query(minhash=hash_value.hash, size=size)

    def remove(self, hash_name):
        self.hash.remove(hash_name)

    def add(self, hash_name, hash_value):
        self.hash.add(key=hash_name, minhash=hash_value.hash)

    def index_forest(self):
        self.hash.index()
        return self
        # self.hash
        # self.hash.index()

    def index_ensemble(self,entries):
        # self.hash.index()
        self.hash.index(entries)
        return self


def minhash_test():
    data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
             'estimating', 'the', 'similarity', 'between', 'datasets']
    data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
             'estimating', 'the', 'similarity', 'between', 'documents']

    m1, m2 = CommonHash("MinHash",params={"seed":1}), CommonHash("MinHash",params={"seed":1})
    for d in data1:
        m1.update(d.encode("utf8"))
    # print(m1.seed())
    for d in data2:
        m2.update(d.encode("utf8"))
    # print(m2.seed())
    print("Estimated Jaccard for data1 and data2 is:", m1.jaccard_distance(m2))

    s1 = set(data1)
    s2 = set(data2)
    actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))   # 交集与并集
    print("Actual Jaccard for data1 and data2 is", actual_jaccard)

    # 大致步骤， MinHash初始化状态 ， 需要预先设定MinHash() 初始化
    # 内容hash化。 update  哈稀化
    # jaccard 距离


def minhash_test2():
    data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
             'estimating', 'the', 'similarity', 'between', 'datasets']
    data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
             'estimating', 'the', 'similarity', 'between', 'documents']

    m1, m2 = CommonHash("MinHash", params={"num_perm":512}), CommonHash("MinHash", params={"num_perm":512})
    for d in data1:
        m1.update(d.encode("utf8"))
    # print(m1.seed())
    for d in data2:
        m2.update(d.encode("utf8"))
    # print(m2.seed())
    print("Estimated Jaccard for data1 and data2 is:", m1.jaccard_distance(m2))

    s1 = set(data1)
    s2 = set(data2)
    actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))   # 交集与并集
    print("Actual Jaccard for data1 and data2 is", actual_jaccard)


def minhash_test3():
    data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
             'estimating', 'the', 'similarity', 'between', 'datasets']
    data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
             'estimating', 'the', 'similarity', 'between', 'documents']

    m1, m2 = CommonHash("MinHash", params={"num_perm": 512}), CommonHash("MinHash", params={"num_perm": 512})
    m1.merge(m2)
    print("Estimated Jaccard for data1 and data2 is:", m1.jaccard_distance(m2))

    s1 = set(data1)
    s2 = set(data2)
    actual_jaccard = float(len(s1.intersection(s2))) / float(len(s1.union(s2)))  # 交集与并集
    print("Actual Jaccard for data1 and data2 is", actual_jaccard)


def minhash_test4():
    data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for', 'estimating', 'the', 'similarity', 'between', 'datasets']
    data2 = ['minhash', 'is', 'probability', 'data', 'structure', 'for','estimating', 'the', 'similarity', 'between', 'documents']
    data3 = ['minhash', 'is', 'probability', 'data', 'structure', 'for','estimating', 'the', 'similarity', 'between', 'documents']
    set1 = set(data1)
    set2 = set(data2)
    set3 = set(data3)
    m1 = CommonHash("MinHash")
    m2 = CommonHash("MinHash")
    m3 = CommonHash("MinHash")
    for d in set1:
        m1.update(d.encode('utf8'))
    for d in set2:
        m2.update(d.encode('utf8'))
    for d in set3:
        m3.update(d.encode('utf8'))

    # Create LSH index
    print()
    lsh = CommonHash("MinHashLSH", params={"threshold":0.5})
    lsh.insert("m2", m2)
    lsh.insert("m3", m3)
    result = lsh.query(m1)
    print("Approximate neighbours with Jaccard similarity > 0.5", result)
    # MinHash初始化状态， 需要预先设定MinHash(threshold=0.5, num_perm=128)  初始化状态
    # 内容hash化， m1.update 哈希化
    # 内容载入LSH系统， lsh.insert("m3", m3),  其中insert(hash名称， minHash值)
    # 查询， lsh.query, 其中查询的内容也必须minHash化
    # 同时， MinHash 不能采用多项内容Top-k查询， 可以使用之后的MinHash Forest. Jaccard 距离可能不是最好的距离，
    # 见MinHash LSH Ensemble.
    lsh.remove("m2")


def minhash_test5():
    data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
             'estimating', 'the', 'similarity', 'between', 'datasets']
    data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
             'estimating', 'the', 'similarity', 'between', 'documents']
    data3 = ['minhash', 'is', 'probability', 'data', 'structure', 'for',
             'estimating', 'the', 'similarity', 'between', 'documents']
    # 创建hash 对象
    m1 = CommonHash("MinHash")
    m2 = CommonHash("MinHash")
    m3 = CommonHash("MinHash")

    for d in data1:
        m1.update(d.encode('utf8'))
    for d in data2:
        m2.update(d.encode('utf8'))
    for d in data3:
        m3.update(d.encode('utf8'))

    # Create a MinHash LSH Forest with the same num_perm parameter
    forest = CommonHash("MinHashLSHForest")

    # 添加m2,m3 到index中
    forest.add("m2", m2)
    forest.add("m3", m3)

    # 必须调用index(), 否则不能进行搜索
    forest.index_forest()
    print(forest.hash)

    # 使用关键字检查关系
    print("m2" in forest.hash)
    print("m1" in forest.hash)

    # 使用m1 作为检索 top-2 的最近的jaccard

    result = forest.query_forest(m1, 2)
    print("Top 2 candidates", result)
    # MinHash初始化状态，需要预先设定MinHash(num_perm=128)
    # 初始化状态
    # 内容哈希化，内容m1.update哈希化
    # MinHashLSHForest初始化， MinHashLSHForest(num_perm=128)
    # 内容载入投影森林之中，forest.add(“m2”, m2)
    # forest.index()，相当于update一下，更新一下
    # 查询，lsh.query，其中查询的内容也必须minHash化。


def minhash_test6():
    data1 = ["cat", "dog", "fish", "cow"]
    data2 = ["cat", "dog", "fish", "cow", "pig", "elephant", "lion", "tiger", "wolf", "bird", "human"]
    data3 = ["cat", "dog", "car", "van", "train", "plane", "ship", "submarine", "rocket", "bike", "scooter", "motorcyle", "SUV", "jet", "horse"]
    set1 = set(data1)
    set2 = set(data2)
    set3 = set(data3)

    # Create MinHash objects
    m1 = CommonHash("MinHash")
    m2 = CommonHash("MinHash")
    m3 = CommonHash("MinHash")
    for d in set1:
        m1.update(d.encode('utf8'))
    for d in set2:
        m2.update(d.encode('utf8'))
    for d in set3:
        m3.update(d.encode('utf8'))

    # Create an LSH Ensemble index with threshold and number of partition
    # settings.
    lshensemble = CommonHash("mhlshe", params={"threshold": 0.8, "num_perm": 128, "num_part": 32})

    # Index takes an iterable of (key, minhash, size)
    lshensemble.index_ensemble([("m2", m2.hash, len(set2)), ("m3", m3.hash, len(set3))])

    # Check for membership using the key
    print("m2" in lshensemble.hash)
    print("m3" in lshensemble.hash)

    # Using m1 as the query, get an result iterator
    print("Sets with containment > 0.8:")
    for key in lshensemble.query_ensemble(hash_value=m1, size=len(set1)):
        print(key)

def minhash_test7():
    data1 = ['hyperloglog', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
             'estimating', 'the', 'cardinality', 'of', 'dataset', 'dataset', 'a']

    h = CommonHash("HyperLogLog")
    for d in data1:
        h.update(d.encode('utf8'))
    print("Estimated cardinality is", h.count())

    s1 = set(data1)
    print("Actual cardinality is", len(s1))  # 一种数据几和


def minhash_test8():
    h1 = CommonHash("HyperLogLog")
    h2 = CommonHash("HyperLogLog")
    h1.update('test'.encode('utf8'))
    # The makes h1 the union of h2 and the original h1.
    h1.merge(h2)
    # This will return the cardinality of the union
    print(h1.count())


if __name__ == "__main__":
    minhash_test()
    minhash_test2()   # 提高精度
    minhash_test3()
    minhash_test4()
    minhash_test5()
    minhash_test6()
    minhash_test7()
    minhash_test8()
