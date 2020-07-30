from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random  as sparse_random
from sklearn.random_projection import sparse_random_matrix


class SVD(object):
    def __init__(self, n_components=2, alg="randomized", n_iter=5,
                 random_state=None, tol=0.0):
        self.n_components = n_components
        self.alg = alg
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol
        self.svd = self.svd_model()

    def svd_model(self):
        return TruncatedSVD(n_components=self.n_components, n_iter=self.n_iter, random_state=self.random_state)

    def fit(self, input_x, input_y=None):  # 训练
        self.svd.fit(x=input_x, y=input_y)

    def fit_transform(self, input_x, input_y=None):   # 训练并预测
        return self.svd.fit_transform(X=input_x, y=input_y)

    def transform(self, input_x):  # 预测
        return self.svd.transform(X=input_x)

    def inverse_transform(self,input_x):  # 将input_x返回到原来的空间中
        return self.svd.inverse_transform(X=input_x)

    def get_components(self):   # 返回相关参数， 数组，shape（n_components, n_features）
        return self.svd.components_

    def get_explained_variance(self):    # 返回解释性方差，训练样本的方差，  shape(n_components)
        return self.svd.explained_variance_

    def get_explained_variance_ratio(self):   # 返回解释性方差的比例， shape(n_components)
        return self.svd.explained_variance_ratio_

    def get_singular_values(self):    # 对应于每个选定组件的奇异值。奇异值等于低维空间中n_分量变量的2-范数。
        return self.svd.singular_values_   # 获取奇异值


# 测试, 利用TruncatedSVD做文本主题分析
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # Tfidf
# 使用TF-IDF对文本进行预处理，将文本化为向量的表示形式
doc = ["In the middle of the night",
       "when our hopes and fear collide",
       "In the midst of all goodbyes",
       "where all human beings lie",
       "Against another lie"]

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(doc)  # 直接进行变换
# print(x)
terms = vectorizer.get_feature_names()  # 获取特征名
print(terms)

# 使用TruncatedSVD,将原先规模为（文本数，词汇数）的特征矩阵x化为规模为(文本数，主题数)的新特征矩阵x2
# (由于主题数一般比词汇数少，这一方法也可以用来降维，以进行分类或聚类操作)

# 设立主题数为3
n_pick_topics = 3       # 设定主题数为3
lsa = SVD(n_pick_topics)   #
x2 = lsa.fit_transform(x)  # 进行训练
print(x2)    # 转成  5*3的向量

# x2[i,t]为第i篇文档在第t个主题上的分布，所以该值越高的文档i，可以认为在主题t上更有代表性，我们便以此筛选出
# 最能代表该主题的文档

n_pick_docs = 2   # 获取每个主题中的top2个主题
topic_docs_id = [x2[:, t].argsort()[:-(n_pick_docs+1):-1] for t in range(n_pick_topics)]
print(topic_docs_id)


# lsa.components_  为规模为（主题数，词汇数）的矩阵，其（t,j）位置的元素代表了词语j在主题t上的权重，同样以此获得
# 主题关键词
n_pick_keywords = 4
topic_keywords_id = [lsa.get_components()[t].argsort()[:-(n_pick_keywords+1):-1]for t in range(n_pick_topics)]
print(topic_keywords_id)

for t in range(n_pick_topics):
    print("topic %d:" % t)
    print(" keywords: %s"%",".join(terms[topic_keywords_id[t][j]] for j in range(n_pick_keywords)))
    for i in range(n_pick_docs):
        print("        doc %d" % i)
        print("\t" + doc[topic_docs_id[t][i]])

# if __main__ == "__name__":