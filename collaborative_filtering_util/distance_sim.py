from scipy.spatial.distance import pdist,squareform
from numpy import mat
import numpy as  np

from sklearn.metrics.pairwise import euclidean_distances  # 欧氏距离
from sklearn.metrics.pairwise import cosine_similarity  # 余弦距离 余弦相似度
from scipy.stats import pearsonr   # 皮尔逊系数

# p, q  为 列表
def jaccard_dist(p,q,data_type=None):   # 杰卡德距离 ,一种计算方式
    if data_type == "mat":
        dis = pdist(mat([p,q]), "jaccard")
    elif data_type == "array":
        dis = pdist(np.array([p,d]),"jaccard")
    else:
        dis = pdist([p,d], "jaccard")
    return dis

def jaccard_sim(p,q,data_type=None):   #  杰卡德相似度，　支持列表，数组，矩阵
    """
    :param p:
    :param q:
    :return:
    """
    if data_type == "mat":
        dis = pdist(mat([p,q]), "jaccard")
    elif data_type == "array":
        dis = pdist(np.array([p,d]),"jaccard")
    else:
        dis = pdist([p,d], "jaccard")

    sim = 1- dis
    return sim

def jaccard_dist_matrix(matrix):
    dis = pdist(matrix, "jaccard")
    return dis

def jaccard_sim_matrix(matrix):
    # print(matrix)
    dis = pdist(matrix, "jaccard")
    dis = squareform(dis)
    sim = 1 - dis
    # sim=squareform(sim)
    return sim

def pearson_sim(x, y):  # 计算两个数组的相似度
    """   支持  x 为列表或者数组， 但不支持mat
    :param x:
    :param y:
    :return:
    """
    sim=pearsonr(x=x, y=y)
    return sim

# 样本之间的相似度, 即支持列表之间的相似度计算， 也支持 mat的计算，　也支持数组之间的相似度
def cal_similar(cal_type, x, y=None):
    """
    # 支持 mat 和数组的输入格式
    :param cal_type:
    :param x:
    :param y:
    :return:
    """
    if cal_type == "euclidean":  # 欧氏相似度
        sim = 1/(euclidean_distances(X=x,Y=y)+1)

    elif cal_type == "cosin":  # 余弦相似度   越接近1 ， 越相似
        sim = cosine_similarity(X=x, Y=y)  #

    elif cal_type == "pearson":  # 皮尔逊系数   越接近1 ， 则越相似
        sim = np.corrcoef(x=x, y=y)
    else:
        sim = jaccard_sim_matrix(matrix=x)  # 杰卡德相似度  越接近1， 月相似
    return sim



if __name__ == "__main__":
    p = [1, 1, 0, 1, 0, 1, 0, 0, 1]
    q = [0, 1, 1, 0, 0, 0, 1, 1, 1]
    d = [1, 1, 1, 1, 1, 1, 0, 1, 1]
    # p 与q  p u q 并集为  8个1，  p n q并集为 2个1    所以相似距离为为1-2/8=0.75
    # p 与d  p u d 并集为  8个1，  p n q交集为 5个1    所以相相似距离为1-5/8=0.375
    # q 与d  q u d 并集为  9个1，  p n q交集为 4个1    所以相似距离为 1-4/9 = 0.55555556
    print(jaccard_dist(p,q))   #　杰卡德距离
    print(jaccard_sim(p,q))  # 杰卡德相似度

    r= mat([p,q, d])  # 变成矩阵的形式
    d = np.array([p,q,d])  # 变成数组的形式
    print(r)

    print(pdist(r, "jaccard"), type(pdist(r, "jaccard")))   # 输出的均为数组
    print(pdist(d, "jaccard"), type(pdist(d, "jaccard")))
    print(jaccard_dist_matrix(r), jaccard_sim_matrix(r))  # 杰卡得距离
    print(jaccard_dist_matrix(d), jaccard_sim_matrix(d))
    print("--------------------------------------")

    x = [0.5, 0.4, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5]
    y = [0.6, 0.4, 0.4, 0.3, 0.7, 0.2, 0.5, 0.6]
    x_ = np.array(x)
    y_ = np.array(y)
    print(pearson_sim(x, y))    # 皮尔逊相似度
    print(pearson_sim(x_, y_))  # 皮尔逊相似度


    #
    print(cal_similar(cal_type="euclidean", x=r))   #
    print(cal_similar(cal_type="euclidean", x=d))
    print(cal_similar(cal_type="cosin", x=r))
    print(cal_similar(cal_type="cosin", x=d))
    print(cal_similar(cal_type="pearson", x=r))
    print(cal_similar(cal_type="pearson", x=d))
    print(cal_similar(cal_type="jaccard", x=r))
    print(cal_similar(cal_type="jaccard", x=d))
    # print(squareform(cal_similar(cal_type="jaccard", x=d)))




    # print(type(cal_similar(cal_type="pearson", x=d)))
    # print("-------------------")
    # d= cal_similar(cal_type="pearson", x=d)
    # print(d)
    # print(d.tolist())
    # f= d.tolist()
    # for i in f:
    #     for j in i:
    #         print(j)


    # print(np.corrcoef(r))
