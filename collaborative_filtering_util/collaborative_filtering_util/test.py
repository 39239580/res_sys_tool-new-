import numpy as np
from collaborative_filtering_util.distance_sim import cal_similar, jaccard_dist, jaccard_sim
from numpy import mat


p = [1, 1, 0, 1, 0, 1, 0, 0, 1]
q = [0, 1, 1, 0, 0, 0, 1, 1, 1]
d = [1, 1, 1, 1, 1, 1, 0, 1, 1]
# p 与q  p u q 并集为  8个1，  p n q并集为 2个1    所以相似距离为为1-2/8=0.75
# p 与d  p u d 并集为  8个1，  p n q交集为 5个1    所以相相似距离为1-5/8=0.375
# q 与d  q u d 并集为  9个1，  p n q交集为 4个1    所以相似距离为 1-4/9 = 0.444444

r = mat([p, q, d])  # 变成矩阵的形式
d = np.array([p, q, d])  # 变成数组的形式

# print(cal_similar(cal_type="pearson", x=r))
print(cal_similar(cal_type="pearson", x=d))
f=cal_similar(cal_type="pearson", x=d)

# f.sort(axis=1)
# print(f)

print(np.argsort(-f))
