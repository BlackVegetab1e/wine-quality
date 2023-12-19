import numpy as np
from TreeTools import g_discrete,g_discrete_ratio,g_contious_ratio,g_contious,entropy, gini, purity, majority, cut_dataset_by_contious_feature, condition_entropy

test = np.array([0,1,0,1,1,0,1,0,1,1,1,1,1,0])
# condition = [[0,1,2,3,7,11,13],[4,5,6,8,9,10,12]]
# print(entropy(test)-condition_entropy(test, condition))
age = np.array([[1,1,3,3,3,1,3,1,3,3,3,3,3,1],[3,3,3,2,1,1,1,2,1,2,2,2,3,2], [0,0,0,0,1,1,1,0,1,1,1,0,1,0]])

# print(g_contious(age, test, 0,2.5))
for i in range(100):
    print(g_contious_ratio(age, test, i%4,2.5))