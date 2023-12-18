import numpy as np
from TreeTools import entropy, gini, purity, majority, cut_dataset_by_contious_feature, condition_entropy

test = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
condition = [[0,1,2,3,7,11,13],[4,5,6,8,9,10,12]]
print(entropy(test)-condition_entropy(test, condition))

