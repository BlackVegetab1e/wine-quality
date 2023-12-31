#  此代码对应C45的具体实现

from dataLoader import dataLoader
import numpy as np
from TreeTools import *


class node():
        def __init__(self,node_type, data_X = None, data_Y = None, feature_index = None,
                      cut_feature = None,cut_condition = None,leaf_type = None):
            """决策树的节点，root决策树代表最刚开始的那一个，用来作为初始化
                leaf是叶子节点，叶子节点不包含next，而且有type，能够确定type
                inner是中间节点，中间节点包含有next，并且有一个feature来确定使用哪个来进行判定
                none是无效节点，比如next的节点的初始化等
            """
            assert node_type in ['root','leaf','inner','none']
            assert node_type != 'leaf' or (leaf_type is not None)
            self.type = node_type
            self.leaf_type = leaf_type
            self.cut_condition = cut_condition

            self.data_X = data_X
            self.data_Y = data_Y
            self.feature_index = feature_index
            self.cut_feature = cut_feature


            l_n = None
            r_n = None
            self.next = [l_n,r_n]




class C45():
    def __init__(self, data_X, data_Y, epsilon):
        self.root = node('root', data_X=data_X, data_Y=data_Y, feature_index=range(len(data_X)))

        self.epsilon = epsilon

    
    def generate_tree(self, now_node:node, feature_index:np.ndarray):
        classes_ = np.unique(now_node.data_Y)  
        

        if len(classes_)<=1 :
            # 如果这个节点里面只有一类，这个就被判定为叶子节点

            now_node.type = 'leaf'
            now_node.leaf_type = classes_[0]
            return now_node
        
        if len(feature_index) == 0:
            #  如果现在已经没有可以用来分类的特征了，判定为叶子节点

            now_node.type = 'leaf'
            now_node.leaf_type = majority(now_node.data_Y)
            return now_node

        # 找到使信息增益比最大的feature 以及对应的切分方法，其中切分方法是遍历所有可能得切分点
        best_feature_index = 0
        best_feature_cut = 0
        g_r_max = 0
        g_temp = 0
        for i in feature_index:
            features = now_node.data_X[:,i]
            classes_ = np.unique(features)  

            buffer = np.zeros((len(classes_)-1, 3))
            alpha = 0
            for j in range(len(classes_)-1):
                feature_cut = (classes_[j]+classes_[j+1])/2
                
                g_r, g = g_contious_ratio(now_node.data_X, now_node.data_Y, i, feature_cut)
                buffer[j,0] = g_r
                alpha +=  g
                buffer[j,1] = g
                buffer[j,2] = feature_cut

            if (len(classes_)-1) >0:
                alpha /= (len(classes_)-1)
            else:
                alpha =0

            for j in range(len(classes_)-1):
                if buffer[j,0] > g_r_max and buffer[j,1] > alpha:
                    best_feature_index = i
                    best_feature_cut = buffer[j,2]
                    g_r_max = buffer[j,0]
        
        
        if g_r_max<= self.epsilon:
            # 最大的信息增益还是比较小，太废物了，
            # 已经没有一个很好地分类方案了，直接摆烂输出一个最大值得了
            now_node.type = 'leaf'
            now_node.leaf_type = majority(now_node.data_Y)

            return now_node

        now_node.cut_feature = best_feature_index
        now_node.cut_condition = best_feature_cut

        # 接下来就是切分子集，这边使用递归，二叉树形式
        # 因为连续特征，这里都是对特征进行二分的，所以也只有两个后代

        p_X,p_Y,n_X,n_Y = cut_dataset_by_contious_feature(now_node.data_X, now_node.data_Y, 
                                                          best_feature_index, best_feature_cut)
      
        feature_index = np.delete(feature_index, np.where(feature_index == best_feature_index)[0])



        if len(p_X) == 0:
            L_node = node('leaf', p_X, p_Y, feature_index, leaf_type=majority(now_node.data_Y))
            now_node.next[0] = L_node
        else:
            L_node = node('inner', p_X, p_Y, feature_index)
            now_node.next[0] = self.generate_tree(L_node,feature_index)
            
        if len(n_X) == 0:
            R_node = node('leaf', n_X, n_Y, feature_index, leaf_type=majority(now_node.data_Y))
            now_node.next[1] = R_node
        else:
            R_node = node('inner', n_X, n_Y, feature_index)
            now_node.next[1] = self.generate_tree(R_node,feature_index)


        return now_node
        

        

if __name__ == "__main__":

    data_path = './Data/winequality-white.csv'

    normization_type = 'Standardization'
    data = dataLoader(data_path)
    data.normalization(normization_type)
    training_data, test_data = data.data_cut(10, 9)

    training_x = training_data[:,0:-1]
    training_y = training_data[:,-1]
    
    print(training_y)
    c45 = C45(training_x, training_y, 0.02)
    tree = c45.generate_tree(c45.root, np.array(range(11)))
    

    counter = 0
    for data in test_data:
        now_node = tree
        while now_node.type!='leaf':
            if data[now_node.cut_feature]>=now_node.cut_condition:
                now_node = now_node.next[0]
            else:
                now_node = now_node.next[1]
        if now_node.leaf_type == data[-1]:
            counter+=1

    print(counter/len(test_data))

    







