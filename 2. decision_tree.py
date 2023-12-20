from dataLoader import dataLoader
from C45 import C45
from CART import CART
import numpy as np



    
def test_correct_rate(tree, test_data):
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
    return counter/len(test_data)


if __name__ == "__main__":

    data_path = './Data/winequality-white.csv'
    normization_type = 'Standardization'
    data = dataLoader(data_path)                          # 导入数据 
    data.normalization(normization_type)                  # 标准化 
    C45_corr = np.zeros((10,))
    CART_corr = np.zeros((10,))

    for i in range(10):

        training_data, test_data = data.data_cut(10, i)       # 十折交叉验证 
        training_x = training_data[:,0:-1]
        training_y = training_data[:,-1]
        # c45决策树   
        c45 = C45(training_x, training_y, 0.02)         
        tree_c45 = c45.generate_tree(c45.root, np.array(range(11)))
        C45_corr[i] = test_correct_rate(tree_c45, test_data)



        # cart决策树
        cart = CART(training_x, training_y, 15)
        tree_cart = cart.generate_tree(cart.root, np.array(range(11)))
        CART_corr[i] = test_correct_rate(tree_cart, test_data)
        
    
    print(C45_corr)
    print(C45_corr.mean())
    print(CART_corr)
    print(CART_corr.mean())

    


