import numpy as np

def g_discrete(X: np.ndarray, Y: np.ndarray, A: int):
    target_feature = X[A, :]
    classes_ = np.unique(target_feature)  
    N = len(Y)
    g = 0.0
    for i in range(len(classes_)):
        index = np.where(target_feature == classes_[i])[0]
        g += entropy(Y[index])*len(index)/N
    return entropy(Y) - g 


def g_discrete_ratio(X: np.ndarray, Y: np.ndarray, A: int):
    g = g_discrete(X,Y,A)
    g_ratio =  g / entropy(X[A, :])
    return g_ratio


def g_contious(X: np.ndarray, Y: np.ndarray, A: int, condition: float):

    target_feature = X[:, A]
    N = len(Y)
    g = 0.0
    index_positive = np.where(target_feature >= condition)[0]
    index_negative = np.where(target_feature < condition)[0]
    g += entropy(Y[index_positive])*len(index_positive)/N
    g += entropy(Y[index_negative])*len(index_negative)/N
    return entropy(Y) - g 



def g_contious_ratio(X: np.ndarray, Y: np.ndarray, A: int, condition):
    
                
    g = g_contious(X, Y, A, condition)
    
    target_feature = X[:, A]
    N = len(Y)
    r_positive = len(np.where(target_feature >= condition)[0])/N
    r_negative = len(np.where(target_feature < condition)[0])/N
    if r_positive == 0:
        r_positive = 1
    if r_negative == 0:
        r_negative = 1
    entropy_A = -np.log2(r_positive)*(r_positive) - np.log2(r_negative)*(r_negative) 
    g_ratio =  g / entropy_A
    return  g_ratio, g



def entropy(Y: np.ndarray) -> float:
    """输入类别标签集y_，输出信息熵"""
    
    N = len(Y)   
        
    classes_,N_ = np.unique(Y, return_counts=True)  
    
    p_ = N_/N 
    s = -np.log2(p_).dot(p_) 

    
    return s


def condition_entropy(Y: np.ndarray, condition) -> float:
    """输入类别标签集y_,输入满足条件的index，输出条件信息熵,输入的conditions是数组，里面包含各种情况的index"""
    result = 0.0
    N = len(Y)     
    for i in range(len(condition)):
        result += entropy(Y[condition[i]])*len(condition[i])/N
        print(entropy(Y[condition[i]]))
    return result




def gini(Y: np.ndarray) -> float:
    """输入类别标签集y_，输出基尼指数gini"""

    N = len(Y)   
    classes_,N_ = np.unique(Y, return_counts=True) 
    p_ = N_/N 
    gini = 1 - sum(p_**2)
    return gini

def gini_index(X: np.ndarray, Y: np.ndarray, A: int, condition) -> float:
    """输入类别标签集y_，输出基尼指数gini"""

    target_feature = X[:, A]
    N = len(Y)
    gini_sum = 0.0
    index_positive = np.where(target_feature >= condition)[0]
    index_negative = np.where(target_feature < condition)[0]
    gini_sum += gini(Y[index_positive])*len(index_positive)/N
    gini_sum += gini(Y[index_negative])*len(index_negative)/N
    return gini_sum



def purity(Y: np.ndarray) -> float:
    """输入标签集y_，输出最大值的纯度"""
    N = len(Y)
    classes_ = np.unique(Y)
    K = len(classes_)
    N_ = np.zeros(K)
    for k, c1ass in enumerate(classes_):
        N_[k] = sum(Y==c1ass)
    p_ = N_/N
    return p_.max()

def majority(Y: np.ndarray) -> np.ndarray:
    """输入标签集y_，输出数量最多的""" 
    classes_ = np.unique(Y)  
    K = len(classes_)
    N_ = np.zeros(K)
    for k, c1ass in enumerate(classes_):
        N_[k] = sum(Y==c1ass) 
    majorityClass = classes_[N_.argmax()] 
    return majorityClass





def cut_dataset_by_contious_feature(X: np.ndarray, Y: np.ndarray,   # 样本
        A: int,        # 用哪个特征来分隔
        condition: float,      # 在哪里分割
        ):
    target_feature = X[:, A]
    N = len(Y)

    index_positive = np.where(target_feature >= condition)[0]
    index_negative = np.where(target_feature < condition)[0]

    return X[index_positive], Y[index_positive],X[index_negative], Y[index_negative]

