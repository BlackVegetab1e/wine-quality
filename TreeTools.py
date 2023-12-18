import numpy as np


def g(X,Y,A):
    pass



def entropy(Y: np.ndarray) -> float:
    """输入类别标签集y_，输出信息熵"""
    N = len(Y)          
    classes_ = np.unique(Y)  
    K = len(classes_)   
    N_ = np.zeros(K)        
    for k, c1ass in enumerate(classes_):
       
        N_[k] = sum(Y==c1ass)  
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
    classes_ = np.unique(Y) 
    K = len(classes_)    
    N_ = np.zeros(K)     
    for k, c1ass in enumerate(classes_):
        N_[k] = sum(Y==c1ass) 
    p_ = N_/N       
    gini = 1 - sum(p_**2)
    return gini

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





def cut_dataset_by_contious_feature(X: np.ndarray, y: np.ndarray,   # 样本
        m: int,        # 用哪个特征来分隔
        t: float,      # 在哪里分割
        ):
    pass

