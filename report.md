# 机器学习第二次作业
董浩宇 502023150001

##  1. 实现方案
### 1.1 DataLoader
为了更加方便的实现训练数据的读取，分割以及特征缩放，在本次作业中设计了dataLoader类，使用此类对数据集进行处理。

本次作业中使用到了所有算法都使用DataLoader类来载入、分割、缩放数据。本次作业默认使用标准化特征缩放，十折分割验证。

### 1.2 softmax回归
softmax多元回归模型公式可以表示为：
$$P(y=k|x;\theta) = \frac{e^{\theta^T_kx}}{\sum^K_{j=1}e^{\theta^T_jx}}=\sigma_k(x,\theta),k=1,2,...,K$$
预测分类为：
$$\hat{y^*}=\mathop{\arg\max}\limits_{k}p_k(y|x^*,\hat{\theta}),k=1,2,..,N$$
也就是说，假设现在的数据集中标签种类数为K，即创建K个线性回归模型，将数据集特征输入线性回归模型，得到K个结果，那个标签的结果大，就认为当前特征对应此标签。softmax的作用是扩大K个线性回归模型输出结果的差距。

在代码实现中：数据集共有11个特征，加上一个偏置项，线性回归的输入共有12维。因为酒分为0~9十个等级，故需要建立10个线性回归模型，这里为了方便运算，我直接创建了一个12输入，10输出的线性回归模型：
$$\hat{y} = x \theta $$
N为训练集中data数量，其中$\hat{y}$为N*10维向量，表示特征属于各个标签的预测概率，$\theta$为12\*10维矩阵，x为N\*12维向量，表示数据集里的特征。使用这样的写法，只需要一次矩阵运算，就可以通过线性回归直接算出来训练集中所有data对于十个标签的概率。

对于模型的训练，使用梯度下降法，梯度的计算方法为：
$$\frac{\partial J(\theta)}{\partial\theta_l}=-\sum^N_{i=1}[x_i 1\{y_i=l\}-\sigma_l(x_i,\theta)]$$

如果直接使用上面的公式，需要用到循环，然而python的循环很慢，使用循环的话是不现实的。为了解决这个问题，我这里使用了one-hot编码，编码方式为：

|   特征\品质   | 0  | 1  |2|3|4|5|6|7|8|9|
|  ----   | ----  | ----  |---- |---- |---- |---- |---- |---- |---- |---- |
| x0  | 1 | 0 | 0|0 |0 |0 |0 |0 |0 |0 |0 |0 |
| x1  | 0 | 1 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
| x2  | 0 | 0 | 1|0 |0 |0 |0 |0 |0 |0 |0 |
| x3  | 0 | 0 |0|1 |0 |0 |0 |0 |0 |0 |0 |
| x4  | 0 | 0 |0 |0 |1 |0 |0 |0 |0 |0 |0 |
| x5  | 0 | 0 | 0|0 |0 |1 |0 |0 |0 |0 |0 |
| x6  | 0 | 0 |0|0 |0 |0 |1 |0 |0 |0 |0 |
| x7  | 0 | 0 |0 |0 |0 |0 |0 |1 |0 |0 |0 |
| x8  | 0 |0 | 0|0 |0 |0 |0 |0 |1 |0 |0 |
| x9  | 0 | 0 |0|0 |0 |0 |0 |0 |0 |1 |0 |0 |

这样的话data的标签y也是一个10维的向量，也就是说，可以直接通过$\hat{y}-y$来一下子计算出十个模型的loss，也可以通过
$$\frac{\partial J(\theta)}{\partial\theta}=X^T(X\theta-Y)$$
的公式直接计算出十个回归模型的梯度。（上面式子中，Y为N*10的矩阵，$X\theta$也就是$\hat{y}$，也是N*10的矩阵。）

使用这样的方法实现的代码速度巨快无比，10000次梯度下降只需要几秒钟，而且只需要在上一次作业的基础上改一下参数矩阵就可以了。

### 1.3 决策树
#### 1.3.1 决策树的实现
在python中使用嵌套列表的方式去实现树形结构。在作业的代码中，使用一个node class来表示树中的节点,每一个节点可以指向另外两个节点，从而生成二叉决策树。在node中保存一些信息，用来表示此节点的类型（leaf root inner），保存此节点决策使用的特征以及特征的划分方式。

```
class node():
        def __init__(self,inputs...):
            infomations...

            l_n = None(代表子节点1)
            r_n = None(代表子节点2)
            self.next = [l_n,r_n]
```
因为node可以指向其他node，可以使用递归的方式进行生成以及遍历。
遍历生成的伪代码如下：
```
def generate_tree(inputs)  
        
        if 停止递归：
            将此node设置为leaf型
            return

        L_node = node(分给子节点的数据以及特征，inner型)
        now_node.子节点1 = generate_tree(L_node)
    
        R_node = node(分给子节点的数据以及特征)
        now_node.子节点1 = generate_tree(R_node，inner型)

        return now_node
        
```
使用这样的递归方法，只需要按照PPT上的停止递归条件，以及数据分割条件，就可以很方便地生成各种各样的二叉决策树。

决策树的遍历很简单，只需要按照不同节点的条件进行递归即可，直到递归到leaf节点为止。

#### 1.3.2 C4.5决策树
#### 1.3.3 CART决策树

##  2. 实验结果
### 1.1 softmax回归
### 1.2 C4.5决策树
### 1.2 CART决策树

## 3. 不同模型的比较