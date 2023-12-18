import numpy as np

class Regression_Trainer():
# 传入参数包含有切分好的数据集与训练集
# algo：使用线性回归，岭回归，或者是Lasso回归，三选一，输入字符串，默认为线性回归


    def __init__(self, training_data:np.ndarray, test_data:np.ndarray, algo:str = "Linear"):
        self.training_data = training_data
        self.test_data = test_data
        self.algo = algo
        assert algo in ['Linear', 'Regu']
            

    
    def Y_hat(self, X:np.ndarray, theta:np.ndarray)->np.ndarray:
        return np.dot(X, theta)
    
    def Y_hat_softmax(self, X:np.ndarray, theta:np.ndarray)->np.ndarray:
        Y = self.Y_hat(X, theta)
        max = np.max(Y, axis=1, keepdims=True)  # returns max of each row and keeps same dims
        e_y = np.exp(Y - max)  # subtracts each row with its max value
        sum = np.sum(
        e_y, axis=1, keepdims=True
        )  # returns sum of each row and keeps same dims
        f_y = e_y / sum
        return f_y

    def distance(self,  Y_Hat:np.ndarray, Y:np.ndarray):
        return np.square(Y_Hat-Y).sum(axis = 1)

    def gradient(self, X:np.ndarray, Y:np.ndarray, theta:np.ndarray, lambda_theta)->np.ndarray:

        grad = np.dot(X.T , (self.Y_hat_softmax(X, theta)-Y) )
        grad /= X.shape[0]

        if self.algo == 'Regu':
            theta_of_x = theta.copy()
            theta_of_x[0,:] = 0
            # print(grad)
            # print(lambda_theta)
            grad += lambda_theta * theta_of_x
        # print(grad)
        return grad

    def correct_rate(self, X:np.ndarray, Y:np.ndarray, theta:np.ndarray)->np.ndarray:
        y_hat = self.Y_hat_softmax(X, theta)
        counter = 0
        for i in range(Y.shape[0]):
            # print(y_hat[i])
            if Y[i,y_hat[i].argmax()] == 1:
                counter += 1

        return counter/Y.shape[0]

    def train(self,lr=1e-1, init_theta = np.zeros((12,10)), lambda_theta = 0.001, l_steps = 5000):
        # 为了让偏置项与其他的参数写成一个矩阵，这边将一行1
        # 写在X的第一行，这样的话结果直接就是y=theta*x
        # 不需要在计算公式中另外加入偏置项。
        
        X_1 = np.ones((len(self.training_data),1))
    
        X = self.training_data[:, :-1]
        
        Y_origin = self.training_data[:, -1].reshape(-1,1)


        X = np.hstack((X_1, X))

        Y = self.Origin2Onehot(Y_origin)
 


        theta = init_theta


        for i in range(l_steps):
            
            if i % 1000 == 0:
                print("CorrectRate@epoch", i, ":", self.correct_rate(X, Y, theta))
                # print(theta)
            # writer.add_scalars('loss', {lables :self.MSE(X, Y, theta)}, i)
            theta = theta - lr * self.gradient(X, Y, theta, lambda_theta)

        print(theta)
        return theta
            

    def test(self, theta):
        X_1 = np.ones((len(self.test_data),1))
    
        X = self.test_data[:, :-1]
        Y = self.test_data[:, -1].reshape(-1,1)
        X = np.hstack((X_1, X))
        # print(X)
        # print(Y)

        print("MSELoss@:", self.MSE(X, Y, theta))
        return self.MSE(X, Y, theta)
    

    def Origin2Onehot(self, Y_origin):
        one_hot = np.zeros((len(Y_origin),10))
        for i in range(len(Y_origin)):

            one_hot[i][int(Y_origin[i])] = 1
        
        return one_hot
        