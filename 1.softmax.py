from dataLoader import dataLoader
from Regression_Trainer import Regression_Trainer as Trainer
import numpy as np
from tensorboard import SummaryWriter

if __name__ == "__main__":

    writer = SummaryWriter(log_dir='./log/1_standard')
    lr=1e-4
    init_theta = np.zeros((12,10))
    lambda_theta = 1
    l_steps = 5000

    data_path = './Data/winequality-white.csv'
    # algo = "Linear"
    algo = "Regu"
    normization_type = 'Standardization'

    # correct_rate = np.zeros((10,))
    # for i in range(10):
    #     data = dataLoader(data_path)
    #     data.normalization(normization_type)
    #     training_data, test_data = data.data_cut(10, i)
    #     t = Trainer(training_data, test_data, algo)
    #     theta = t.train(lr=lr, init_theta=init_theta , l_steps=l_steps ,lambda_theta=lambda_theta)
    #     correct_rate[i] = t.test(theta)
    # print(theta)
    # print('correct rate:',correct_rate)
    # print('mean correct rate:',correct_rate.mean())
    data = dataLoader(data_path)
    data.normalization(normization_type)
    training_data, test_data = data.data_cut(10, 0)
    t = Trainer(training_data, test_data, algo)
    theta = t.train(lr=lr, init_theta=init_theta , l_steps=l_steps ,lambda_theta=lambda_theta, writer = writer , lable = 'Regulation')
    correct_rate = t.test(theta)
    print(theta)
    print('correct rate:',correct_rate)
    





