from dataLoader import dataLoader
from Regression_Trainer import Regression_Trainer as Trainer
import numpy as np
if __name__ == "__main__":


    lr=1e-1
    init_theta = np.zeros((12,10))
    lambda_theta = 0.001
    l_steps = 10000

    data_path = './Data/winequality-white.csv'

    algo = "Linear"
    # algo = "Regu"


    normization_type = 'Standardization'

    data = dataLoader(data_path)

    data.normalization(normization_type)
    training_data, test_data = data.data_cut(10, 9)
    

    t = Trainer(training_data, test_data, algo)
    theta = t.train(lr=lr, init_theta=init_theta , l_steps=l_steps ,lambda_theta=0.001)





