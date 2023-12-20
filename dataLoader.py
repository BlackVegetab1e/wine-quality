import pandas as pd
import numpy as np

class dataLoader():
    def __init__(self, data_loc):
        df = pd.read_csv(data_loc)
        # self.datas = np.array(df)
        self.datas = df.to_numpy()
        
        self.datas = self.datas.astype(dtype=np.float64)

   
    def data_cut(self, segment_number:int = 10, segment_selected:int = 0):
    # 进行数据分割的函数，将training_data分为segment_number份，
    # 并且选中segment_selected当作测试集，其余数据当作训练集
        assert segment_selected>=0 and segment_selected < segment_number
        # 要求分段的合理性，选中数字应该在0到n-1中选择
        data_length = len(self.datas)
        segment_length = int(data_length/segment_number)
        segment_index_begain = segment_length*segment_selected
        segment_index_end = segment_length*(segment_selected+1)
        
        if segment_selected == segment_number-1:
            # 如果是最后一块，将最后末尾的当作测试集，即从尾开始数
            test_data = self.datas[-segment_length:]
            training_data = self.datas[0:data_length-segment_length]
        else:
            # 其他的都是从头开始算
            test_data = self.datas[segment_index_begain:segment_index_end]
            training_data = np.vstack((self.datas[0:segment_index_begain], self.datas[segment_index_end:]))

        return training_data, test_data
    

    def normalization(self, type = None):
        # 各种正则化方法，需要注意的是，一个程序只能用一次，不能重复使用两个
        # emm确实有点不合理,但是就这样吧 
        if type == None:
            return
        elif type == 'MinMax':
            min_col = np.min(self.datas, axis=0)
            max_col = np.max(self.datas, axis=0)
            
            for i in range(self.datas[0].shape[0]-1):
                self.datas[:,i] = (self.datas[:,i] - min_col[i])/(max_col[i] - min_col[i])
            
        elif type == 'Mean':
            min_col = np.min(self.datas, axis=0)
            max_col = np.max(self.datas, axis=0)
            mean_col = np.mean(self.datas, axis=0)
            for i in range(self.datas[0].shape[0]-1):
                self.datas[:,i] = (self.datas[:,i] - mean_col[i])/(max_col[i] - min_col[i])

        elif type == 'Standardization':
            var_col = np.var(self.datas, axis=0)
            mean_col = np.mean(self.datas, axis=0)
            for i in range(self.datas[0].shape[0]-1):
                self.datas[:,i] = (self.datas[:,i] - mean_col[i])/var_col[i]
