import torch.utils.data as data
import pandas as pd
import numpy as np
from random import randint
import torch
import random
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.decomposition import PCA
import cfg
def generate_seed():
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed) 
    torch.manual_seed(cfg.seed) 

class TrainData(data.Dataset):
    def __init__(self, train_dir, SNR):
        super().__init__()
        dataframe = np.load(train_dir)
        self.dataset = np.array(dataframe, dtype=np.float32)
        n, d = self.dataset.shape
        self.dataset = self.dataset[0:n//2,:]
        n, d = self.dataset.shape
        self.datas = self.dataset[:, :d - 1]
        self.datas = preprocessing.normalize(self.datas)
        self.d = d - 1
        self.n = n
        self.fakes = self.get_fakes_noise(self.datas, SNR)

    def minmaxscaler(self, data):
        min = np.amin(data)
        max = np.amax(data)    
        return (data - min)/(max-min)

    def feature_normalize(self, data):
        mu = np.mean(data,axis=0)
        std = np.std(data,axis=0)
        return (data - mu)/std

    def __getitem__(self, index):
        data = self.datas[index, :]

        data = torch.from_numpy(data)
        data_label = torch.tensor(1)

        fake_data = self.fakes[index, :]
        fake_data = torch.from_numpy(fake_data)
        fake_label = torch.tensor(0)

        return data, data_label, fake_data, fake_label
    
    def __len__(self):
        return self.n

    # def get_fakes(self, datas):
    #     fakes = datas.copy()
    #     random.seed(166)
    #     random_pair = [(randint(0, self.d - 1), randint(0, self.d - 1)) for i in range(self.shuffle_num)]
    #     for j in range(self.n):
    #         #fake_data = fakes[:, j]
    #         for i in range(self.shuffle_num):
    #             fakes[j, :][random_pair[i][0]], fakes[j, :][random_pair[i][1]] = \
    #             fakes[j, :][random_pair[i][1]], fakes[j, :][random_pair[i][0]]
    #     return fakes

    def get_fakes_noise(self, datas, SNR):
        fakes = datas.copy()
        generate_seed()
        # 给数据加指定SNR的高斯噪声
        SNR = 5
        noise = np.random.randn(datas.shape[0],datas.shape[1]) 	#产生N(0,1)噪声数据
        noise = noise-np.mean(noise) 								#均值为0
        signal_power = np.linalg.norm( fakes )**2 / fakes.size	#此处是信号的std**2
        noise_variance = signal_power/np.power(10,(SNR/10))         #此处是噪声的std**2
        noise = (np.sqrt(noise_variance) / np.std(noise) )*noise    ##此处是噪声的std**2
        signal_noise = noise + fakes
        #print(type(signal_noise))
        Ps = ( np.linalg.norm(fakes - fakes.mean()) )**2          #signal power
        Pn = ( np.linalg.norm(fakes - signal_noise ) )**2          #noise power
        snr = 10*np.log10(Ps/Pn)
        print(snr)
        return np.array(signal_noise, dtype=np.float32)

