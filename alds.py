import torch
import torch.nn as nn
import argparse
from model import ALDS_model
from loss_ALDS import ALDSloss
from dataset import TrainData
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import cfg
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
import time 
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import preprocessing
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser(description='Hyper-parameters')
parser.add_argument('-lr', default=1e-3, type=float)
parser.add_argument('-pretrain_epochs', default=100, type=int, help='Set epochs for training')

args = parser.parse_args()
epochs = args.pretrain_epochs
learning_rate = args.lr

_C = cfg._C_musk
# Wavaform Dataset
dataset_dir = _C['dataset_dir']
dataset_size = _C['train_size']
sketch_size = _C['sketch_size']
batch_size = _C['batch_size']
print('reading...')
dataframe = np.load(dataset_dir)
print('readok')

# Save
save_dir = _C['save_model_dir']

# GPU
device_ids = [Id for Id in range(torch.cuda.device_count())]
device_ids = [0, 1, 2, 3]


def generate_seed(seed):
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed) 
    torch.manual_seed(cfg.seed) 

def train(alpha, beta, gamma, lamb):
    
    def norm_21(x):
        x = x * x
        x = torch.sum(x, 0)
        x = torch.sqrt(x)
        return torch.sum(x)
    def norm_inf(x):
        x = torch.abs(x)
        x = torch.max(x, 0)[0]
        print(x.size())
        return torch.sum(x)

    ALDS = ALDS_model(_C['feature_in'], origin_size=dataset_size, sketch_size=sketch_size, _C=_C, fine_tune=False)
    ALDS = ALDS.cuda()
    train_data_loader = DataLoader(TrainData(dataset_dir, _C['SNR']), batch_size=_C['batch_size'], shuffle=False, num_workers=0)
    # Define optimizer
    opt = torch.optim.Adam(ALDS.parameters(), lr=learning_rate)
    #define losses #
    CELoss = nn.CrossEntropyLoss()
    MSELoss = nn.MSELoss(reduction='sum')
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in ALDS.parameters())))
    # pretrain
    for epoch in range(epochs):
        for batch_id, train_data in enumerate(train_data_loader):
            # Get train_data
            X, X_label, Fake, Fake_label = train_data
            X, X_label, Fake, Fake_label = X.cuda(), X_label.cuda(), Fake.cuda(), Fake_label.cuda()
            opt.zero_grad()
            X_prime, score_x, score_fake = ALDS(X, Fake)
            loss_r = MSELoss(X_prime, X)
            
            loss_c_true = CELoss(score_x, X_label)
            loss_c_false = CELoss(score_fake, Fake_label)
            loss_c = loss_c_true + loss_c_false

            total_loss = loss_r + alpha*loss_c
            total_loss.backward() 
            opt.step()
        if epoch % 2 == 0:
            print("epoch:{}, loss_r:{}, loss_c:{}, total_loss:{}"\
            .format(epoch, loss_r.item(), loss_c.item(), total_loss.item()))
    state_dict = ALDS.state_dict()
    torch.save(state_dict,  os.path.join(save_dir, 'pretrain'))

    print('---------------------- fine-tuning -----------------------')
    
    #fine-tuning 
    ALDS = ALDS_model(_C['feature_in'], origin_size=dataset_size, sketch_size=sketch_size, fine_tune=True, _C=_C)
    ALDS = ALDS.cuda()
    state_dict = torch.load(os.path.join(save_dir, 'pretrain'))
    ALDS.load_state_dict(state_dict)
    
    opt = torch.optim.Adam(ALDS.parameters(), lr=learning_rate / 2)
    # use whole dataset as a batch
    batch_size = dataset_size
    print("batch_size:{}".format(batch_size))
    train_data_loader = DataLoader(TrainData(dataset_dir, _C['SNR']), \
                                   batch_size=batch_size, shuffle=False, num_workers=0
                                   )

    for epoch in range(20):
        for batch_id, train_data in enumerate(train_data_loader):
            X, X_label, Fake, Fake_label = train_data
            X, X_label, Fake, Fake_label = X.cuda(), X_label.cuda(), Fake.cuda(), Fake_label.cuda()
            opt.zero_grad()
            S, S_prime, Z, Z_prime, X_prime, score_x, score_fake, Q = ALDS(X, Fake)

            loss_c_true = CELoss(score_x, X_label)
            loss_c_false = CELoss(score_fake, Fake_label)
            loss_c = loss_c_true + loss_c_false

            Loss = ALDSloss()
            loss = Loss(S, S_prime, Z, Z_prime, X, X_prime, Q, lamb)
            loss_S = loss['loss_S']
            loss_recS = loss['loss_recS']
            norm_q = loss['norm_q']
            loss_Z = loss['loss_Z']
            loss_r = loss['loss_r']
            total_loss = beta*loss_S + loss_r + gamma*loss_Z + alpha*loss_c
            total_loss.backward()   
            opt.step()
        print("epoch:{}, loss_recS:{}, norm:{}, loss_r:{}, loss_c:{}, total_loss:{}"\
                .format(epoch, loss_recS.item(), norm_q.item(), loss_r.item(), loss_c.item(), total_loss.item()))
    test(ALDS.state_dict(), state_dict)

def test(ft_state_dict, pre_state_dict):

    query_list = np.linspace(50, 600, 12, dtype=np.int32)
    # query_list = np.linspace(25, 300, 12, dtype=np.int32)

    ALDS = ALDS_model(_C['feature_in'], origin_size=dataset_size, sketch_size=sketch_size, fine_tune=True, _C=_C, test=True)
    ALDS = ALDS.cuda()
    ALDS.load_state_dict(pre_state_dict)

    def select(Q, query):
        dataset = np.array(dataframe, dtype=np.float32)[0:_C['train_size']]
        _Q = Q.cpu().numpy()
        _Q = np.sum(_Q, 0)
        idx = np.argsort(-_Q)
        np.save(os.path.join(save_dir, 'samples_index.npy'), idx)
        # sio.savemat(os.path.join(save_dir,'order{}.mat'.format(seq)),{'order':idx})
        idx = idx[0:query]
        n, d = dataset.shape
        datas = dataset[idx, :d - 1]
        label = dataset[idx, -1]

        label = label.reshape(label.shape[0], 1)
        datas = ALDS(torch.from_numpy(datas).cuda())
        datas = datas.cpu().detach().numpy()
        datas = np.concatenate((datas, label), axis=1)
        return datas

    def calc_acc(clf, X, y):
        y_test = clf.predict(X)
        acc = sum(y_test == y) / len(y)
        return acc

    def calc_auc(clf, X, y):
        n_class = np.unique(y).shape[0]
        print(n_class)
        y_one_hot = label_binarize(y, np.arange(n_class))
        print(y_one_hot)
        y_score = clf.predict_proba(X)
        auc = AUC(y_one_hot, y_score, average='micro')
        return auc

    def svm_train(query):
        dataset = select(ft_state_dict['Q'], query)
        n, d = dataset.shape
        X = dataset[:, :d - 1]
        y = dataset[:, -1]
        from sklearn.svm import LinearSVC, SVC
        clf = SVC(C=100, kernel='linear')

        start = time.time()
        clf.fit(X, y)
        end = time.time()
        return clf

    def svm_test(clf):
        dataset = np.array(dataframe, dtype=np.float32)[_C['test_size']:,:]
        n, d = dataset.shape
        X = dataset[:, :d - 1]
        y = dataset[:, -1]
        X = ALDS(torch.from_numpy(X).cuda())
        X = X.cpu().detach().numpy()
        acc = calc_acc(clf, X, y)
        return acc

    acc = []
    for query in query_list:
        start = time.time()
        clf = svm_train(query)
        train_time = time.time()
        acc.append(svm_test(clf))
    print(acc)
    

def main():
    # hyper-parameters
    alpha = 0.1
    beta = 1
    gamma = 0.01
    lamb = 10
    train(alpha, beta, gamma, lamb)
                        
if __name__ == '__main__':
    main()