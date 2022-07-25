
import sys
import torch
import gplvm
import numpy as np
import scipy.stats as st

"""
# pmf methods
use the l1 distance between datasets to recommend
"""
class L1Recommender:
    def __init__(self):
        self.name = 'L1'
    
    def train(self, Ytrain, Ftrain):
        self.Ytrain = Ytrain
        self.Ftrain = Ftrain
    
    def recommend(self, ftest, n_init=5):
        if len(ftest.shape)==1:
            dis = np.abs(self.Ftrain - ftest).sum(axis=1)
            ix_closest = np.argsort(dis)[:n_init]
            ix_nonnan_pipelines \
                = np.where(np.invert(np.isnan(self.Ytrain[:,ix_closest].sum(axis=1))))[0]
            ranks = np.apply_along_axis(st.rankdata, 0,
                                    self.Ytrain[ix_nonnan_pipelines[:,None],ix_closest])
            ave_pipeline_ranks = ranks.mean(axis=1)
            ix_init = ix_nonnan_pipelines[np.argsort(ave_pipeline_ranks)[::-1]]
            return ix_init

        else:
            ix_inits = []
            for ft in ftest:
                ix_inits.append(self.recommend(ft, n_init).tolist())
            return np.array(ix_inits)


"""
# Basic methods
use the ranks of previous pipelines' result0ois to recommend
"""
class BasicRecommender:
    def __init__(self):
        return

    def train(self, Ytrain, Ftrain):
        self.model = np.argsort(-Ytrain.mean(axis=1))
    
    def recommend(self, ftest, n_init=5):
        if len(ftest.shape) == 1:
            return self.model#[:n_init]
        else:
            return np.tile(self.model, (ftest.shape[0],1))

"""
# kNN method
use the knn method to recommend
"""
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
class myKNN:
    def __init__(self, kwargs={}):
        self.name = 'Basic'
        self.model = KNeighborsClassifier(**kwargs)
    
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def predict(self, x):
        return self.model.predict(x)
    
    def predict_proba(self, x):
        return self.model.predict_proba(x)
    
    def evaluate(self,  x, y):
        return self.model.score(x, y)
    
    def get_classes(self):
        return self.model.classes_


class KnnRecommender:
    def __init__(self, kwargs={}):
        self.name = 'Knn'
        self.model = myKNN(kwargs)
    
    def train(self, Ytrain, Ftrain):
        x = Ftrain
        x[np.isnan(x)] = 0.
        y = (-Ytrain).argsort(axis=0)[0]
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        self.model.train(x_train, y_train)
        return
        print("train: {:.2f}%, val: {:.2f}%".format(\
            self.model.evaluate(x_train, y_train)*100.,
            self.model.evaluate(x_val, y_val)*100.))
    
    def recommend(self, ftest, n_init=5):
        if len(ftest.shape) == 1:
            recommend_rank = (-self.model.predict_proba([ftest])[0]).argsort()
            ix_init = np.array([self.model.get_classes()[i] for i in recommend_rank])
        else:
            recommend_rank = (-self.model.predict_proba(ftest)).argsort(axis=1)
            ix_init = np.apply_along_axis(lambda x: self.model.get_classes()[x], 0, recommend_rank)
        
        
        return ix_init#[:n_init]


"""
# NN method
use the NN method to recommend
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing

class NNClassifier(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=10, output_dim=3):
        super(NNClassifier, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        #self.nn1 = nn.Linear(input_dim, 2*hidden_dim)
        #self.nn2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.nn1 = nn.Linear(input_dim, 2*hidden_dim)
        self.pred = nn.Linear(2*hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.bn1(x)
        x = self.nn1(x)
        x = F.relu(x)

        #x = self.nn2(x)
        #x = F.relu(x)

        return self.pred(x)

class myNN:
    def __init__(self, lr=0.01, momentum=0.9, my_criterion=nn.CrossEntropyLoss(), \
        input_dim=10, hidden_dim=10, output_dim=3, n_epoch=100, is_print=False, val_size=0.2,\
        cuda=False, batch_size=32, save_path='./model/oml-nnc.pkl'):
        self.model = NNClassifier(input_dim, hidden_dim, output_dim)
        self.my_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0001)
        self.my_criterion = my_criterion
        self.n_epoch = n_epoch
        self.is_print = is_print
        self.val_size = 0.2
        self.cuda = cuda
        self.batch_size = batch_size
        self.save_name = save_path

    def train(self, x, y):
        torch.save(self.model, self.save_name)
        
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.val_size)

        # Get DataLoader
        x_train = np.mat(x_train)
        x_train = torch.tensor(x_train).float()
        y_train = torch.tensor(y_train.tolist()).long()
        train_data = torch.utils.data.TensorDataset(x_train, y_train)

        x_val = np.mat(x_val)
        x_val = torch.tensor(x_val).float()
        y_val = torch.tensor(y_val.tolist()).long()
        val_data = torch.utils.data.TensorDataset(x_val, y_val)

        kwargs = {'num_workers':1, 'pin_memory':True} if self.cuda else {}
        train_iter = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_iter = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=False, **kwargs)

        # Train
        best_acc = 0
        best_epoch = 0
        for epoch in range(self.n_epoch):
            train_loss, train_acc = self._train_func(train_iter)
            val_loss, val_acc = self._test_func(val_iter)
            
            if best_acc<val_acc:
                torch.save(self.model, self.save_name)
                best_acc  = val_acc
                best_epoch = epoch

            if epoch%1==0 and self.is_print:
                print('[Epoch {:<4d}] Train {:.2f}%({:.4f}) | Val {:.2f}%({:.4f})'.\
                    format(epoch, train_acc*100, train_loss, val_acc*100, val_loss))

        if self.is_print:
            print('Best Val Acc: {:.2f}% (epoch={})'.format(best_acc*100, best_epoch))
        
        self.model = torch.load(self.save_name)

    def predict(self, x):
        self.model.eval()
        x = np.mat(x)
        x = torch.tensor(x).float()
        
        with torch.no_grad():
            out = self.model(x)

        return out.argmax(1).tolist()
    
    def predict_proba(self, x):
        self.model.eval()
        x = np.mat(x)
        x = torch.tensor(x).float()
        
        with torch.no_grad():
            out = self.model(x)

        return out

    def evaluate(self, x, y):
        x = np.mat(x)
        x = torch.tensor(x).float()
        y = torch.tensor(y.tolist()).long()

        with torch.no_grad():
            out = self.model(x)
            loss = self.my_criterion(out, y)

        return (out.argmax(1)==y).sum().item()/len(y)*100.
    
    def _train_func(self, train_iter):
        self.model.train()
        train_loss = 0
        train_acc = 0
        train_num = 0

        for i, (x,y) in enumerate(train_iter):
            train_num += len(y)
            self.my_optimizer.zero_grad()

            out = self.model(x)
            loss = self.my_criterion(out, y) 
            train_loss += loss.item()

            loss.backward()
            self.my_optimizer.step()

            train_acc += (out.argmax(1)==y).sum().item()
        return train_loss/train_num, train_acc/train_num
    
    def _test_func(self, test_iter):
        self.model.eval()
        loss = 0
        test_acc = 0
        test_num = 0

        for i, (x, y) in enumerate(test_iter):
            test_num += len(y)

            with torch.no_grad():
                out = self.model(x)
                loss = self.my_criterion(out, y)
                loss += loss.item()
                test_acc += (out.argmax(1)==y).sum().item()

        return loss/test_num, test_acc/test_num

class NNRecommender:
    def __init__(self, kwargs={}):
        self.name = 'NN'
        self.kwargs = kwargs
        #self.model = myNN(kwargs)
    
    def train(self, Ytrain, Ftrain):
        x = Ftrain
        x[np.isnan(x)] = 0.

        y = (-Ytrain).argsort(axis=0)[0]
        self.enc = preprocessing.LabelEncoder()
        y = self.enc.fit_transform(y)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

        self.kwargs['output_dim'] = len(self.enc.classes_)
        self.model = myNN(**self.kwargs)
        self.model.train(x_train, y_train)

        train_acc = self.model.evaluate(x_train, y_train)
        val_acc = self.model.evaluate(x_val, y_val)
        print("train: {:.2f}%, val: {:.2f}%".format(\
            train_acc, val_acc))
    
    def recommend(self, ftest, n_init=5):
        if len(ftest.shape)==1:
            recommend_rank = (-self.model.predict_proba([ftest])[0]).argsort()
        else:
            recommend_rank = (-self.model.predict_proba(ftest)).argsort(axis=1)
        ix_init = self.enc.inverse_transform(recommend_rank.reshape(-1)).reshape(recommend_rank.shape)

        return ix_init#[:n_init]


"""
# PMM method
use the PMM method to recommend
"""
import random

# Lam Distance
def lam_distance(x1, x2):
  return torch.sigmoid(x1-x2).reshape(-1)

# Embedding Net
class LamNet(nn.Module):
  def __init__(self, n_input, n_hidden, n_output=1):
    super(LamNet, self).__init__()

    self.hidden1 = nn.Linear(n_input, n_hidden)
    self.hidden2 = nn.Linear(n_hidden, n_hidden)
    self.pred = nn.Linear(n_hidden, n_output)

    #self.dropout = nn.Dropout(p=0.2)

  def forward(self, x):
    x = self.hidden1(x)
    x = F.relu(x)
    #x = self.dropout(x)

    x = self.hidden2(x)
    x = F.relu(x)
    #x = self.dropout(x)

    out = self.pred(x)
    return out

# Siamese Network
class SiameseNet(nn.Module):
  def __init__(self, lamNet):
    super(SiameseNet, self).__init__()
    self.lam = lamNet
  
  def forward(self, x1, x2):
    out1 = self.lam(x1)
    out2 = self.lam(x2)

    return out1, out2
  
  def get_lam(self, x):
    return self.lam(x)

  def _predict_proba(self, x, FPipeline):
    self.eval()
    x = np.expand_dims(x, 0).repeat(FPipeline.shape[0], axis=0)
    x = np.concatenate((x, FPipeline), axis=1)
    x = torch.tensor(x).float()

    with torch.no_grad():
        outs = self.get_lam(x)
    
    return outs.reshape(-1)

  def predict_proba(self, xs, FPipeline):
    y = []
    for x in xs:
        y.append(self._predict_proba(x, FPipeline).numpy().tolist())
    return np.array(y)

  def evaluate(self, x, y, FPipeline):
      outs = self.predict_proba(x, FPipeline)
      return (outs.argmax(1)==y).sum().item()/len(y)*100.
    


# Loss
class ContrastiveLoss(nn.Module):
  def __init__(self, margin=0.8):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin
    self.eps = 1e-9
  
  def forward(self, out, target, size_average=True):
    out1, out2 = out
    distance = lam_distance(out1, out2)
    losses = 0.5 * ((1 - target).float()*distance + 
                    target.float() * F.relu(self.margin - (distance + self.eps)))
    
    return losses.mean() if size_average else loss.num()


# LamDataset
from torch.utils.data import Dataset
class LamDataset(Dataset):
  def __init__(self, pairs1, pairs2, labels=None):
    self.pairs1 = pairs1
    self.pairs2 = pairs2
    self.labels = labels
  
  def __getitem__(self, index):
    if self.labels is None:
      return self.pairs1[index], self.pairs2[index]
    else:
      return self.pairs1[index], self.pairs2[index], self.labels[index]
  
  def __len__(self):
    return len(self.pairs1)


class PmmRecommender:
    def __init__(self, kwargs={}):
        self.name = 'PMM'
        self.kwargs = kwargs

    def train(self, Ytrain, Ftrain, FPipeline):
        self.FPipeline = FPipeline

        # create lam pairs
        pairs1, pairs2, labels = self._create_pairs(Ytrain, Ftrain, FPipeline, self.kwargs['total_pairs'], 
                                [self.kwargs['pair_sd'], self.kwargs['pair_sp']], [self.kwargs['rank_s1'], self.kwargs['rank_s2']])

        # get dataloader
        train_data, val_data, test_data = self._split_lam_datasets(pairs1, pairs2, labels, part_ratio=0.8, val_ratio=0.2)

        kwargs = {'num_workers':1, 'pin_memory':True} if self.kwargs['cuda'] else {}
        train_iter = torch.utils.data.DataLoader(train_data, \
                            batch_size=self.kwargs['batch_size'], shuffle=True, **kwargs)
        val_iter = torch.utils.data.DataLoader(val_data, \
                            batch_size=self.kwargs['batch_size'], shuffle=True, **kwargs)
        test_iter  = torch.utils.data.DataLoader(test_data, \
                            batch_size=self.kwargs['batch_size'], shuffle=False, **kwargs)

        # Set Model
        my_lamNet = LamNet(n_input=self.kwargs['input_dim'], n_hidden=self.kwargs['hidden_dim'], n_output=self.kwargs['output_dim'])
        my_model = SiameseNet(my_lamNet)
        my_optimizer = torch.optim.SGD(my_model.parameters(), lr=self.kwargs['lr'], momentum=0.9, weight_decay=0.0001)
        my_criterion = ContrastiveLoss()
        
        # Train Model
        self.model, lam_train_acc, lam_val_acc, lam_test_acc, lam_epoch = \
            self._train_lam_model(train_iter, val_iter, test_iter, \
                my_model, my_optimizer, my_criterion, \
                n_epoch=self.kwargs['n_epoch'], batch_size=self.kwargs['batch_size'], \
                cuda=self.kwargs['cuda'], save_path=self.kwargs['save_path'], is_print=self.kwargs['is_print'])
        
        #print(self.model.predict_proba(Ftrain[0], FPipeline))
        x = Ftrain
        x[np.isnan(x)] = 0.

        y = (-Ytrain).argsort(axis=0)[0]
        #self.enc = preprocessing.LabelEncoder()
        #y = self.enc.fit_transform(y)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

        print("train: {:.2f}%, val: {:.2f}%".format(\
            self.model.evaluate(x_train, y_train, FPipeline),
            self.model.evaluate(x_val, y_val, FPipeline)))


    def recommend(self, ftest, n_init=5, FPipeline=None):
        if FPipeline is None:
            FPipeline = self.FPipeline
        
        if len(ftest.shape)==1:
            ix_init = (-self.model.predict_proba([ftest], FPipeline)[0]).argsort()
        else:
            ix_init = (-self.model.predict_proba(ftest, FPipeline)).argsort(axis=1)
        #ix_init = self.enc.inverse_transform(recommend_rank)

        return ix_init#[:n_init]
    
    def recommend_with_score(self, ftest, n_init=5, FPipeline=None):
        if FPipeline is None:
            FPipeline = self.FPipeline

        ix_init = self.model.predict_proba([ftest], FPipeline)[0]
        ix_init = (ix_init-np.min(ix_init))/(np.max(ix_init)-np.min(ix_init))

        return (-ix_init).argsort(), ix_init
    
    def _create_pairs(self, Ytrain, Ftrain, FPipeline, total_pairs=0, pair_split=[0,0], rank_split=[1, 1]):
        '''
        Ytrain: (Npipeline, Ndataset)
        Ftrain: (Ndataset, Ndatasetfeats)
        FPipeline: (Npipeline, Npipelinefeats)
        '''

        if total_pairs==0:
            return self._create_pairs_full(Ytrain, Ftrain, FPipeline)
        elif total_pairs==-1:
            return self._create_pairs_all(Ytrain, Ftrain, FPipeline, pair_split, rank_split)

        pairNum = 0
        pipelineNum = Ytrain.shape[0]
        datasetNum = Ytrain.shape[1]

        if rank_split[0]<1:
            rank_split[0] = int(pipelineNum*datasetNum*rank_split[0])
        if rank_split[1]<1:
            rank_split[1] = int(pipelineNum*datasetNum*rank_split[1])

        # generate pairs
        pairs1 = []
        pairs2 = []
        labels = []

        while pairNum < total_pairs:
            pipeIndex1, pipeIndex2 = np.random.randint(0, pipelineNum, size=(2))
            #np.random.choice(range(pipelineNum), size=(2), replace=False)
            datIndex1, datIndex2 = np.random.randint(0, datasetNum, size=(2))
            #np.random.choice(range(datasetNum), size=(2), replace=False)

            if np.random.uniform()<pair_split[0]:
                datIndex1 = datIndex2
            elif np.random.uniform()<(pair_split[0]+pair_split[1]):
                pipeIndex1 = pipeIndex2

            y1 = Ytrain[pipeIndex1, datIndex1]
            y1_rank = np.argsort(np.argsort(-Ytrain[:, datIndex1]))[pipeIndex1]
            y2 = Ytrain[pipeIndex2, datIndex2]
            y2_rank = np.argsort(np.argsort(-Ytrain[:, datIndex2]))[pipeIndex2]

            if np.isnan(y1) or np.isnan(y2):
                continue
            if (y1_rank > rank_split[0]*pipelineNum or y2_rank > rank_split[0]*pipelineNum) and np.random.random()>rank_split[1]:
                continue

            f1 = np.concatenate((Ftrain[datIndex1], FPipeline[pipeIndex1]))
            f2 = np.concatenate((Ftrain[datIndex2], FPipeline[pipeIndex2]))

            pairs1.append(torch.tensor(f1).float())
            pairs2.append(torch.tensor(f2).float())
            
            if y1>=y2:
                y = 1
            else:
                y = 0
            
            labels.append(y)
            pairNum += 1
        
        return pairs1, pairs2, labels

    def _create_pairs_full(self, Ytrain, Ftrain, FPipeline):
        '''
        Ytrain: (Npipeline, Ndataset)
        Ftrain: (Ndataset, Ndatasetfeats)
        FPipeline: (Npipeline, Npipelinefeats)
        '''

        pipelineNum = Ytrain.shape[0]
        datasetNum = Ytrain.shape[1]

        # generate pairs
        pairs1 = []
        pairs2 = []
        labels = []

        for datIndex1 in range(datasetNum):
            for pipeIndex1 in range(pipelineNum):
                
                datIndex2 = np.random.randint(datasetNum)
                pipeIndex2 = np.random.randint(pipelineNum)

                while datIndex1==datIndex2 and pipeIndex1==pipeIndex2:
                    datIndex2 = np.random.randint(datasetNum)
                    pipeIndex2 = np.random.randint(pipelineNum)
                
                y1 = Ytrain[pipeIndex1, datIndex1]
                y2 = Ytrain[pipeIndex2, datIndex2]

                if np.isnan(y1) or np.isnan(y2):
                    continue

                f1 = np.concatenate((Ftrain[datIndex1], FPipeline[pipeIndex1]))
                f2 = np.concatenate((Ftrain[datIndex2], FPipeline[pipeIndex2]))

                pairs1.append(torch.tensor(f1).float())
                pairs2.append(torch.tensor(f2).float())
        
                if y1>=y2:
                    y = 1
                else:
                    y = 0
                
                labels.append(y)

        return pairs1, pairs2, labels
    
    def _create_pairs_all(self, Ytrain, Ftrain, FPipeline, pair_split, rank_split):
        '''
        Ytrain: (Npipeline, Ndataset)
        Ftrain: (Ndataset, Ndatasetfeats)
        FPipeline: (Npipeline, Npipelinefeats)
        '''

        pipelineNum = Ytrain.shape[0]
        datasetNum = Ytrain.shape[1]

        # generate pairs
        pairs1 = []
        pairs2 = []
        labels = []

        for datIndex1 in range(datasetNum):
            for pipeIndex1 in range(pipelineNum):

                y1 = Ytrain[pipeIndex1, datIndex1]
                if np.isnan(y1):
                    continue
                y1_rank = np.argsort(np.argsort(-Ytrain[:, datIndex1]))[pipeIndex1]

                is_good = True
                if y1_rank<pipelineNum*0.5:
                    i=rank_split[0]
                else:
                    i=rank_split[1]
                    is_good = False

                while i>0:
                    datIndex2 = np.random.randint(datasetNum)
                    pipeIndex2 = np.random.randint(pipelineNum)

                    while datIndex1==datIndex2 and pipeIndex1==pipeIndex2:
                        datIndex2 = np.random.randint(datasetNum)
                        pipeIndex2 = np.random.randint(pipelineNum)
                
                    if np.random.uniform()<pair_split[0]:
                        datIndex1 = datIndex2
                    elif np.random.uniform()<(pair_split[0]+pair_split[1]):
                        pipeIndex1 = pipeIndex2

                    y2 = Ytrain[pipeIndex2, datIndex2]
                    y2_rank = np.argsort(np.argsort(-Ytrain[:, datIndex2]))[pipeIndex2]

                    if np.isnan(y2):
                        continue
                    #if (is_good and y2_rank>=pipelineNum*0.2):
                    #    continue
                    if (is_good and y2_rank>=pipelineNum*0.5) or (not is_good and y2_rank<pipelineNum*0.5):
                        continue

                    f1 = np.concatenate((Ftrain[datIndex1], FPipeline[pipeIndex1]))
                    f2 = np.concatenate((Ftrain[datIndex2], FPipeline[pipeIndex2]))

                    pairs1.append(torch.tensor(f1).float())
                    pairs2.append(torch.tensor(f2).float())
            
                    if y1>=y2:
                        y = 1
                    else:
                        y = 0
                    
                    labels.append(y)

                    i=i-1

        return pairs1, pairs2, labels

    def _split_lam_datasets(self, pairs1, pairs2, labels, part_ratio=0.8, val_ratio=0.2):
        num = len(labels)
        train_num = int(num*part_ratio)
        val_num = int(train_num*val_ratio)

        val_data = LamDataset(pairs1[:val_num], pairs2[:val_num], labels[:val_num])
        train_data = LamDataset(pairs1[val_num:train_num], pairs2[val_num:train_num], labels[val_num:train_num])
        test_data  = LamDataset(pairs1[train_num:], pairs2[train_num:], labels[train_num:])

        return train_data, val_data, test_data


    def _train_lam_model(self, train_iter, val_iter, test_iter, my_model, my_optimizer, my_criterion, \
                    n_epoch=20, batch_size=32, cuda=False, save_path='./model/jam.pkl',\
                    is_print=False):
        # Train
        def train_func(train_iter):
            my_model.train()
            train_loss = 0
            train_num = 0
            correct_num = 0

            for i, (x1, x2, y) in enumerate(train_iter):
                train_num += len(y)
                my_optimizer.zero_grad()

                out = my_model(x1, x2)
                loss = my_criterion(out, y)

                loss.backward()

                train_loss += loss.item()*len(y)
                my_optimizer.step()

                out1, out2 = out
                distance = lam_distance(out1, out2)
                correct_num += (y==(distance>=0.8).float()).sum().item()

            return train_loss / train_num, correct_num / train_num*100
        
        # Test
        def test_func(test_iter):
            my_model.eval()
            test_loss = 0
            test_num = 0
            correct_num = 0

            for i, (x1, x2, y) in enumerate(test_iter):
                test_num += len(y)

                with torch.no_grad():
                    out = my_model(x1, x2)
                    loss = my_criterion(out, y)
                    test_loss += loss.item()*len(y)

                    out1, out2 = out
                    distance = lam_distance(out1, out2)
                    correct_num += (y==(distance>=0.8).float()).sum().item()
            
            return test_loss/test_num, correct_num/test_num*100

        # Train Lam Model
        best_loss  = 1000000000
        best_acc   = 0
        best_epoch = 0

        for epoch in range(n_epoch):
            train_loss, train_acc = train_func(train_iter)
            val_loss, val_acc   = test_func(val_iter)

            if best_acc < val_acc:
                torch.save(my_model, save_path)
                best_acc = val_acc
                best_loss  = val_loss
                best_epoch = epoch
            
            if epoch%10==0 and is_print:
                print('[Epoch {:<4d}] Train {:.2f}%({:.4f}) | Val {:.2f}%({:.4f})'.\
                    format(epoch, train_acc, train_loss, val_acc, val_loss))
        
        if is_print:
            print('Best Val: {:.2f}% (epoch={})'.format(best_acc, best_epoch))
        
        my_model = torch.load(save_path)
        train_loss, train_acc = test_func(train_iter)
        val_loss, val_acc   = test_func(val_iter)
        test_loss, test_acc = test_func(test_iter)

        if is_print:
            print('Result: {:.2f}% | {:.2f}% | {:.2f}% (epoch={})'.format(train_acc, val_acc, test_acc, best_epoch))
            train_val_acc = (len(train_iter)*train_acc + len(val_iter)*val_acc) / (len(train_iter) + len(val_iter))
        
        return torch.load(save_path), train_acc, val_acc, test_acc, best_epoch



"""
# BPMM method
use the balanced PMM method to recommend
"""
# Embedding Net
class LamNetSeperate(nn.Module):
  def __init__(self, n_input1, n_input2, n_hidden, n_output=1):
    super(LamNetSeperate, self).__init__()

    self.n_input1 = n_input1
    self.input1 = nn.Linear(n_input1, n_hidden)
    self.input2 = nn.Linear(n_input2, n_hidden)
    self.hidden1 = nn.Linear(2*n_hidden, n_hidden)
    self.pred = nn.Linear(n_hidden, n_output)

    self.bn1 = nn.BatchNorm1d(n_input1+n_input2)

  def forward(self, x):
    x = self.bn1(x)
    x1 = x[:,:self.n_input1]
    x2 = x[:,self.n_input1:]
    #x1, x2 = x.split(self.n_input1, dim=1)

    x1 = self.input1(x1)
    x1 = F.relu(x1)

    x2 = self.input2(x2)
    x2 = F.relu(x2)

    x = self.hidden1(torch.cat([x1, x2], dim=1))
    x = F.relu(x)

    out = self.pred(x)
    return out


class BalancedPmmRecommender(PmmRecommender):
    def __init__(self, kwargs={}):
        self.name = 'PMM'
        self.kwargs = kwargs

    def train(self, Ytrain, Ftrain, FPipeline):
        self.FPipeline = FPipeline

        # create lam pairs
        pairs1, pairs2, labels = self._create_pairs(Ytrain, Ftrain, FPipeline, self.kwargs['total_pairs'], 
                                [self.kwargs['pair_sd'], self.kwargs['pair_sp']], [self.kwargs['rank_s1'], self.kwargs['rank_s2']])
        
        # get dataloader
        train_data, val_data, test_data = self._split_lam_datasets(pairs1, pairs2, labels, part_ratio=0.8, val_ratio=0.2)

        kwargs = {'num_workers':1, 'pin_memory':True} if self.kwargs['cuda'] else {}
        train_iter = torch.utils.data.DataLoader(train_data, \
                            batch_size=self.kwargs['batch_size'], shuffle=True, **kwargs)
        val_iter = torch.utils.data.DataLoader(val_data, \
                            batch_size=self.kwargs['batch_size'], shuffle=True, **kwargs)
        test_iter  = torch.utils.data.DataLoader(test_data, \
                            batch_size=self.kwargs['batch_size'], shuffle=False, **kwargs)

        # Set Model
        my_lamNet = LamNetSeperate(n_input1=self.kwargs['input_dim1'], 
                        n_input2=self.kwargs['input_dim2'],
                        n_hidden=self.kwargs['hidden_dim'], 
                        n_output=self.kwargs['output_dim'])
        my_model = SiameseNet(my_lamNet)
        my_optimizer = torch.optim.SGD(my_model.parameters(), lr=self.kwargs['lr'], momentum=0.9, weight_decay=0.0001)
        my_criterion = ContrastiveLoss()
        
        # Train Model
        self.model, lam_train_acc, lam_val_acc, lam_test_acc, lam_epoch = \
            self._train_lam_model(train_iter, val_iter, test_iter, \
                my_model, my_optimizer, my_criterion, \
                n_epoch=self.kwargs['n_epoch'], batch_size=self.kwargs['batch_size'], \
                cuda=self.kwargs['cuda'], save_path=self.kwargs['save_path'], is_print=self.kwargs['is_print'])
        
        #print(self.model.predict_proba(Ftrain[0], FPipeline))
        x = Ftrain
        x[np.isnan(x)] = 0.

        y = (-Ytrain).argsort(axis=0)[0]
        #self.enc = preprocessing.LabelEncoder()
        #y = self.enc.fit_transform(y)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

        print("train: {:.2f}%, val: {:.2f}%".format(\
            self.model.evaluate(x_train, y_train, FPipeline),
            self.model.evaluate(x_val, y_val, FPipeline)))





"""
# Regressor method
use the Regressor method to recommend
"""

class NNRegressor(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=10, output_dim=1):
        super(NNRegressor, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.nn1 = nn.Linear(input_dim, 2*hidden_dim)
        self.nn2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.pred = nn.Linear(hidden_dim, output_dim)

        self.i = 0
    
    def forward(self, x):
        x = self.bn1(x)

        x = self.nn1(x)
        x = F.relu(x)

        x = self.nn2(x)
        x = F.relu(x)
        
        x = self.pred(x)
        return x#self.pred(x)

class myNNR:
    def __init__(self, lr=0.01, momentum=0.9, my_criterion=nn.MSELoss(), \
        input_dim=10, hidden_dim=10, output_dim=1, n_epoch=100, is_print=False, val_size=0.2,\
        cuda=False, batch_size=32, save_path='./model/reg.pkl'):
        self.model = NNRegressor(input_dim, hidden_dim, output_dim)
        self.my_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0001)
        self.my_criterion = my_criterion
        self.n_epoch = n_epoch
        self.is_print = is_print
        self.val_size = 0.2
        self.cuda = cuda
        self.batch_size = batch_size
        self.save_path = save_path

    def train(self, x, y):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.val_size)

        # Get DataLoader
        x_train = np.mat(x_train)
        x_train = torch.tensor(x_train).float()
        y_train = torch.tensor(y_train.tolist()).float()#long()
        train_data = torch.utils.data.TensorDataset(x_train, y_train)

        x_val = np.mat(x_val)
        x_val = torch.tensor(x_val).float()
        y_val = torch.tensor(y_val.tolist())#.long()
        val_data = torch.utils.data.TensorDataset(x_val, y_val)

        kwargs = {'num_workers':1, 'pin_memory':True} if self.cuda else {}
        train_iter = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_iter = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=False, **kwargs)

        # Train
        best_loss = 10000000
        best_epoch = 0
        for epoch in range(self.n_epoch):
            train_loss = self._train_func(train_iter)
            val_loss = self._test_func(val_iter)
            
            if best_loss>val_loss:
                torch.save(self.model, self.save_path)
                best_loss  = val_loss
                best_epoch = epoch

            if epoch%1==0 and self.is_print:
                print('[Epoch {:<4d}] Train {:.4f} | Val {:.4f}'.\
                    format(epoch, train_loss, val_loss))

        if self.is_print:
            print('Best Val Loss: {:.4f} (epoch={})'.format(best_loss, best_epoch))
        
        self.model = torch.load(self.save_path)

    def predict(self, x):
        x = np.mat(x)
        x = torch.tensor(x).float()
        
        with torch.no_grad():
            out = self.model(x)

        return out
    
    def _predict_proba(self, x, FPipeline):
        self.model.eval()
        x = np.expand_dims(x, 0).repeat(FPipeline.shape[0], axis=0)
        x = np.concatenate((x,FPipeline), axis=1)
        x = torch.tensor(x).float()

        with torch.no_grad():
            outs = self.model(x)
        return outs.reshape(-1)

    def predict_proba(self, xs, FPipeline):
        y = []
        for x in xs:
            y.append(self._predict_proba(x, FPipeline).numpy().tolist())
        
        return np.array(y)

    def evaluate(self, x, y, FPipeline):
        outs = self.predict_proba(x, FPipeline)

        return (outs.argmax(1)==y).sum().item()/len(y)*100.
    
    def _train_func(self, train_iter):
        train_loss = 0
        train_num = 0

        for i, (x,y) in enumerate(train_iter):
            train_num += len(y)
            self.my_optimizer.zero_grad()

            out = self.model(x).reshape(-1)
            loss = self.my_criterion(out, y) 
            train_loss += loss.item()

            loss.backward()
            self.my_optimizer.step()

        return train_loss/train_num
    
    def _test_func(self, test_iter):
        loss = 0
        test_num = 0

        for i, (x, y) in enumerate(test_iter):
            test_num += len(y)

            with torch.no_grad():
                out = self.model(x).reshape(-1)
                loss = self.my_criterion(out, y)
                loss += loss.item()

        return loss/test_num

class RegressorRecommender:
    def __init__(self, kwargs={}):
        self.name = 'Regressor'
        self.kwargs = kwargs
        #self.model = myNN(kwargs)
    
    def train(self, Ytrain, Ftrain, FPipeline):
        self.FPipeline = FPipeline

        x_datas = Ftrain
        x_datas[np.isnan(x_datas)] = 0.
        x = None

        for x_pipe in FPipeline:
            _x = np.expand_dims(x_pipe, 0).repeat(x_datas.shape[0], axis=0)
            _x = np.concatenate((x_datas, _x), axis=1)
            if x is None:
                x = _x
            else:
                x = np.concatenate((x, _x), axis=0)
        
        y = Ytrain.reshape(-1)
        nonan_ix = np.where(np.invert(np.isnan(y)))

        x = x[nonan_ix]
        y = y[nonan_ix]
        #print('nan', np.isnan(y).sum(), x.shape, y.shape)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

        self.kwargs['output_dim'] = 1#len(self.enc.classes_)
        self.model = myNNR(**self.kwargs)
        self.model.train(x_train, y_train)

        
        x = Ftrain
        x[np.isnan(x)] = 0.

        y = (-Ytrain).argsort(axis=0)[0]
        #self.enc = preprocessing.LabelEncoder()
        #y = self.enc.fit_transform(y)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        
        train_acc = self.model.evaluate(x_train, y_train, FPipeline)
        val_acc = self.model.evaluate(x_val, y_val, FPipeline)
        print("train: {:.2f}%, val: {:.2f}%".format(\
            train_acc, val_acc))
    
    def recommend(self, ftest, n_init=5):
        if len(ftest.shape)==1:
            recommend_rank = (-self.model.predict_proba([ftest], self.FPipeline)[0]).argsort()
        else:
            recommend_rank = (-self.model.predict_proba(ftest, self.FPipeline)).argsort(axis=1)
        ix_init = recommend_rank
        #ix_init = self.enc.inverse_transform(recommend_rank.reshape(-1)).reshape(recommend_rank.shape)

        return ix_init#[:n_init]