import torch
import random
import argparse
import numpy as np
import pandas as pd
import numpy as np

'''load arguments'''
def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_name', default='pmf', help='pmf, 110-classifiers, openml')
    parser.add_argument(
        '--random_seed', type=int, default=0, help='for random seed')
    parser.add_argument(
        '--nan_ratio', type=float, default=0, help='for random seed')
    parser.add_argument(
        '--save_name', default='default', help='save the reuslts')
    parser.add_argument(
        '--part_name', default='0', help='part')
    parser.add_argument(
        '--model_path', default='None', help='the pmf model path')
    parser.add_argument(
        '--warm_start', default='default', help='the path for the define of warm starters')
    parser.add_argument(
        '--bo_n_init', type=int, default=5, help='the number of pipelines for warm start')
    parser.add_argument(
        '--bo_n_iters', type=int, default=200, help='the number of pipelines for search')

    args, unparsed = parser.parse_known_args()
    return args


'''set random seed'''
def setup_seed(seed):
    #setup_seed()使用后，只作用于random()函数一次，如之后再次调用random（）函数>，则需要再次调用setup_seed
     torch.manual_seed(seed) #cpu
     torch.cuda.manual_seed(seed)#当前gpu
     torch.cuda.manual_seed_all(seed) #所有gpu
     np.random.seed(seed) #numpy
     random.seed(seed) #python
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True


'''get data'''
def get_data(dataset_name='pmf', pipeline_ixs=None, save_name=None, nan_ratio=0):
    """
    returns the train/test splits of the dataset as N x D matrices and the
    train/test dataset features used for warm-starting bo as D x F matrices.
    N is the number of pipelines, D is the number of datasets (in train/test),
    and F is the number of dataset features.
    """

    fn_data = '../data/{}/performance.csv'.format(dataset_name)
    fn_data_feats = '../data/{}/dataset_feats.csv'.format(dataset_name)
    fn_pipelines_feats = '../data/{}/pipelines.json'.format(dataset_name)

    if dataset_name == 'pmf':
        fn_train_ix = '../data/{}/ids_train.csv'.format(dataset_name)
        fn_test_ix = '../data/{}/ids_test.csv'.format(dataset_name)
    else:
        fn_train_ix = None
        fn_test_ix = None

    '''load performance '''
    df = pd.read_csv(fn_data)
    if pipeline_ixs is not None:
        df = df.iloc[pipeline_ixs]

    #pipeline_ids = df['Unnamed: 0'].tolist()
    dataset_ids = df.columns.tolist()[1:]
    dataset_ids = [int(dataset_ids[i]) for i in range(len(dataset_ids))]
    Y = df.values[:,1:].astype(np.float64)

    '''train / test'''
    if fn_train_ix is None:
        if dataset_name=='openml':
            random_rank = np.random.permutation(dataset_ids)
            n_train = int(len(dataset_ids)*0.8)
            n_test = len(dataset_ids)-n_train
            ids_train = []
            ids_test = []

            for i in random_rank:
                Yi = Y[:, dataset_ids.index(i)]

                if n_test>0 and len(Yi)-np.isnan(Yi).sum()>150:
                    ids_test.append(i)
                    n_test -=1
                else:
                    ids_train.append(i)
            
            ids_train = np.array(ids_train)
            ids_test = np.array(ids_test)

        else:
            random_rank = np.random.permutation(dataset_ids)
            ids_train = random_rank[:int(len(random_rank)*0.8)]
            ids_test = random_rank[int(len(random_rank)*0.8):]
    else:
        ids_train = np.loadtxt(fn_train_ix).astype(int).tolist()
        ids_test = np.loadtxt(fn_test_ix).astype(int).tolist()
    
    random_rank = np.random.permutation(ids_train)
    ids_train = random_rank[:int(len(random_rank)*0.8)]
    ids_val = random_rank[int(len(random_rank)*0.8):]

    np.save('../result/{}/ids_train.npy'.format(save_name), np.array(ids_train))
    np.save('../result/{}/ids_val.npy'.format(save_name), np.array(ids_train))
    np.save('../result/{}/ids_test.npy'.format(save_name), np.array(ids_test))

    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_val = [dataset_ids.index(i) for i in ids_val]
    ix_test = [dataset_ids.index(i) for i in ids_test]

    Ytrain = Y[:, ix_train]
    Yval = Y[:, ix_val]
    Ytest = Y[:, ix_test]

    ''''''
    nan_num = np.isnan(Ytrain).sum()
    total_num = Ytrain.size
    target_num = int(total_num * nan_ratio)
    
    while nan_num < target_num:
        a = np.random.randint(Ytrain.shape[0])
        b = np.random.randint(Ytrain.shape[1])

        while np.isnan(Ytrain[a,b]):
            a = np.random.randint(Ytrain.shape[0])
            b = np.random.randint(Ytrain.shape[1])
        
        Ytrain[a,b] = np.nan
        nan_num += 1
    ''''''

    '''load dataset features'''
    df = pd.read_csv(fn_data_feats)
    df.replace([np.inf,-np.inf], -1, inplace=True)
    df = df.fillna(0)
    dataset_ids = df[df.columns[0]].tolist()

    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_val = [dataset_ids.index(i) for i in ids_val]
    ix_test = [dataset_ids.index(i) for i in ids_test]

    Ftrain = df.values[ix_train, 1:]
    Fval = df.values[ix_val, 1:]
    Ftest = df.values[ix_test, 1:]

    '''add Normalize'''
    df_norm = df.fillna(-1)
    df_norm = (df_norm - df_norm.mean())/df_norm.std()
    df_norm = df_norm.fillna(0)

    FtrainNorm = df.values[ix_train, 1:]
    FvalNorm =df.values[ix_val, 1:]
    FtestNorm  = df.values[ix_test, 1:]

    '''Get Pipeline Feats'''
    df = pd.read_json(fn_pipelines_feats)
    df = df.fillna(-1)

    df.pop('id')
    if dataset_name == 'pmf':
        df.pop('model')
        df.pop('pre-processor')

    if pipeline_ixs is not None:
        df = df.iloc[pipeline_ixs]

    for k,j in zip(df.keys(), df.dtypes):
        if j==object:
            df[k] = pd.to_numeric(df[k], errors='coerce').fillna('0').astype('int32')       

    FPipeline = df.values

    return Ytrain, Yval, Ytest, Ftrain, Fval, Ftest, FtrainNorm, FvalNorm, FtestNorm, FPipeline