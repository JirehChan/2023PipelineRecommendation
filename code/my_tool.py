import bo
import torch
import random
import argparse
import numpy as np
import pandas as pd
import numpy as np
from utils import transform_forward, transform_backward

'''load arguments'''
def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_name', default='pmf', help='pmf, 110-classifiers, openml')
    parser.add_argument(
        '--save_path', default='default', help='the path to save result')
    parser.add_argument(
        '--warm_path', default='default', help='the path to warm starters setting')
    parser.add_argument(
        '--data_path', default='None', help='the path to save data')
    
        
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
        '--warm_trained', type=int, default=0, help='is the warm starter is trained')
    parser.add_argument(
        '--bo_n_init', type=int, default=5, help='the number of pipelines for warm start')
    parser.add_argument(
        '--bo_n_iters', type=int, default=200, help='the number of pipelines for search')
    parser.add_argument(
        '--is_bayes', type=int, default=1)
    parser.add_argument(
        '--is_narrow', type=int, default=0)

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
def get_data(dataset_name='pmf', pipeline_ixs=None, save_path=None, nan_ratio=0, data_path='None', random_seed=0):
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

    if dataset_name == 'openml':
        Y = Y*100.

    '''train / test'''
    ids_train = None
    ids_val = None
    ids_test = None
    if data_path=='None':
        if fn_train_ix is None:
            if dataset_name=='openml':
                random_rank = np.random.permutation(dataset_ids)
                n_train = int(len(dataset_ids)*0.8)
                n_test = len(dataset_ids)-n_train
                ids_train = []
                ids_test = []

                my_nn = 0
                for i in random_rank:
                    Yi = Y[:, dataset_ids.index(i)]

                    if n_test>0 and len(Yi)-np.isnan(Yi).sum()>261:
                        ids_test.append(i)
                        n_test -=1
                        my_nn+=1
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

        np.save('../result/{}/{}/ids_train.npy'.format(dataset_name, save_path), np.array(ids_train))
        np.save('../result/{}/{}/ids_val.npy'.format(dataset_name, save_path), np.array(ids_val))
        np.save('../result/{}/{}/ids_test.npy'.format(dataset_name, save_path), np.array(ids_test))
    else:
        ids_train = np.load('../result/{}/{}/ids_train.npy'.format(dataset_name, data_path))
        ids_val = np.load('../result/{}/{}/ids_val.npy'.format(dataset_name, data_path))
        ids_test = np.load('../result/{}/{}/ids_test.npy'.format(dataset_name, data_path))

    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_val = [dataset_ids.index(i) for i in ids_val]
    ix_test = [dataset_ids.index(i) for i in ids_test]


    Ytrain = Y[:, ix_train]
    Yval = Y[:, ix_val]
    Ytest = Y[:, ix_test]

    setup_seed(random_seed)
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

    pipeline_names = df['id'].tolist()

    if dataset_name == 'pmf':
        df.pop('model')
        df.pop('pre-processor')

    if pipeline_ixs is not None:
        df = df.iloc[pipeline_ixs]

    for k,j in zip(df.keys(), df.dtypes):
        if j==object:
            df[k] = pd.to_numeric(df[k], errors='coerce').fillna('0').astype('int32')       

    FPipeline = df.values

    return Ytrain, Yval, Ytest, Ftrain, Fval, Ftest, FtrainNorm, FvalNorm, FtestNorm, FPipeline, pipeline_names


'''warm_starter test'''
def test_warmstarter(bo_n_init, Yt, Ft, do_print=False, warm_start='l1', 
    warm_starter=None, is_narrow=0, narrow_list=None):

    # warm start
    ix_init = warm_starter.recommend(Ft)

    accs = np.zeros([Yt.shape[1], bo_n_init])
    best_accs = np.zeros([Yt.shape[1], bo_n_init])

    for ind, (ixs, y_train) in enumerate(zip(ix_init, Yt.T)):
        n_init = 0
        best_acc = 0

        for i in ixs:
            if not np.isnan(y_train[i]):
                if (is_narrow) and (i not in narrow_list):
                    continue
                accs[ind][n_init] = y_train[i]
                
                if best_acc<y_train[i]:
                    best_acc = y_train[i]
                best_accs[ind][n_init] = best_acc

                n_init += 1

                if bo_n_init==n_init:
                    break
                
    return accs, best_accs, ix_init

'''Bayesian Search'''
def bo_search(m, bo_n_init, bo_n_iters, Ytrain, Ftrain, ftest, ytest,
        do_print=False, warm_start='l1', warm_starter=None, pipeline_ixs=None,
        is_narrow=0, narrow_list=None):
    
    preds = bo.BO(m.dim, m.kernel, bo.ei,
                  variance=transform_forward(m.variance))
    ix_evaled = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ybest_list = []

    # warm start
    ix_init = warm_starter.recommend(ftest)
    n_init = 0
    if is_narrow:
        ix_init = [i for i in ix_init if i in narrow_list]
    
    for ix in ix_init:
        if not np.isnan(ytest[ix]):
            preds.add(m.X[ix], ytest[ix])
            ix_evaled.append(ix)

            ix_candidates.remove(ix)

            yb = preds.ybest
            ybest_list.append(yb)

            if do_print:
                print('Iter: %d, %g [%d], Best: %g' % (l, ytest[ix], ix, yb))
            
            n_init +=1
            if n_init==bo_n_init:
                break
    
    if len(ybest_list)==0:
        ix_evaled.append(-1)
        ybest_list.append(0)

    while len(ybest_list)<bo_n_init:
        ix_evaled.append(ix_evaled[-1])
        ybest_list.append(ybest_list[-1])
    
    # Optimization
    for l in range(bo_n_init, bo_n_iters):
        if len(ix_candidates)==0:
            ix_evaled.append(-1)
            ybest_list.append(preds.ybest)
            continue

        i = preds.next(m.X[ix_candidates])

        ix = ix_candidates[i]
        preds.add(m.X[ix], ytest[ix])
        ix_evaled.append(ix)
        ix_candidates.remove(ix)
        ybest_list.append(preds.ybest)

        if do_print:
            print('Iter: %d, %g [%d], Best: %g' \
                                    % (l, ytest[ix], ix, preds.ybest))

    return np.asarray(ybest_list), ix_evaled

'''Random Search'''
def random_search(bo_n_iters, ytest, speed=1, do_print=False, pipeline_ixs=None, ndcgk=[None],
              is_narrow=0, narrow_list=None):
    """
    speed denotes how many random queries are performed per iteration.
    """
    
    ix_evaled = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ybest_list = []
    ndcg_list = []

    ybest = np.nan
    n_init = 0

    for l in range(bo_n_iters):
        for ll in range(speed):
            if len(ix_candidates)==0:
                ix_evaled.append(-1)
                continue

            random_rank = np.random.permutation(len(ix_candidates))

            if is_narrow and n_init<5:
                random_rank = [i for i in random_rank if random_rank[i] in narrow_list]
                ix = ix_candidates[random_rank[0]]
                if not np.isnan(ybest):
                    if ytest[ix] > ybest:
                        ybest = ytest[ix]
                    ix_evaled.append(ix)
                    ix_candidates.remove(ix)
                    n_init+=1
            else:
                ix = ix_candidates[random_rank[0]]
                
                if np.isnan(ybest):
                    ybest = ytest[ix]
                else:
                    if ytest[ix] > ybest:
                        ybest = ytest[ix]

                ix_evaled.append(ix)
                ix_candidates.remove(ix)

        ybest_list.append(ybest)

        if do_print:
            print('Iter: %d, %g [%d], Best: %g' % (l, ytest[ix], ix, ybest))

    return np.asarray(ybest_list), ix_evaled


'''Ours Search'''
def ours_search(m, bo_n_init, bo_n_iters, Ytrain, Ftrain, ftest, ytest,
        do_print=False, warm_start='l1', warm_starter=None, pipeline_ixs=None,
        is_narrow=0, narrow_list=None):

    ix_evaled = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ybest = 0
    ybest_list = []

    # warm start
    ix_init, init_score = warm_starter.recommend_with_score(ftest)
    n_init = 0
    if is_narrow:
        ix_init = [i for i in ix_init if i in narrow_list]
    
    for ix in ix_init:
        if not np.isnan(ytest[ix]):
            ix_evaled.append(ix)
            ix_candidates.remove(ix)
            
            if ytest[ix]>ybest:
                ybest = ytest[ix]
            ybest_list.append(ybest)
            
            n_init +=1
            if n_init==bo_n_iters:
                break

    while len(ybest_list)<bo_n_iters:
        ix_evaled.append(ix_evaled[-1])
        ybest_list.append(ybest_list[-1])
    
    return np.asarray(ybest_list), ix_evaled

'''
SMAC Search
'''
from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO 
#from smac.configspace import CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter
from smac.utils.io.traj_logging import TrajLogger
from smac.stats.stats import Stats

def smac_search(m, bo_n_init, bo_n_iters, Ytrain, Ftrain, ftest, ytest,
        do_print=False, warm_start='l1', warm_starter=None, pipeline_ixs=None,
        is_narrow=0, narrow_list=None, pipeline_names=None):
    ix_evaled = []
    init_names = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    all_candidates = ix_candidates.copy()
    id_evaled = []
    id_candidates = [str(i) for i in list(range(len(ix_candidates)))]
    candidate_names = [pipeline_names[i] for i in ix_candidates]
    ybest = 0
    ybest_list = []

    # warm start
    ix_init = warm_starter.recommend(ftest)
    n_init = 0
    if is_narrow:
        ix_init = [i for i in ix_init if i in narrow_list]
    
    for ix in ix_init:
        if not np.isnan(ytest[ix]):
            ix_evaled.append(ix)
            init_names.append(pipeline_names[ix])
            ix_candidates.remove(ix)
            candidate_names.remove(pipeline_names[ix])
            
            #id = str(all_candidates.index(ix))
            #id_evaled.append(id)
            #id_candidates.remove(id)
            
            if ytest[ix]>ybest:
                ybest = ytest[ix]
            ybest_list.append(ybest)
            
            n_init +=1
            if n_init==bo_n_init:
                break
    
    # optimization
    init_cs = ConfigurationSpace()
    #init_ids = OrdinalHyperparameter('name', [str(i) for i in id_evaled], default_value=str(id_evaled[0]))
    init_ids = OrdinalHyperparameter('name', init_names, default_value=init_names[0])
    init_cs.add_hyperparameter(init_ids)

    optim_cs = ConfigurationSpace()
    #optim_ids = OrdinalHyperparameter('name', [str(i) for i in id_candidates], default_value=str(id_candidates[0]))
    optim_ids = OrdinalHyperparameter('name', candidate_names, default_value=candidate_names[0])
    optim_cs.add_hyperparameter(optim_ids)
    scenario = Scenario({'run_obj':'quality', 'runcount-limit':bo_n_iters, 'cs':optim_cs, 'deterministic':'true',
        'initial_incumbent':"DEFAULT"})

    def cfg_ours(cfg):
        #print(cfg['name'])
        ix = pipeline_names.index(cfg['name'])
        #id = int(cfg['names'])
        #ix = all_candidates[int(cfg['name'])]
        
        ix_evaled.append(ix)
        if ytest[ix]>ybest_list[-1]:
            ybest_list.append(ytest[ix])
        else:
            ybest_list.append(ybest_list[-1])

        return np.float32(ytest[ix])

    init_confs = []
    for name in init_names:
        conf = init_cs.sample_configuration()
        conf['name'] = name
        init_confs.append(conf)
    

    smac = SMAC4HPO(scenario=scenario, rng=0, tae_runner=cfg_ours, 
        smbo_kwargs={'min_samples_model':10},
        initial_design_kwargs={'configs':init_confs, 'init_budget':bo_n_init}
        )#initial_design_kwargs={"cs": init_cs, 'init_budget':100, 'ta_run_limit':100},)
    
    incumbent = smac.optimize()
    
    while len(ybest_list)<bo_n_iters:
        ix_evaled.append(ix_evaled[-1])
        ybest_list.append(ybest_list[-1])
        
    return np.asarray(ybest_list[bo_n_init:]), ix_evaled[bo_n_init:]