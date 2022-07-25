import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.impute
import time
import torch
import kernels
import gplvm
from utils import transform_forward, transform_backward
import bo
import os
import my_recommends as mr
import my_tool as mt
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import multiprocessing

torch.set_default_tensor_type(torch.FloatTensor)
'''
fn_data = '../data/pmf/performance.csv'
fn_train_ix = '../data/pmf/ids_train.csv'
fn_test_ix = '../data/pmf/ids_test.csv'
fn_data_feats = '../data/pmf/dataset_feats.csv'
fn_pipelines_feats = '../data/pmf/pipelines.json'
'''
    

def train(m, optimizer, f_callback=None, f_stop=None):

    it = 0
    while True:

        try:
            t = time.time()

            optimizer.zero_grad()
            nll = m()
            nll.backward()
            optimizer.step()

            it += 1
            t = time.time() - t

            if f_callback is not None:
                f_callback(m, nll, it, t)

            # f_stop should not be a substantial portion of total iteration time
            if f_stop is not None and f_stop(m, nll, it, t):
                break

        except KeyboardInterrupt:
            break

    return m

def test_warmstarter(bo_n_init, Yt, Ft, do_print=False, warm_start='l1', 
    warm_starter=None):

    # warm start
    ix_init = warm_starter.recommend(Ft)

    accs = np.zeros([Yt.shape[1], bo_n_init])

    for ind, (ixs, y_train) in enumerate(zip(ix_init, Yt.T)):
        n_init = 0

        for i in ixs:
            if not np.isnan(y_train[i]):
                accs[ind][n_init] = y_train[i]
                n_init += 1
                if n_init>=bo_n_init:
                    break
    return accs


def bo_search(m, bo_n_init, bo_n_iters, Ytrain, Ftrain, ftest, ytest,
              do_print=False, warm_start='l1', warm_starter=None, pipeline_ixs=None, topk=[1], ndcgk=[None]):
    """
    initializes BO with L1 warm-start (using dataset features). returns a
    numpy array of length bo_n_iters holding the best performance attained
    so far per iteration (including initialization).

    bo_n_iters includes initialization iterations, i.e., after warm-start, BO
    will run for bo_n_iters - bo_n_init iterations.
    """

    preds = bo.BO(m.dim, m.kernel, bo.ei,
                  variance=transform_forward(m.variance))
    ix_evaled = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ybest_list = []
    ndcg_list = []
    topk_list = []

    # warm start
    ix_init = warm_starter.recommend(ftest)
    n_init = 0

    best_ix = ftest.argmax()
    for top_k in topk:
        if best_ix in ix_init[:top_k]:
            topk_list.append(True)
        else:
            topk_list.append(False)
    
    ix_to_rank = []
    for i in ix_init:
        if i in ix_candidates:
            ix_to_rank.append(i)
    ix_to_rank = np.array(ix_to_rank)#np.intersect1d(ix_init, ix_candidates)

    for ndcg_k in ndcgk:
        if len(ix_to_rank)<2:
            score=0
        else:
            score = ndcg_score([ytest[ix_to_rank]], [np.flipud(ix_to_rank)], k=ndcg_k)
        ndcg_list.append(score)
    
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
    '''
    for l in range(bo_n_init):
        ix = ix_init[l]
        if not np.isnan(ytest[ix]):
            preds.add(m.X[ix], ytest[ix])
            ix_evaled.append(ix)
            ix_candidates.remove(ix)
        yb = preds.ybest
        if yb is None:
            yb = np.nan
        ybest_list.append(yb)

        if do_print:
            print('Iter: %d, %g [%d], Best: %g' % (l, ytest[ix], ix, yb))
    '''

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

    return np.asarray(ybest_list), ix_evaled, ndcg_list, topk_list

def bowarm_search(m, bo_n_init, bo_n_iters, Ytrain, Ftrain, ftest, ytest,
              do_print=False, warm_start='l1', warm_starter=None, pipeline_ixs=None, topk=[1], ndcgk=[None]):

    preds = bo.BOwarm(m.dim, m.kernel, bo.ei_warm,
                  variance=transform_forward(m.variance))
    ix_evaled = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ybest_list = []
    ndcg_list = []
    topk_list = []

    # warm start
    ix_init, init_score = warm_starter.recommend_with_score(ftest)
    n_init = 0

    best_ix = ftest.argmax()
    for top_k in topk:
        if best_ix in ix_init[:top_k]:
            topk_list.append(True)
        else:
            topk_list.append(False)
    
    ix_to_rank = []
    for i in ix_init:
        if i in ix_candidates:
            ix_to_rank.append(i)
    ix_to_rank = np.array(ix_to_rank)#np.intersect1d(ix_init, ix_candidates)

    for ndcg_k in ndcgk:
        score = ndcg_score([ytest[ix_to_rank]], [np.flipud(ix_to_rank)], k=ndcg_k)
        ndcg_list.append(score)
    
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

    # Optimization
    for l in range(bo_n_init, bo_n_iters):
        if len(ix_candidates)==0:
            ix_evaled.append(-1)
            ybest_list.append(preds.ybest)
            continue
        means, res, i = preds.next(m.X[ix_candidates], init_score[ix_candidates])

        ix = ix_candidates[i]
        preds.add(m.X[ix], ytest[ix])
        ix_evaled.append(ix)
        ix_candidates.remove(ix)
        ybest_list.append(preds.ybest)

        if do_print:
            print('Iter: %d, %g [%d], Best: %g' \
                                    % (l, ytest[ix], ix, preds.ybest))

    return np.asarray(ybest_list), ix_evaled, ndcg_list, topk_list


def random_search(bo_n_iters, ytest, speed=1, do_print=False, pipeline_ixs=None, ndcgk=[None]):
    """
    speed denotes how many random queries are performed per iteration.
    """
    
    ix_evaled = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ybest_list = []
    ndcg_list = []

    ybest = np.nan

    for l in range(bo_n_iters):
        for ll in range(speed):
            if len(ix_candidates)==0:
                ix_evaled.append(-1)
                continue
            random_rank = np.random.permutation(len(ix_candidates))
            if l==0 and ll==0:
                for ndcg_k in ndcgk:
                    ndcg_list.append(ndcg_score([ytest[ix_candidates]], [np.flipud(random_rank)], k=ndcg_k))

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

    return np.asarray(ybest_list), ix_evaled, ndcg_list

if __name__=='__main__':
    import warnings 
    warnings.filterwarnings('ignore')

    '''set & load arguments'''
    args = mt.parse_arg()
    
    # set random seed
    mt.setup_seed(args.random_seed)

    # check the save path
    if not os.path.exists('../result/{}'.format(args.save_name)):
        os.makedirs('../result/{}'.format(args.save_name))

    # train and evaluation settings
    Q = 20
    batch_size = 50
    n_epochs = 300
    lr = 1e-7
    N_max = 1000
    save_checkpoint = False
    fn_checkpoint = None
    checkpoint_period = 50

    # use for warm start
    pipeline_ixs = None
    n_pipelines = 120
    #pipeline_ixs = list(range(20))
    #[3715, 1830, 5740, 6776, 3939, 1605, 3263, 2497, 1444, 5542, 3769, 3704, 6430, 6367, 2813, 3235, 3277, 4063, 3629, 6465, 4393, 724, 3206, 6877, 1974, 3806, 5709, 13, 5022, 4426, 3389, 2296, 967, 2334, 1491, 3965, 4781, 1509, 4968, 2220, 5565, 5317, 63, 3571, 2588, 5339, 3222, 3664, 3225, 5156, 5091, 3784, 909, 5520, 813, 1736, 5866, 225, 5842, 2004, 5011, 3682, 1821, 4840, 6519, 288, 3204, 3592, 3766, 1929, 4895, 5826, 1712, 1378, 5689, 2164, 6766, 554, 4405, 37, 6786, 547, 730, 6193, 5059, 1017, 1563, 838, 477, 3369, 750, 4639, 937, 5557, 1206, 5397, 3103, 4587, 2324, 3098]

    # use for save
    train_accs = {}
    val_accs = {}
    test_accs = {}

    '''train'''
    # load dataset
    Ytrain, Yval, Ytest, Ftrain, Fval, Ftest, FtrainNorm, FvalNorm, FtestNorm, FPipeline = \
        mt.get_data(args.dataset_name, pipeline_ixs, args.save_name, args.nan_ratio)
    
    maxiter = int(Ytrain.shape[1]/batch_size*n_epochs)

    def f_stop(m, v, it, t):

        if it >= maxiter-1:
            print('maxiter (%d) reached' % maxiter)
            return True

        return False

    varn_list = []
    logpr_list = []
    t_list = []
    def f_callback(m, v, it, t):
        varn_list.append(transform_forward(m.variance).item())
        logpr_list.append(m().item()/m.D)
        if it == 1:
            t_list.append(t)
        else:
            t_list.append(t_list[-1] + t)

        if save_checkpoint and not (it % checkpoint_period):
            torch.save(m.state_dict(), fn_checkpoint + '_it%d.pt' % it)

        print('it=%d, f=%g, varn=%g, t: %g'
              % (it, logpr_list[-1], transform_forward(m.variance), t_list[-1]))

    # create initial latent space with PCA, first imputing missing observations
    imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    X = sklearn.decomposition.PCA(Q).fit_transform(
                                            imp.fit(Ytrain).transform(Ytrain))

    # define model
    kernel = kernels.Add(kernels.RBF(Q, lengthscale=None), kernels.White(Q))
    m = gplvm.GPLVM(Q, X, Ytrain, kernel, N_max=N_max, D_max=batch_size)

    #"""
    if save_checkpoint:
        torch.save(m.state_dict(), fn_checkpoint + '_it%d.pt' % 0)
    
    # optimize
    if args.model_path == 'None':
        print('training...')
        optimizer = torch.optim.SGD(m.parameters(), lr=lr)
        m = train(m, optimizer, f_callback=f_callback, f_stop=f_stop)
        if save_checkpoint:
            torch.save(m.state_dict(), fn_checkpoint + '_itFinal.pt')
        torch.save(m.state_dict(), '../result/{}/pmf.pt'.format(args.save_name))
    else:
        print('loading...')
        m.load_state_dict(torch.load('../result/{}/pmf.pt'.format(args.model_path)))

    # train for warm-start
    print('training for warm start ...')

    f = open('../result/{}/warm_starters-{}.txt'.format(args.warm_start, args.part_name), 'r')
    exec(f.read())

    for k in warm_starters.keys():

        print('- {:10}:'.format(k))
        if k[:2]=='nn':
            if pipeline_ixs is not None:
                warm_starters[k].train(Ytrain, FtrainNorm)
            else:
                warm_starters[k].train(Ytrain, FtrainNorm)
        elif k[:3]=='pmm' or k[:4]=='bpmm' or k[:3]=='reg':
            if pipeline_ixs is not None:
                warm_starters[k].train(Ytrain, FtrainNorm, FPipeline)
            else:
                warm_starters[k].train(Ytrain, FtrainNorm, FPipeline)
        else:
            if pipeline_ixs is not None:
                warm_starters[k].train(Ytrain, FtrainNorm)#Ftrain)
            else:
                warm_starters[k].train(Ytrain, FtrainNorm)#Ftrain)
    
    '''Test warm start'''
    for k in warm_starters.keys():
        train_accs[k] = test_warmstarter(args.bo_n_init, Ytrain, FtrainNorm, 
                        do_print=False, warm_start=k,
                        warm_starter=warm_starters[k])

        val_accs[k] = test_warmstarter(args.bo_n_init, Yval, FvalNorm, 
                        do_print=False, warm_start=k,
                        warm_starter=warm_starters[k])
        

    '''Evaluate'''
    regrets_list = {}
    evaled_list = {}
    ndcg_list = {}
    topk_list = {}

    for k in ['random1x', 'random2x', 'random4x']+list(warm_starters.keys()):
        regrets_list[k] = np.zeros((args.bo_n_iters, Ytest.shape[1]))
        test_accs[k] = np.zeros((args.bo_n_iters, Ytest.shape[1]))
        evaled_list[k] = []
        ndcg_list[k] = []
        topk_list[k] = [] 

        if k[:3]=='pmm' or k[:4]=='bpmm':
            regrets_list[k+'(warm)'] = np.zeros((args.bo_n_iters, Ytest.shape[1]))
            test_accs[k+'(warm)'] = np.zeros((args.bo_n_iters, Ytest.shape[1]))
            evaled_list[k+'(warm)'] = []
            ndcg_list[k+'(warm)'] = []
            topk_list[k+'(warm)'] = []
            
            #regrets_list[k+'(bo-warm)'] = np.zeros((args.bo_n_iters, Ytest.shape[1]))
            #evaled_list[k+'(bo-warm)'] = []
            #ndcg_list[k+'(bo-warm)'] = []
            #topk_list[k+'(bo-warm)'] = []

    print('evaluating...')
    topk = [1, 5, 10, 20, 50]
    ndcgk = [None, 5, 10, 20]

    with torch.no_grad():
        Ytest = Ytest.astype(np.float32)

        for d in tqdm(np.arange(Ytest.shape[1])):
            ybest = np.nanmax(Ytest[:,d])
            
            for k in tqdm(['random1x', 'random2x', 'random4x']+list(warm_starters.keys()), leave=False):
                topks = None
                if k[:6]=='random':
                    regrets, ix_evaled, ndcgs = random_search(
                        args.bo_n_iters, Ytest[:,d], speed=int(k[6]),
                        pipeline_ixs=pipeline_ixs, ndcgk=ndcgk)
                elif k[:2]=='nn' or k[:3]=='pmm' or k[:4]=='bpmm':
                    regrets, ix_evaled, ndcgs, topks = bo_search(m, args.bo_n_init, args.bo_n_iters,
                                                    Ytrain, FtrainNorm, FtestNorm[d,:],
                                                    Ytest[:,d], warm_start=k,
                                                    warm_starter=warm_starters[k],
                                                    pipeline_ixs=pipeline_ixs, topk=topk, ndcgk=ndcgk)
                else:  
                    regrets, ix_evaled, ndcgs, topks = bo_search(m, args.bo_n_init, args.bo_n_iters,
                                                    Ytrain, FtrainNorm, Ftest[d,:],
                                                    Ytest[:,d], warm_start=k,
                                                    warm_starter=warm_starters[k],
                                                    pipeline_ixs=pipeline_ixs, topk=topk, ndcgk=ndcgk)

                test_accs[k][:,d] = regrets
                regrets_list[k][:,d] = ybest - regrets
                evaled_list[k].append(ix_evaled)
                ndcg_list[k].append(ndcgs)
                topk_list[k].append(topks)
            
            for k in tqdm(['random1x', 'random2x', 'random4x']+list(warm_starters.keys()), leave=False):
                if k[:3]=='pmm' or k[:4]=='bpmm':

                    '''
                    regrets, ix_evaled, ndcgs, topks = bowarm_search(m, args.bo_n_init, args.bo_n_iters,
                                                    Ytrain, Ftrain, FtestNorm[d,:],
                                                    Ytest[:,d], warm_start=k,
                                                    warm_starter=warm_starters[k],
                                                    pipeline_ixs=pipeline_ixs, topk=topk, ndcgk=ndcgk)
                
                    regrets_list[k+'(bo-warm)'][:,d] = ybest - regrets
                    evaled_list[k+'(bo-warm)'].append(ix_evaled)
                    ndcg_list[k+'(bo-warm)'].append(ndcgs)
                    topk_list[k+'(bo-warm)'].append(topks)
                    '''
                    
                    regrets, ix_evaled, ndcgs, topks = bo_search(m, args.bo_n_iters, args.bo_n_iters,
                                                    Ytrain, FtrainNorm, FtestNorm[d,:],
                                                    Ytest[:,d], warm_start=k,
                                                    warm_starter=warm_starters[k],
                                                    pipeline_ixs=pipeline_ixs, topk=topk, ndcgk=ndcgk)
                
                    test_accs[k+'(warm)'][:,d] = regrets
                    regrets_list[k+'(warm)'][:,d] = ybest - regrets
                    evaled_list[k+'(warm)'].append(ix_evaled)
                    ndcg_list[k+'(warm)'].append(ndcgs)
                    topk_list[k+'(warm)'].append(topks)
        
        import pickle
        f = open('../result/{}/trainaccs-{}.pkl'.format(args.save_name, args.part_name),'wb')
        pickle.dump(train_accs,f)
        f.close()

        f = open('../result/{}/valaccs-{}.pkl'.format(args.save_name, args.part_name),'wb')
        pickle.dump(val_accs,f)
        f.close() 

        f = open('../result/{}/testaccs-{}.pkl'.format(args.save_name, args.part_name),'wb')
        pickle.dump(test_accs,f)
        f.close() 

        f = open('../result/{}/regrets-{}.pkl'.format(args.save_name, args.part_name),'wb')
        pickle.dump(regrets_list,f)
        f.close()

        f = open('../result/{}/evaled-{}.pkl'.format(args.save_name, args.part_name),'wb')
        pickle.dump(evaled_list, f)
        f.close()

        f = open('../result/{}/ndcgs-{}.pkl'.format(args.save_name, args.part_name),'wb')
        pickle.dump(ndcg_list,f)
        f.close()

        f = open('../result/{}/topks-{}.pkl'.format(args.save_name, args.part_name),'wb')
        pickle.dump(topk_list,f)
        f.close()

        

