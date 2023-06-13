import os
import pickle
import warnings 
import numpy as np

from collections import Counter

import time
import torch
import kernels
import gplvm
import sklearn.impute
from utils import transform_forward, transform_backward

import my_tool as mt
import my_recommends as mr
from tqdm import tqdm


from scipy.stats import rankdata
from sklearn.metrics import ndcg_score


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

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    result_to_save = {}#{'random1x':{}}

    '''set & load arguments'''
    args = mt.parse_arg()

    mt.setup_seed(args.random_seed)

    if not os.path.exists('../result/{}/{}'.format(args.dataset_name, args.save_path)):
        os.makedirs('../result/{}/{}'.format(args.dataset_name, args.save_path))
    
    '''load dataset'''
    pipeline_ixs= None # used when the pipeline is setting
    Ytrain, Yval, Ytest, Ftrain, Fval, Ftest, FtrainNorm, FvalNorm, FtestNorm, FPipeline, pipeline_names = \
        mt.get_data(args.dataset_name, pipeline_ixs, args.save_path, args.nan_ratio, args.data_path, args.random_seed)
    
    narrow_pipeline = None # used for only choose the pipelines which used on all datasets
    if args.is_narrow:
        narrow_pipeline = list(Counter(np.nanargmax(Ytrain,axis=0)).keys())
    
    '''train for warm-starters'''
    print('\ntraining for warm starters ...')

    f = open('../settings/{}/warm_starters-{}.txt'.format(args.warm_path, args.part_name), 'r')
    exec(f.read())

    warm_train_losses = {}
    warm_val_losses = {}

    if args.warm_trained:
        for k in warm_starters.keys():
            warm_starters[k].FPipeline = FPipeline
            warm_starters[k].model = torch.load(warm_starters[k].kwargs['save_path'])
    else:
        for k in warm_starters.keys():
            result_to_save[k] = {}
            mt.setup_seed(args.random_seed)
            start_time = time.time()
            print('- {:10}:'.format(k))
            if k[:6]=='clf-nn' or k[:7]=='clf-knn' or k[:6]=='clf-rf' or k[:7]=='rank-ar' or k[:6]=='autosk':
                warm_starters[k].train(Ytrain, FtrainNorm)
            elif k[:3]=='pmm' or k[:6]=='reg-nn' or k[:6]=='reg-ab':
                warm_starters[k].train(Ytrain, FtrainNorm, FPipeline)
            elif k[:4]=='bpmm':
                warm_train_losses[k], warm_val_losses[k] = warm_starters[k].train(Ytrain, FtrainNorm, FPipeline)
            else:
                warm_starters[k].train(Ytrain, FtrainNorm)
            end_time = time.time()
            result_to_save[k]['train_time'] = end_time - start_time

    
    '''Test Warm Starters'''
    print('\ntesting for warm starters ...')

    for k in warm_starters.keys():
        print('[ {:^10} ]'.format(k))

        accs, best_accs, ixs = mt.test_warmstarter(
            args.bo_n_init, Ytrain, FtrainNorm, 
            do_print=False, warm_start=k,
            warm_starter=warm_starters[k],
            is_narrow=args.is_narrow, narrow_list=narrow_pipeline
        )
        result_to_save[k]['train_accs'], result_to_save[k]['train_best_accs'], result_to_save[k]['train_ixs'] = accs[:,:args.bo_n_init], best_accs[:,:10], ixs

        print('train : ', end='')
        for acc in result_to_save[k]['train_best_accs'].mean(axis=0)[:5]:
            print('{:5.2f} | '.format(acc), end='')
        print()

        accs, best_accs, ixs = mt.test_warmstarter(
            args.bo_n_init, Yval, FvalNorm, 
            do_print=False, warm_start=k,
            warm_starter=warm_starters[k],
            is_narrow=args.is_narrow, narrow_list=narrow_pipeline
        )
        result_to_save[k]['val_accs'], result_to_save[k]['val_best_accs'], result_to_save[k]['val_ixs'] = accs[:,:args.bo_n_init], best_accs[:,:10], ixs
        print('val   : ', end='')
        for acc in result_to_save[k]['val_best_accs'].mean(axis=0)[:5]:
            print('{:5.2f} | '.format(acc), end='')
        print()

        accs, best_accs, ixs = mt.test_warmstarter(
            args.bo_n_init, Ytest, FtestNorm, 
            do_print=False, warm_start=k,
            warm_starter=warm_starters[k],
            is_narrow=args.is_narrow, narrow_list=narrow_pipeline
        )
        
        result_to_save[k]['ndcg_scores'] = [
            ndcg_score(accs[:,:5], np.tile(np.arange(accs[:,:5].shape[1],0,-1), (accs[:,:5].shape[0], 1))),
            ndcg_score(accs[:,:10], np.tile(np.arange(accs[:,:10].shape[1],0,-1), (accs[:,:10].shape[0], 1))),
            ndcg_score(accs[:,:20], np.tile(np.arange(accs[:,:20].shape[1],0,-1), (accs[:,:20].shape[0], 1))),
            ndcg_score(accs[:,:50], np.tile(np.arange(accs[:,:50].shape[1],0,-1), (accs[:,:50].shape[0], 1))),]
        result_to_save[k]['test_accs'], result_to_save[k]['test_best_accs'], result_to_save[k]['test_ixs'] = accs[:,:args.bo_n_init], best_accs[:,:10], ixs
        
        print('test  : ', end='')
        for acc in result_to_save[k]['test_best_accs'].mean(axis=0)[:5]:
            print('{:5.2f} | '.format(acc), end='')

        print(' || {:.4f} | {:.4f} | {:.4f} | {:.4f}'.format(\
            result_to_save[k]['ndcg_scores'][0], result_to_save[k]['ndcg_scores'][1],
            result_to_save[k]['ndcg_scores'][2], result_to_save[k]['ndcg_scores'][3]))
        
        print()

    if args.is_bayes==1:
        start_time = time.time()

        # train and evaluation settings
        Q = 20
        batch_size = 50
        n_epochs = 300
        lr = 1e-7
        N_max = 1000
        save_checkpoint = False
        fn_checkpoint = None
        checkpoint_period = 50
        
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

        if save_checkpoint:
            torch.save(m.state_dict(), fn_checkpoint + '_it%d.pt' % 0)
        
        # optimize
        if args.model_path == 'None':
            print('training...')
            optimizer = torch.optim.SGD(m.parameters(), lr=lr)
            m = train(m, optimizer, f_callback=f_callback, f_stop=f_stop)
            if save_checkpoint:
                torch.save(m.state_dict(), fn_checkpoint + '_itFinal.pt')
            torch.save(m.state_dict(), './{}-{}.pt'.format(args.dataset_name, args.save_path))
        else:
            print('loading...')
            m.load_state_dict(torch.load('./{}-{}.pt'.format(args.dataset_name, args.save_path)))

        end_time = time.time()
        result_to_save['pmf-l1']['train_time'] = end_time - start_time

        '''Evaluate'''
        test_accs = {}

        for k in list(warm_starters.keys()):
            result_to_save[k]['test_time'] = 0.
            result_to_save[k]['bo_accs'] = np.zeros((args.bo_n_iters, Ytest.shape[1]))
        
        with torch.no_grad():
            Ytest = Ytest.astype(np.float32)

            for d in tqdm(np.arange(Ytest.shape[1])):
                ybest = np.nanmax(Ytest[:,d])

                for k in tqdm(list(warm_starters.keys()), leave=False):
                    topks = None

                    start_time = time.time()

                    if k[:6]=='random':
                        accs, _ = mt.random_search(args.bo_n_iters, Ytest[:,d], speed=1,
                            pipeline_ixs=pipeline_ixs, is_narrow=args.is_narrow,
                            narrow_list=narrow_pipeline)
                    elif k[:4]=='bpmm':
                        accs, _ = mt.ours_search(m, args.bo_n_init, args.bo_n_iters,
                                                        Ytrain, Ftrain, FtestNorm[d,:],
                                                        Ytest[:,d], warm_start=k,
                                                        warm_starter=warm_starters[k],
                                                        pipeline_ixs=pipeline_ixs, is_narrow=args.is_narrow,
                                                        narrow_list=narrow_pipeline )
                    elif k[:6]=='autosk':
                        accs, _ = mt.smac_search(m, args.bo_n_init, args.bo_n_iters,
                                                        Ytrain, FtrainNorm, FtestNorm[d,:],
                                                        Ytest[:,d], warm_start=k,
                                                        warm_starter=warm_starters[k],
                                                        pipeline_ixs=pipeline_ixs, is_narrow=args.is_narrow,
                                                        narrow_list=narrow_pipeline, pipeline_names=pipeline_names)
                    else:
                        accs, _ = mt.bo_search(m, args.bo_n_init, args.bo_n_iters,
                                                        Ytrain, FtrainNorm, FtestNorm[d,:],
                                                        Ytest[:,d], warm_start=k,
                                                        warm_starter=warm_starters[k],
                                                        pipeline_ixs=pipeline_ixs, is_narrow=args.is_narrow,
                                                        narrow_list=narrow_pipeline)
                    end_time = time.time()

                    result_to_save[k]['test_time'] += (end_time-start_time)
                    result_to_save[k]['bo_accs'][:,d] = accs

    f = open('../result/{}/{}/result-{}.pkl'.format(args.dataset_name, args.save_path, args.part_name), 'wb')
    pickle.dump(result_to_save, f)
    f.close()
