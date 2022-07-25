import torch
import gplvm
import numpy as np
import scipy.stats as st

def ei(fm, fv, ybest, xi=0.01, eps=1e-12):

    fsd = torch.sqrt(fv) + eps
    gamma = (fm - ybest - xi)/fsd

    return fsd * (torch.distributions.Normal(0,1).cdf(gamma) * gamma
                  + torch.distributions.Normal(0,1).log_prob(gamma).exp())

def ei_warm(fm, fv, ybest, init_score, xi=0.01, eps=1e-12, warm_weight=0.1):

    fsd = torch.sqrt(fv) + eps
    gamma = (fm - ybest - xi)/fsd
    ei_score = fsd * (torch.distributions.Normal(0,1).cdf(gamma) * gamma
                  + torch.distributions.Normal(0,1).log_prob(gamma).exp())

    return (1-warm_weight) * ei_score + warm_weight * init_score.reshape(-1,1)


def init_l1(Ytrain, Ftrain, ftest, n_init=5):

    dis = np.abs(Ftrain - ftest).sum(axis=1)
    ix_closest = np.argsort(dis)[:n_init]
    ix_nonnan_pipelines \
            = np.where(np.invert(np.isnan(Ytrain[:,ix_closest].sum(axis=1))))[0]
    ranks = np.apply_along_axis(st.rankdata, 0,
                                Ytrain[ix_nonnan_pipelines[:,None],ix_closest])
    ave_pipeline_ranks = ranks.mean(axis=1)
    ix_init = ix_nonnan_pipelines[np.argsort(ave_pipeline_ranks)[::-1]]

    return ix_init[:n_init]

class BO(gplvm.GP):

    def __init__(self, dim, kernel, acq_func, **kwargs):
        super(BO, self).__init__(dim, np.asarray([]), np.asarray([]), kernel,
                                 **kwargs)

        self.acq_func = acq_func
        self.ybest = None
        self.xbest = None

    def add(self, xnew, ynew):

        xnew_ = torch.tensor(xnew, dtype=self.X.dtype).reshape((1,-1))
        self.X = torch.cat((self.X, xnew_))
        ynew_ = torch.tensor([ynew], dtype=self.y.dtype)
        self.y = torch.cat((self.y, ynew_))
        if self.ybest is None or ynew_ > self.ybest:
            self.ybest = ynew_
            self.xbest = xnew_
        self.N += 1

    def next(self, Xcandidates):

        if not self.N:
            return torch.randperm(Xcandidates.size()[0])[0]

        fmean, fvar = self.posterior(Xcandidates)
        acq_res = self.acq_func(fmean, fvar, self.ybest)

        return torch.argmax(acq_res)



class BOwarm(gplvm.GP):

    def __init__(self, dim, kernel, acq_func, **kwargs):
        super(BOwarm, self).__init__(dim, np.asarray([]), np.asarray([]), kernel,
                                 **kwargs)

        self.acq_func = acq_func
        self.ybest = None
        self.xbest = None

    def add(self, xnew, ynew):

        xnew_ = torch.tensor(xnew, dtype=self.X.dtype).reshape((1,-1))
        self.X = torch.cat((self.X, xnew_))
        ynew_ = torch.tensor([ynew], dtype=self.y.dtype)
        self.y = torch.cat((self.y, ynew_))
        if self.ybest is None or ynew_ > self.ybest:
            self.ybest = ynew_
            self.xbest = xnew_
        self.N += 1

    def next(self, Xcandidates, init_score):

        if not self.N:
            return torch.randperm(Xcandidates.size()[0])[0]

        fmean, fvar = self.posterior(Xcandidates)
        acq_res = self.acq_func(fmean, fvar, self.ybest, init_score)

        return fmean, acq_res, torch.argmax(acq_res)