#!/usr/bin/env python
import numpy as np

def smooth_ccdf(ccdf, P=10):
    # smooth by raising CDF to the Pth power
    # tweaked for numerical stability
    return np.where(ccdf > 1e-6, 1-(1-ccdf)**P, P*ccdf)

max_symbols = 3000
def constant_plus_geometric_ccdf(const, tau, P=1):
    y = np.clip(np.exp(-(np.arange(max_symbols)-const)/tau), 0, 1)
    y[-1] = 0.
    y = smooth_ccdf(y, P)
    return y

def condition_ccdf(ccdf, i):
    return np.clip(ccdf[:,np.newaxis]/ccdf[i], 0, 1).squeeze()

class strategizer(object):
    def __init__(self, ccdf, ccdf_x=None):
        self.ccdf = ccdf
        self.max_symbols = ccdf.size
        self.steady_state_index = min(2400, self.max_symbols-1)
        self.k = np.arange(self.max_symbols)
        self.m = self.k if ccdf_x is None else ccdf_x
        # complimentary conditional cdf
        try:
            old = np.seterr(invalid='ignore', divide='ignore', over='ignore')
            self.cc = condition_ccdf(ccdf, np.arange(self.max_symbols))
        finally:
            np.seterr(**old)
        self.cc[-1,-1] = 0.
        self.epsilon = 1e-40
        if np.any(ccdf < self.epsilon):
            self.ccdf_valid_index = np.where(ccdf < self.epsilon)[0][0]
        else:
            self.ccdf_valid_index = self.max_symbols-1
        self.t = np.empty(self.max_symbols, float)
        self.st = np.empty(self.max_symbols, int)
        self.i_init = min(self.ccdf_valid_index, self.steady_state_index)
        self.n_f = None
        self.a0 = (np.diff(np.log(self.ccdf[self.ccdf_valid_index-5:self.ccdf_valid_index]))).sum()/4.
        if self.a0 >= 0. or self.a0 < -5.:
            zero_after = np.where(self.ccdf==0)[0]
            if zero_after.size:
                time_constant = np.amin(zero_after)/4.
            else:
                time_constant = self.ccdf.size/4.
            self.a0 = -1./time_constant
        self.b0 = 1-np.exp(self.a0)
    def steady_state(self, n_f, c=20):
        try:
            old = np.seterr(invalid='ignore', divide='ignore')
            t0 = self.max_symbols
            a0, b0 = self.a0, self.b0
            for i in xrange(c):
                t1 = 1/b0 - np.log(b0*(t0+n_f))/a0
                if np.isnan(t1):
                    return t0+n_f, self.max_symbols-1
                dt = t1 - t0
                if np.abs(dt) < 10.:
                    dt *= 1.8
                t0 += dt
            st0 = int(np.ceil(1-np.log(b0*(t0+n_f))/a0))
            return t0+n_f, st0
        finally:
            np.seterr(**old)
    def compute(self, n_f):
        i = self.i_init
        cc, k, t, st, m = self.cc, self.k, self.t, self.st, self.m
        t[i:], self.st0 = self.steady_state(n_f)
        st[i:] = self.st0
        t[:i] = np.inf
        while i>0:
            l = 1
            I = i+1 + st[i]
            i -= l
            et = cc[i+1:I,i:i+l]*t[i+1:I,np.newaxis] + m[i+1:I,np.newaxis]
            j = et.argmin(0)
            t[i:i+l] = et[j,k[:l]] + n_f - m[i:i+l]
            st[i:i+l] = j-k[:l]+1
        self.n_f = n_f
    def strategy(self, n_f):
        if n_f != self.n_f:
            self.compute(n_f)
        return self.st
    def summary(self, n_f):
        x = self.strategy(n_f)
        max = self.max_symbols
        ll = 0
        l = []
        while True:
            j = x[ll]
            ll += j
            if ll >= max: break
            l.append(ll)
        if l[-1] != max-1:
            l.append(max-1)
        return np.array(l)

def strategy_time(l, ccdf, x_ccdf, n_f, pseudo=False):
    n = x_ccdf[l]
    q = np.clip(-np.diff(np.r_[1, ccdf[l]]), 0, 1)
    p_success = 1-ccdf[l[-1]]
    q[-1] += 1-p_success
    feedback_time = np.sum(q*np.arange(1,q.size+1))*n_f
    data_time = np.sum(q*n)
    return feedback_time + data_time - pseudo*n_f, p_success

def memoryless_strategy(tau, n_f):
    gamma = 1 + (1.*n_f) / tau
    return tau * reduce(lambda n,_: np.log(n+gamma), np.empty(10), 0.)

def memoryless_efficiency(tau, n_f):
    m = np.around(memoryless_strategy(tau, n_f))
    en = 1/(1-np.exp(-1./tau))
    return en*(1-np.exp(-m/tau))/(m+n_f)

def c_p_g_efficiency(const, tau, n_f):
    m = np.around(memoryless_strategy(tau, n_f))
    en = 1/(1-np.exp(-1./tau)) + const
    return en/((m+n_f)/(1-np.exp(-m/tau)) + const)
