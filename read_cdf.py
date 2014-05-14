import numpy as np, scipy as sp, scipy.stats, scipy.interpolate
from rlp_fast import strategizer, strategy_time

def load_cdf_csv(fn='spinal.csv', spinal=True, strider=False, raptor=False):
    global lsnr, x_cdf, cdf, ccdf
    a = open(fn, 'r').readlines()
    a = [aa.strip().split(',') for aa in a]
    names = [aa[0] for aa in a[1:]]
    lsnr = np.array([float(n.split('=')[1].split('dB')[0]) for n in names])
    values = [[float(aaa) if aaa else None for aaa in aa[1:]] for aa in a]
    for v in values:
        if v[0] == None:
            v[0] = 0.
        for i, vv in enumerate(v):
            if vv == None:
                v[i] = v[i-1]

    values = np.array(values)
    x_cdf = values[0].astype(int)
    cdf = values[1:]
    if spinal:
        x_cdf = x_cdf[:-1]
        cdf = cdf[:,:-1]
        reps = int((np.amax(x_cdf)+1)/65)
        x_desired = np.repeat(np.arange(reps) * 65, 8) + np.tile(np.array([0, 8, 17, 25, 33, 41, 49, 57]), reps)
        cdf = np.array([interpolate_cdf(x_cdf, c, x_desired) for c in cdf])
        x_cdf = x_desired

    idx = np.where(lsnr <= 35)
    lsnr = lsnr[idx]
    cdf = cdf[idx[0]]
    snr = 10. ** (.1 * lsnr)

    if strider or raptor:
        x_cdf *= 2

    ccdf = 1-cdf

def interpolate_cdf(x, cdf, x_out):
    i = scipy.interpolate.interp1d(x, cdf)
    l = np.amin(x)
    r = np.amax(x)
    out = np.zeros(x_out.size, cdf.dtype)
    good_left = (l < x_out)
    good_right = (x_out < r)
    good = good_left * good_right
    out[np.where(1-good_left)] = 0.
    out[np.where(1-good_right)] = 1.
    out[np.where(good)] = i(x_out[np.where(good)])
    return out

def get_min_time(n_f):
    pdf = np.diff(np.hstack((np.zeros(cdf.shape[0])[:,np.newaxis],cdf)), axis=1)
    p_success_free = pdf.sum(1)
    pdf /= p_success_free[:, np.newaxis]
    x_pdf = x_cdf
    time = (x_pdf * pdf).sum(1) / p_success_free
    time = time[:,np.newaxis] + n_f / p_success_free[:,np.newaxis]
    return time

def get_known_cdf_time(n_f):
    time = np.zeros((lsnr.size, n_f.size), float)
    for j, n in enumerate(n_f):
        for i, c in enumerate(ccdf):
            st = strategizer(c, x_cdf).summary(n)
            t, p = strategy_time(st, c, x_cdf, n, False)
            time[i, j] = t/p
    return time

def get_try_after_n_time(n_f, after_n):
    time = np.zeros((lsnr.size, n_f.size), float)
    for j, n in enumerate(n_f):
        for i, c in enumerate(ccdf):
            st = np.arange(0, x_cdf.size, after_n)
            t, p = strategy_time(st, c, x_cdf, n, False)
            time[i, j] = t/p
    return time

def get_arq_time(n_f, after_n):
    time = np.zeros((lsnr.size, n_f.size), float)
    for j, n in enumerate(n_f):
        for i, c in enumerate(ccdf):
            st = [after_n]
            t, p = strategy_time(st, c, x_cdf, n, False)
            time[i, j] = t/p
    return time

class multinomial_ccdf_learner:
    def __init__(self, x_cdf, alpha=.99):
        self.x_cdf = np.r_[x_cdf, inf]
        self.alpha = alpha
        self.K = x_cdf.size
        self.alpha_dirichlet = np.ones(K, float)
    def learn(self, sample):
        i = np.where(self.x_cdf >= sample)[0][0]
        self.alpha_dirichlet = 1 + (self.alpha_dirichlet-1)*self.alpha
        self.alpha_dirichlet[i] += 1
    def getstate(self):
        pdf = (self.alpha_dirichlet - 1) / (np.sum(self.alpha_dirichlet) - self.K + 1e-6)
        pdf /= np.sum(pdf)
        return np.cumsum(pdf)[:-1]

class gaussian_ccdf_learner:
    def __init__(self, x_cdf, alpha=.99):
        self.x_cdf = x_cdf
        self.alpha = alpha
        self.samples = 1.
        self.sum = 100.
        self.sumsq = 110.**2
    def learn(self, sample):
        self.samples *= self.alpha
        self.samples += 1
        self.sum *= self.alpha
        self.sum += sample
        self.sumsq *= self.alpha
        self.sumsq += sample**2
    def getstate(self):
        mu = self.sum / self.samples
        sigmasq = self.sumsq / self.samples - (self.sum / self.samples)**2
        return scipy.stats.norm(mu, sigmasq**.5).sf(self.x_cdf)

def replace_line(s):
    import sys
    sys.stdout.write('\r\x1b[K'+s)
    sys.stdout.flush()

def get_cdf_trace(Fs, t, lsnr_trace, n_f, length, known_cdfs, alpha):
    out_throughput = np.zeros(10000)
    out_time = np.zeros(10000)
    if not known_cdfs:
        learner = gaussian_ccdf_learner(x_cdf, alpha)
    time = 0.
    i = 0
    while time < t[-1]:
        # get the current ccdf
        current_lsnr = lsnr_trace[np.nonzero(t>=time)[0][0]]
        idx = np.clip(current_lsnr-np.amin(lsnr), 0, lsnr.size-1)
        c = ccdf[int(idx)]
        if int(idx)+1 < lsnr.size:
            a = idx - int(idx)
            c = c + a * (ccdf[int(idx)+1] - c)
        # pick a strategy
        if known_cdfs:
            st = strategizer(c, x_cdf).summary(n_f)
        else:
            st = strategizer(learner.getstate(), x_cdf).summary(n_f)
        # follow the strategy for one transmission
        if not known_cdfs:
            r = np.random.random()
            n_success = x_cdf[np.where(r > c)[0]]
            n_success = n_success[0] if n_success.size else x_cdf[-1]
            learner.learn(n_success)
        time_, p_success_ = strategy_time(st, c, x_cdf, n_f, False)
        # time is in half symbols
        time += time_/Fs/2
        throughput = Fs*length*p_success_/(time_/2)
        i += 1
        if i % 100 == 0:
            replace_line('known_cdfs=%s alpha=%.2f t=%5.1f ms  SNR=%5.2f dB  throughput=%4.1f Mbps  capacity=%4.1f Mbps' % (known_cdfs, alpha, 1000*time, current_lsnr, throughput*1e-6, 12*np.log2(1+10.**(.1*current_lsnr))))
        if i >= out_throughput.size:
            out_throughput = np.r_[out_throughput, np.zeros(10000)]
            out_time = np.r_[out_time, np.zeros(10000)]
        out_throughput[i] = throughput
        out_time[i] = time
    replace_line('')
    return out_throughput[:i], out_time[:i]

