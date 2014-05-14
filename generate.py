#!/usr/bin/env python
import numpy as np, pylab as pl
from read_cdf import load_cdf_csv, get_cdf_trace

import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times'
matplotlib.rcParams['font.size'] = 16.0
#matplotlib.rcParams['mathtext.fontset'] = 'cm'

np.random.seed(1337)

def sample_rayleigh(fd=100., N=4096):
    N_power_of_2 = int(2**np.ceil(np.log2(N)))
    fd_power_of_2 = float(fd) / N * N_power_of_2
    e = np.seterr(invalid='ignore', divide='ignore')
    nu = np.arange(N_power_of_2+1)-N_power_of_2/2
    S = np.diff(np.where(np.abs(nu)<fd_power_of_2, np.arctan(nu/(fd_power_of_2**2-nu**2)**.5) / np.pi, np.sign(nu)*.5))
    X = np.fft.fftshift(S**.5*.5**.5*(np.random.standard_normal(N_power_of_2) + 1j * np.random.standard_normal(N_power_of_2))) * N_power_of_2
    x = np.fft.ifft(X)
    np.seterr(**e)
    return x[:N]

use_dumb_straw_men = False
use_super_straw_men = False
use_limited_straw_men = True

max_delay = 100e-3
bandwidth_high = 1500e3
bandwidth_low = 30e3

velocity = 10. # m/s
carrier = 2.4e9 # Hz
c = 299792458.
Fs = 1e6
duration = 100e-3
n_f = 10.

#------------------------------

nu_doppler = carrier * velocity / c

N = int(duration * Fs)
r = sample_rayleigh(N * nu_doppler / Fs, N)

# length = 256 bits
load_cdf_csv('spinal.csv', spinal=True, strider=False, raptor=False)

r_snr = np.abs(r)**2 * 10.**(.1 * 15.)
r_snr = np.clip(r_snr, 10.**(.1 * -5.), 10.**(.1 * 35.))
r_lsnr = 10*np.log10(r_snr)
t = np.arange(r_snr.size) / float(r.size) * duration

dummy_traces = False

params = [(True ,.99,'r' ,dict(label='Ideal adaptation', linewidth=2)),
          (False,.80,'b' ,dict(label='Learning with $\\alpha$=.80')),
          (False,.90,'g' ,dict(label='Learning with $\\alpha$=.90')),
          (False,.95,None,None),
          (False,.99,'c' ,dict(label='Learning with $\\alpha$=.99')),
          ]

print 'Learning Overhead:'
pl.figure(figsize=(16,4))
pl.subplots_adjust(left=.05, right=.95, hspace=0., bottom=.12)
pl.plot(1000*t, 12 * np.log2(1+r_snr), 'k--', label='Channel capacity')

for i, (known_cdfs, alpha, color, plot_params) in enumerate(params):
    if not dummy_traces:
        y, x = get_cdf_trace(12e6, t, r_lsnr, n_f, 256, known_cdfs, alpha)
    else:
        y, x = np.r_[0,12e6 * np.log2(1+r_snr)], np.r_[t,0]
    throughput = np.sum(y * np.r_[0, np.diff(x)])
    if known_cdfs:
        throughput_ideal = throughput
    else:
        print 'With alpha=%.2f, %5.2f%%' % (alpha, 100*(1-throughput/throughput_ideal))
    if color is not None:
        pl.plot(1000*x[:-1], 1e-6*y[1:], color, **plot_params)
    pl.draw()

pl.ylabel('Throughput (Mbps)')
pl.xlabel('Time (ms)')
pl.ylim(0,duration*1e3)
pl.legend(loc='upper center', ncol=5, prop={'size':16.})
pl.savefig('result4.pdf')
pl.xlim(71, 83)
pl.savefig('result4b.pdf')
