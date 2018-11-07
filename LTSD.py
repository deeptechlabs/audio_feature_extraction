# License: BSD 3-clause
# Authors: Kyle Kastner
# LTSD routine from jfsantos (Joao Felipe Santos)
# Harvest, Cheaptrick, D4C, WORLD routines based on MATLAB code from M. Morise
# http://ml.cs.yamanashi.ac.jp/world/english/
# MGC code based on r9y9 (Ryuichi Yamamoto) MelGeneralizedCepstrums.jl
# Pieces also adapted from SPTK
from __future__ import division
import numpy as np
import scipy as sp
from scipy.cluster.vq import vq
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import as_strided
from scipy import linalg, fftpack
from numpy.testing import assert_almost_equal
from scipy.linalg import svd
from scipy.io import wavfile
from scipy.signal import firwin
from multiprocessing import Pool
from PIL import Image


class LTSD():
    """
    LTSD VAD code from jfsantos
    """
    def __init__(self,winsize,window,order):
        self.winsize = int(winsize)
        self.window = window
        self.order = order
        self.amplitude = {}

    def get_amplitude(self,signal,l):
        if self.amplitude.has_key(l):
            return self.amplitude[l]
        else:
            amp = sp.absolute(sp.fft(get_frame(signal, self.winsize,l) * self.window))
            self.amplitude[l] = amp
            return amp

    def compute_noise_avg_spectrum(self, nsignal):
        windownum = int(len(nsignal)//(self.winsize//2) - 1)
        avgamp = np.zeros(self.winsize)
        for l in range(windownum):
            avgamp += sp.absolute(sp.fft(get_frame(nsignal, self.winsize,l) * self.window))
        return avgamp/float(windownum)

    def compute(self,signal):
        self.windownum = int(len(signal)//(self.winsize//2) - 1)
        ltsds = np.zeros(self.windownum)
        #Calculate the average noise spectrum amplitude based 20 frames in the head parts of input signal.
        self.avgnoise = self.compute_noise_avg_spectrum(signal[0:self.winsize*20])**2
        for l in range(self.windownum):
            ltsds[l] = self.ltsd(signal,l,5)
        return ltsds

    def ltse(self,signal,l,order):
        maxamp = np.zeros(self.winsize)
        for idx in range(l-order,l+order+1):
            amp = self.get_amplitude(signal,idx)
            maxamp = np.maximum(maxamp,amp)
        return maxamp

    def ltsd(self,signal,l,order):
        if l < order or l+order >= self.windownum:
            return 0
        return 10.0 * np.log10(np.sum(self.ltse(signal,l,order)**2/self.avgnoise)/float(len(self.avgnoise)))


def ltsd_vad(x, fs, threshold=9, winsize=8192):
    # winsize based on sample rate
    # 1024 for fs = 16000
    orig_dtype = x.dtype
    orig_scale_min = x.min()
    orig_scale_max = x.max()
    x = (x - x.min()) / (x.max() - x.min())
    # works with 16 bit
    x = x * (2 ** 15)
    x = x.astype("int32")
    window = sp.hanning(winsize)
    ltsd = LTSD(winsize, window, 5)
    s_vad = ltsd.compute(x)
    # LTSD is 50% overlap, so each "step" covers 4096 samples
    # +1 to cover the extra edge window
    n_samples = int(((len(s_vad) + 1) * winsize) // 2)
    time_s = n_samples / float(fs)
    time_points = np.linspace(0, time_s, len(s_vad))
    time_samples = (fs * time_points).astype(np.int32)
    time_samples = time_samples
    f_vad = np.zeros_like(x, dtype=np.bool)
    offset = winsize
    for n, (ss, es) in enumerate(zip(time_samples[:-1], time_samples[1:])):
        sss = ss - offset
        if sss < 0:
            sss = 0
        ses = es - offset
        if ses < 0:
            ses = 0
        if s_vad[n + 1] < threshold:
            f_vad[sss:ses] = False
        else:
            f_vad[sss:ses] = True
    f_vad[ses:] = False
    x = x.astype("float64")
    x = x / float(2 ** 15)
    x = x * (orig_scale_max - orig_scale_min) + orig_scale_min
    x = x.astype(orig_dtype)
    return x[f_vad], f_vad


def run_ltsd_example():
    fs, d = fetch_sample_speech_tapestry()
    winsize = 1024
    d = d.astype("float32") / 2 ** 15
    d -= d.mean()

    pad = 3 * fs
    noise_pwr = np.percentile(d, 1) ** 2
    noise_pwr = max(1E-9, noise_pwr)
    d = np.concatenate((np.zeros((pad,)) + noise_pwr * np.random.randn(pad), d))
    _, vad_segments = ltsd_vad(d, fs, winsize=winsize)
    v_up = np.where(vad_segments == True)[0]
    s = v_up[0]
    st = v_up[-1] + int(.5 * fs)
    d = d[s:st]

    bname = "tapestry.wav".split(".")[0]
    wavfile.write("%s_out.wav" % bname, fs, soundsc(d))
