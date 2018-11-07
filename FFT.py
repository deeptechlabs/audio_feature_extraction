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
from numpy.lib.stride_tricks import as_strided
import scipy.signal as sg
from scipy.interpolate import interp1d
import wave
from scipy.cluster.vq import vq
from scipy import linalg, fftpack
from numpy.testing import assert_almost_equal
from scipy.linalg import svd
from scipy.io import wavfile
from scipy.signal import firwin
import zipfile
import tarfile
import os
import copy
import multiprocessing
from multiprocessing import Pool
import functools
import time


def stft(X, fftsize=128, step="half", mean_normalize=True, real=False,
         compute_onesided=True):
    """
    Compute STFT for 1D real valued input X
    """
    if real:
        local_fft = fftpack.rfft
        cut = -1
    else:
        local_fft = fftpack.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2 + 1
    if mean_normalize:
        X -= X.mean()
    if step == "half":
        X = halfoverlap(X, fftsize)
    else:
        X = overlap(X, fftsize, step)
    size = fftsize
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    return X


def istft(X, fftsize=128, step="half", wsola=False, mean_normalize=True,
          real=False, compute_onesided=True):
    """
    Compute ISTFT for STFT transformed X
    """
    if real:
        local_ifft = fftpack.irfft
        X_pad = np.zeros((X.shape[0], X.shape[1] + 1)) + 0j
        X_pad[:, :-1] = X
        X = X_pad
    else:
        local_ifft = fftpack.ifft
    if compute_onesided:
        X_pad = np.zeros((X.shape[0], 2 * X.shape[1])) + 0j
        X_pad[:, :fftsize // 2 + 1] = X
        X_pad[:, fftsize // 2 + 1:] = 0
        X = X_pad
    X = local_ifft(X).astype("float64")
    if step == "half":
        X = invert_halfoverlap(X)
    else:
        X = overlap_add(X, step, wsola=wsola)
    if mean_normalize:
        X -= np.mean(X)
    return X


"""
Filter data with an FIR filter using the overlap-add method.
from http://projects.scipy.org/scipy/attachment/ticket/837/fftfilt.py
"""
def nextpow2(x):
    """Return the first integer N such that 2**N >= abs(x)"""
    return np.ceil(np.log2(np.abs(x)))


def fftfilt(b, x, *n):
    """Filter the signal x with the FIR filter described by the
    coefficients in b using the overlap-add method. If the FFT
    length n is not specified, it and the overlap-add block length
    are selected so as to minimize the computational cost of
    the filtering operation."""

    N_x = len(x)
    N_b = len(b)

    # Determine the FFT length to use:
    if len(n):
        # Use the specified FFT length (rounded up to the nearest
        # power of 2), provided that it is no less than the filter
        # length:
        n = n[0]
        if n != int(n) or n <= 0:
            raise ValueError('n must be a nonnegative integer')
        if n < N_b:
            n = N_b
        N_fft = 2**nextpow2(n)
    else:
        if N_x > N_b:
            # When the filter length is smaller than the signal,
            # choose the FFT length and block size that minimize the
            # FLOPS cost. Since the cost for a length-N FFT is
            # (N/2)*log2(N) and the filtering operation of each block
            # involves 2 FFT operations and N multiplications, the
            # cost of the overlap-add method for 1 length-N block is
            # N*(1+log2(N)). For the sake of efficiency, only FFT
            # lengths that are powers of 2 are considered:
            N = 2**np.arange(np.ceil(np.log2(N_b)),
                             np.floor(np.log2(N_x)))
            cost = np.ceil(N_x/(N-N_b+1))*N*(np.log2(N)+1)
            N_fft = N[np.argmin(cost)]
        else:
            # When the filter length is at least as long as the signal,
            # filter the signal using a single block:
            N_fft = 2**nextpow2(N_b+N_x-1)

    N_fft = int(N_fft)

    # Compute the block length:
    L = int(N_fft - N_b + 1)

    # Compute the transform of the filter:
    H = np.fft.fft(b, N_fft)

    y = np.zeros(N_x, dtype=np.float32)
    i = 0
    while i <= N_x:
        il = min([i+L,N_x])
        k = min([i+N_fft,N_x])
        yt = np.fft.ifft(np.fft.fft(x[i:il],N_fft)*H,N_fft) # Overlap..
        y[i:k] = y[i:k] + yt[:k-i]            # and add
        i += L
    return y
