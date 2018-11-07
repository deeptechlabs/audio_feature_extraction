# License: BSD 3-clause
# Authors: Kyle Kastner
# LTSD routine from jfsantos (Joao Felipe Santos)
# Harvest, Cheaptrick, D4C, WORLD routines based on MATLAB code from M. Morise
# http://ml.cs.yamanashi.ac.jp/world/english/
# MGC code based on r9y9 (Ryuichi Yamamoto) MelGeneralizedCepstrums.jl
# Pieces also adapted from SPTK
from __future__ import division
import numpy as np
import wave
import zipfile
import tarfile
import os

from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import interp1d
from scipy.cluster.vq import vq
from scipy import linalg, fftpack
from numpy.testing import assert_almost_equal
from scipy.linalg import svd
from scipy.io import wavfile
from scipy.signal import firwin
from multiprocessing import Pool


try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib


def download(url, server_fname, local_fname=None, progress_update_percentage=5,
             bypass_certificate_check=False):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    if bypass_certificate_check:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        u = urllib.urlopen(url, context=ctx)
    else:
        u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                print(status)
                p += progress_update_percentage


def fetch_sample_speech_tapestry():
    url = "https://www.dropbox.com/s/qte66a7haqspq2g/tapestry.wav?dl=1"
    wav_path = "tapestry.wav"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    # file is stereo? - just choose one channel
    return fs, d


def fetch_sample_file(wav_path):
    if not os.path.exists(wav_path):
        raise ValueError("Unable to find file at path %s" % wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    # file is stereo - just choose one channel
    if len(d.shape) > 1:
        d = d[:, 0]
    return fs, d


def fetch_sample_music():
    url = "http://www.music.helsinki.fi/tmt/opetus/uusmedia/esim/"
    url += "a2002011001-e02-16kHz.wav"
    wav_path = "test.wav"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    # file is stereo - just choose one channel
    d = d[:, 0]
    return fs, d


def fetch_sample_speech_fruit(n_samples=None):
    url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
    wav_path = "audio.tar.gz"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    tf = tarfile.open(wav_path)
    wav_names = [fname for fname in tf.getnames()
                 if ".wav" in fname.split(os.sep)[-1]]
    speech = []
    print("Loading speech files...")
    for wav_name in wav_names[:n_samples]:
        f = tf.extractfile(wav_name)
        fs, d = wavfile.read(f)
        d = d.astype('float32') / (2 ** 15)
        speech.append(d)
    return fs, speech


def fetch_sample_speech_eustace(n_samples=None):
    """
    http://www.cstr.ed.ac.uk/projects/eustace/download.html
    """
    # data
    url = "http://www.cstr.ed.ac.uk/projects/eustace/down/eustace_wav.zip"
    wav_path = "eustace_wav.zip"
    if not os.path.exists(wav_path):
        download(url, wav_path)

    # labels
    url = "http://www.cstr.ed.ac.uk/projects/eustace/down/eustace_labels.zip"
    labels_path = "eustace_labels.zip"
    if not os.path.exists(labels_path):
        download(url, labels_path)

    # Read wavfiles
    # 16 kHz wav
    zf = zipfile.ZipFile(wav_path, 'r')
    wav_names = [fname for fname in zf.namelist()
                 if ".wav" in fname.split(os.sep)[-1]]
    fs = 16000
    speech = []
    print("Loading speech files...")
    for wav_name in wav_names[:n_samples]:
        wav_str = zf.read(wav_name)
        d = np.frombuffer(wav_str, dtype=np.int16)
        d = d.astype('float32') / (2 ** 15)
        speech.append(d)

    zf = zipfile.ZipFile(labels_path, 'r')
    label_names = [fname for fname in zf.namelist()
                   if ".lab" in fname.split(os.sep)[-1]]
    labels = []
    print("Loading label files...")
    for label_name in label_names[:n_samples]:
        label_file_str = zf.read(label_name)
        labels.append(label_file_str)
    return fs, speech

def voiced_unvoiced(X, window_size=256, window_step=128, copy=True):
    """
    Voiced unvoiced detection from a raw signal
    Based on code from:
        https://www.clear.rice.edu/elec532/PROJECTS96/lpc/code.html
    Other references:
        http://www.seas.ucla.edu/spapl/code/harmfreq_MOLRT_VAD.m
    Parameters
    ----------
    X : ndarray
        Raw input signal
    window_size : int, optional (default=256)
        The window size to use, in samples.
    window_step : int, optional (default=128)
        How far the window steps after each calculation, in samples.
    copy : bool, optional (default=True)
        Whether to make a copy of the input array or allow in place changes.
    """
    X = np.array(X, copy=copy)
    if len(X.shape) < 2:
        X = X[None]
    n_points = X.shape[1]
    n_windows = n_points // window_step
    # Padding
    pad_sizes = [(window_size - window_step) // 2,
                 window_size - window_step // 2]
    # TODO: Handling for odd window sizes / steps
    X = np.hstack((np.zeros((X.shape[0], pad_sizes[0])), X,
                   np.zeros((X.shape[0], pad_sizes[1]))))

    clipping_factor = 0.6
    b, a = sg.butter(10, np.pi * 9 / 40)
    voiced_unvoiced = np.zeros((n_windows, 1))
    period = np.zeros((n_windows, 1))
    for window in range(max(n_windows - 1, 1)):
        XX = X.ravel()[window * window_step + np.arange(window_size)]
        XX *= sg.windows.hann(len(XX))
        XX = sg.lfilter(b, a, XX)
        left_max = np.max(np.abs(XX[:len(XX) // 3]))
        right_max = np.max(np.abs(XX[-len(XX) // 3:]))
        clip_value = clipping_factor * np.min([left_max, right_max])
        XX_clip = np.clip(XX, clip_value, -clip_value)
        XX_corr = np.correlate(XX_clip, XX_clip, mode='full')
        center = np.argmax(XX_corr)
        right_XX_corr = XX_corr[center:]
        prev_window = max([window - 1, 0])
        if voiced_unvoiced[prev_window] > 0:
            # Want it to be harder to turn off than turn on
            strength_factor = .29
        else:
            strength_factor = .3
        start = np.where(right_XX_corr < .3 * XX_corr[center])[0]
        # 20 is hardcoded but should depend on samplerate?
        try:
            start = np.max([20, start[0]])
        except IndexError:
            start = 20
        search_corr = right_XX_corr[start:]
        index = np.argmax(search_corr)
        second_max = search_corr[index]
        if (second_max > strength_factor * XX_corr[center]):
            voiced_unvoiced[window] = 1
            period[window] = start + index - 1
        else:
            voiced_unvoiced[window] = 0
            period[window] = 0
    return np.array(voiced_unvoiced), np.array(period)


def rolling_mean(X, window_size):
    w = 1.0 / window_size * np.ones((window_size))
    return np.correlate(X, w, 'valid')


def rolling_window(X, window_size):
    # for 1d data
    shape = X.shape[:-1] + (X.shape[-1] - window_size + 1, window_size)
    strides = X.strides + (X.strides[-1],)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


def soundsc(X, gain_scale=.9, copy=True):
    """
    Approximate implementation of soundsc from MATLAB without the audio playing.
    Parameters
    ----------
    X : ndarray
        Signal to be rescaled
    gain_scale : float
        Gain multipler, default .9 (90% of maximum representation)
    copy : bool, optional (default=True)
        Whether to make a copy of input signal or operate in place.
    Returns
    -------
    X_sc : ndarray
        (-32767, 32767) scaled version of X as int16, suitable for writing
        with scipy.io.wavfile
    """
    X = np.array(X, copy=copy)
    X = (X - X.min()) / (X.max() - X.min())
    X = 2 * X - 1
    X = gain_scale * X
    X = X * 2 ** 15
    return X.astype('int16')


def _wav2array(nchannels, sampwidth, data):
    # wavio.py
    # Author: Warren Weckesser
    # License: BSD 3-Clause (http://opensource.org/licenses/BSD-3-Clause)

    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.fromstring(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result


def readwav(file):
    # wavio.py
    # Author: Warren Weckesser
    # License: BSD 3-Clause (http://opensource.org/licenses/BSD-3-Clause)
    """
    Read a wav file.
    Returns the frame rate, sample width (in bytes) and a numpy array
    containing the data.
    This function does not read compressed wav files.
    """
    wav = wave.open(file)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)
    return rate, sampwidth, array


def csvd(arr):
    """
    Do the complex SVD of a 2D array, returning real valued U, S, VT
    http://stemblab.github.io/complex-svd/
    """
    C_r = arr.real
    C_i = arr.imag
    block_x = C_r.shape[0]
    block_y = C_r.shape[1]
    K = np.zeros((2 * block_x, 2 * block_y))
    # Upper left
    K[:block_x, :block_y] = C_r
    # Lower left
    K[:block_x, block_y:] = C_i
    # Upper right
    K[block_x:, :block_y] = -C_i
    # Lower right
    K[block_x:, block_y:] = C_r
    return svd(K, full_matrices=False)


def icsvd(U, S, VT):
    """
    Invert back to complex values from the output of csvd
    U, S, VT = csvd(X)
    X_rec = inv_csvd(U, S, VT)
    """
    K = U.dot(np.diag(S)).dot(VT)
    block_x = U.shape[0] // 2
    block_y = U.shape[1] // 2
    arr_rec = np.zeros((block_x, block_y)) + 0j
    arr_rec.real = K[:block_x, :block_y]
    arr_rec.imag = K[:block_x, block_y:]
    return arr_rec


def sinusoid_analysis(X, input_sample_rate, resample_block=128, copy=True):
    """
    Contruct a sinusoidal model for the input signal.
    Parameters
    ----------
    X : ndarray
        Input signal to model
    input_sample_rate : int
        The sample rate of the input signal
    resample_block : int, optional (default=128)
       Controls the step size of the sinusoidal model
    Returns
    -------
    frequencies_hz : ndarray
       Frequencies for the sinusoids, in Hz.
    magnitudes : ndarray
       Magnitudes of sinusoids returned in ``frequencies``
    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    X = np.array(X, copy=copy)
    resample_to = 8000
    if input_sample_rate != resample_to:
        if input_sample_rate % resample_to != 0:
            raise ValueError("Input sample rate must be a multiple of 8k!")
        # Should be able to use resample... ?
        # resampled_count = round(len(X) * resample_to / input_sample_rate)
        # X = sg.resample(X, resampled_count, window=sg.windows.hann(len(X)))
        X = sg.decimate(X, input_sample_rate // resample_to, zero_phase=True)
    step_size = 2 * round(resample_block / input_sample_rate * resample_to / 2.)
    a, g, e = lpc_analysis(X, order=8, window_step=step_size,
                           window_size=2 * step_size)
    f, m = lpc_to_frequency(a, g)
    f_hz = f * resample_to / (2 * np.pi)
    return f_hz, m


def slinterp(X, factor, copy=True):
    """
    Slow-ish linear interpolation of a 1D numpy array. There must be some
    better function to do this in numpy.
    Parameters
    ----------
    X : ndarray
        1D input array to interpolate
    factor : int
        Integer factor to interpolate by
    Return
    ------
    X_r : ndarray
    """
    sz = np.product(X.shape)
    X = np.array(X, copy=copy)
    X_s = np.hstack((X[1:], [0]))
    X_r = np.zeros((factor, sz))
    for i in range(factor):
        X_r[i, :] = (factor - i) / float(factor) * X + (i / float(factor)) * X_s
    return X_r.T.ravel()[:(sz - 1) * factor + 1]


def sinusoid_synthesis(frequencies_hz, magnitudes, input_sample_rate=16000,
                       resample_block=128):
    """
    Create a time series based on input frequencies and magnitudes.
    Parameters
    ----------
    frequencies_hz : ndarray
        Input signal to model
    magnitudes : int
        The sample rate of the input signal
    input_sample_rate : int, optional (default=16000)
        The sample rate parameter that the sinusoid analysis was run with
    resample_block : int, optional (default=128)
       Controls the step size of the sinusoidal model
    Returns
    -------
    synthesized : ndarray
        Sound vector synthesized from input arguments
    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    rows, cols = frequencies_hz.shape
    synthesized = np.zeros((1 + ((rows - 1) * resample_block),))
    for col in range(cols):
        mags = slinterp(magnitudes[:, col], resample_block)
        freqs = slinterp(frequencies_hz[:, col], resample_block)
        cycles = np.cumsum(2 * np.pi * freqs / float(input_sample_rate))
        sines = mags * np.cos(cycles)
        synthesized += sines
    return synthesized


def dct_compress(X, n_components, window_size=128):
    """
    Compress using the DCT
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        The input signal to compress. Should be 1-dimensional
    n_components : int
        The number of DCT components to keep. Setting n_components to about
        .5 * window_size can give compression with fairly good reconstruction.
    window_size : int
        The input X is broken into windows of window_size, each of which are
        then compressed with the DCT.
    Returns
    -------
    X_compressed : ndarray, shape=(num_windows, window_size)
       A 2D array of non-overlapping DCT coefficients. For use with uncompress
    Reference
    ---------
    http://nbviewer.ipython.org/github/craffel/crucialpython/blob/master/week3/stride_tricks.ipynb
    """
    if len(X) % window_size != 0:
        append = np.zeros((window_size - len(X) % window_size))
        X = np.hstack((X, append))
    num_frames = len(X) // window_size
    X_strided = X.reshape((num_frames, window_size))
    X_dct = fftpack.dct(X_strided, norm='ortho')
    if n_components is not None:
        X_dct = X_dct[:, :n_components]
    return X_dct


def dct_uncompress(X_compressed, window_size=128):
    """
    Uncompress a DCT compressed signal (such as returned by ``compress``).
    Parameters
    ----------
    X_compressed : ndarray, shape=(n_samples, n_features)
        Windowed and compressed array.
    window_size : int, optional (default=128)
        Size of the window used when ``compress`` was called.
    Returns
    -------
    X_reconstructed : ndarray, shape=(n_samples)
        Reconstructed version of X.
    """
    if X_compressed.shape[1] % window_size != 0:
        append = np.zeros((X_compressed.shape[0],
                           window_size - X_compressed.shape[1] % window_size))
        X_compressed = np.hstack((X_compressed, append))
    X_r = fftpack.idct(X_compressed, norm='ortho')
    return X_r.ravel()


def sine_window(X):
    """
    Apply a sinusoid window to X.
    Parameters
    ----------
    X : ndarray, shape=(n_samples, n_features)
        Input array of samples
    Returns
    -------
    X_windowed : ndarray, shape=(n_samples, n_features)
        Windowed version of X.
    """
    i = np.arange(X.shape[1])
    win = np.sin(np.pi * (i + 0.5) / X.shape[1])
    row_stride = 0
    col_stride = win.itemsize
    strided_win = as_strided(win, shape=X.shape,
                             strides=(row_stride, col_stride))
    return X * strided_win


def kaiserbessel_window(X, alpha=6.5):
    """
    Apply a Kaiser-Bessel window to X.
    Parameters
    ----------
    X : ndarray, shape=(n_samples, n_features)
        Input array of samples
    alpha : float, optional (default=6.5)
        Tuning parameter for Kaiser-Bessel function. alpha=6.5 should make
        perfect reconstruction possible for DCT.
    Returns
    -------
    X_windowed : ndarray, shape=(n_samples, n_features)
        Windowed version of X.
    """
    beta = np.pi * alpha
    win = sg.kaiser(X.shape[1], beta)
    row_stride = 0
    col_stride = win.itemsize
    strided_win = as_strided(win, shape=X.shape,
                             strides=(row_stride, col_stride))
    return X * strided_win


def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    overlap_sz = window_size - window_step
    new_shape = X.shape[:-1] + ((X.shape[-1] - overlap_sz) // window_step, window_size)
    new_strides = X.strides[:-1] + (window_step * X.strides[-1],) + X.strides[-1:]
    X_strided = as_strided(X, shape=new_shape, strides=new_strides)
    return X_strided


def halfoverlap(X, window_size):
    """
    Create an overlapped version of X using 50% of window_size as overlap.
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    window_step = window_size // 2
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    num_frames = len(X) // window_step - 1
    row_stride = X.itemsize * window_step
    col_stride = X.itemsize
    X_strided = as_strided(X, shape=(num_frames, window_size),
                           strides=(row_stride, col_stride))
    return X_strided


def invert_halfoverlap(X_strided):
    """
    Invert ``halfoverlap`` function to reconstruct X
    Parameters
    ----------
    X_strided : ndarray, shape=(n_windows, window_size)
        X as overlapped windows
    Returns
    -------
    X : ndarray, shape=(n_samples,)
        Reconstructed version of X
    """
    # Hardcoded 50% overlap! Can generalize later...
    n_rows, n_cols = X_strided.shape
    X = np.zeros((((int(n_rows // 2) + 1) * n_cols),)).astype(X_strided.dtype)
    start_index = 0
    end_index = n_cols
    window_step = n_cols // 2
    for row in range(X_strided.shape[0]):
        X[start_index:end_index] += X_strided[row]
        start_index += window_step
        end_index += window_step
    return X


def overlap_add(X_strided, window_step, wsola=False):
    """
    overlap add to reconstruct X
    Parameters
    ----------
    X_strided : ndarray, shape=(n_windows, window_size)
        X as overlapped windows
    window_step : int
       step size for overlap add
    Returns
    -------
    X : ndarray, shape=(n_samples,)
        Reconstructed version of X
    """
    n_rows, window_size = X_strided.shape

    # Start with largest size (no overlap) then truncate after we finish
    # +2 for one window on each side
    X = np.zeros(((n_rows + 2) * window_size,)).astype(X_strided.dtype)
    start_index = 0

    total_windowing_sum = np.zeros((X.shape[0]))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(window_size) / (
        window_size - 1))
    for i in range(n_rows):
        end_index = start_index + window_size
        if wsola:
            offset_size = window_size - window_step
            offset = xcorr_offset(X[start_index:start_index + offset_size],
                                  X_strided[i, :offset_size])
            ss = start_index - offset
            st = end_index - offset
            if start_index - offset < 0:
                ss = 0
                st = 0 + (end_index - start_index)
            X[ss:st] += X_strided[i]
            total_windowing_sum[ss:st] += win
            start_index = start_index + window_step
        else:
            X[start_index:end_index] += X_strided[i]
            total_windowing_sum[start_index:end_index] += win
            start_index += window_step
    # Not using this right now
    #X = np.real(X) / (total_windowing_sum + 1)
    X = X[:end_index]
    return X


def overlap_dct_compress(X, n_components, window_size):
    """
    Overlap (at 50% of window_size) and compress X.
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to compress
    n_components : int
        number of DCT components to keep
    window_size : int
        Size of windows to take
    Returns
    -------
    X_dct : ndarray, shape=(n_windows, n_components)
        Windowed and compressed version of X
    """
    X_strided = halfoverlap(X, window_size)
    X_dct = fftpack.dct(X_strided, norm='ortho')
    if n_components is not None:
        X_dct = X_dct[:, :n_components]
    return X_dct


# Evil voice is caused by adding double the zeros before inverse DCT...
# Very cool bug but makes sense
def overlap_dct_uncompress(X_compressed, window_size):
    """
    Uncompress X as returned from ``overlap_compress``.
    Parameters
    ----------
    X_compressed : ndarray, shape=(n_windows, n_components)
        Windowed and compressed version of X
    window_size : int
        Size of windows originally used when compressing X
    Returns
    -------
    X_reconstructed : ndarray, shape=(n_samples,)
        Reconstructed version of X
    """
    if X_compressed.shape[1] % window_size != 0:
        append = np.zeros((X_compressed.shape[0], window_size -
                           X_compressed.shape[1] % window_size))
        X_compressed = np.hstack((X_compressed, append))
    X_r = fftpack.idct(X_compressed, norm='ortho')
    return invert_halfoverlap(X_r)


def herz_to_mel(freqs):
    """
    Based on code by Dan Ellis
    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    f_0 = 0  # 133.33333
    f_sp = 200 / 3.  # 66.66667
    bark_freq = 1000.
    bark_pt = (bark_freq - f_0) / f_sp
    # The magic 1.0711703 which is the ratio needed to get from 1000 Hz
    # to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz
    # and the preceding linear filter center at 933.33333 Hz
    # (actually 1000/933.33333 = 1.07142857142857 and
    # exp(log(6.4)/27) = 1.07117028749447)
    if not isinstance(freqs, np.ndarray):
        freqs = np.array(freqs)[None]
    log_step = np.exp(np.log(6.4) / 27)
    lin_pts = (freqs < bark_freq)
    mel = 0. * freqs
    mel[lin_pts] = (freqs[lin_pts] - f_0) / f_sp
    mel[~lin_pts] = bark_pt + np.log(freqs[~lin_pts] / bark_freq) / np.log(
        log_step)
    return mel


def mel_to_herz(mel):
    """
    Based on code by Dan Ellis
    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    f_0 = 0  # 133.33333
    f_sp = 200 / 3.  # 66.66667
    bark_freq = 1000.
    bark_pt = (bark_freq - f_0) / f_sp
    # The magic 1.0711703 which is the ratio needed to get from 1000 Hz
    # to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz
    # and the preceding linear filter center at 933.33333 Hz
    # (actually 1000/933.33333 = 1.07142857142857 and
    # exp(log(6.4)/27) = 1.07117028749447)
    if not isinstance(mel, np.ndarray):
        mel = np.array(mel)[None]
    log_step = np.exp(np.log(6.4) / 27)
    lin_pts = (mel < bark_pt)

    freqs = 0. * mel
    freqs[lin_pts] = f_0 + f_sp * mel[lin_pts]
    freqs[~lin_pts] = bark_freq * np.exp(np.log(log_step) * (
        mel[~lin_pts] - bark_pt))
    return freqs


def mel_freq_weights(n_fft, fs, n_filts=None, width=None):
    """
    Based on code by Dan Ellis
    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    min_freq = 0
    max_freq = fs // 2
    if width is None:
        width = 1.
    if n_filts is None:
        n_filts = int(herz_to_mel(max_freq) / 2) + 1
    else:
        n_filts = int(n_filts)
        assert n_filts > 0
    weights = np.zeros((n_filts, n_fft))
    fft_freqs = np.arange(n_fft // 2) / n_fft * fs
    min_mel = herz_to_mel(min_freq)
    max_mel = herz_to_mel(max_freq)
    partial = np.arange(n_filts + 2) / (n_filts + 1.) * (max_mel - min_mel)
    bin_freqs = mel_to_herz(min_mel + partial)
    bin_bin = np.round(bin_freqs / fs * (n_fft - 1))
    for i in range(n_filts):
        fs_i = bin_freqs[i + np.arange(3)]
        fs_i = fs_i[1] + width * (fs_i - fs_i[1])
        lo_slope = (fft_freqs - fs_i[0]) / float(fs_i[1] - fs_i[0])
        hi_slope = (fs_i[2] - fft_freqs) / float(fs_i[2] - fs_i[1])
        weights[i, :n_fft // 2] = np.maximum(
            0, np.minimum(lo_slope, hi_slope))
    # Constant amplitude multiplier
    weights = np.diag(2. / (bin_freqs[2:n_filts + 2]
                      - bin_freqs[:n_filts])).dot(weights)
    weights[:, n_fft // 2:] = 0
    return weights


def time_attack_agc(X, fs, t_scale=0.5, f_scale=1.):
    """
    AGC based on code by Dan Ellis
    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    # 32 ms grid for FFT
    n_fft = 2 ** int(np.log(0.032 * fs) / np.log(2))
    f_scale = float(f_scale)
    window_size = n_fft
    window_step = window_size // 2
    X_freq = stft(X, window_size, mean_normalize=False)
    fft_fs = fs / window_step
    n_bands = max(10, 20 / f_scale)
    mel_width = f_scale * n_bands / 10.
    f_to_a = mel_freq_weights(n_fft, fs, n_bands, mel_width)
    f_to_a = f_to_a[:, :n_fft // 2 + 1]
    audiogram = np.abs(X_freq).dot(f_to_a.T)
    fbg = np.zeros_like(audiogram)
    state = np.zeros((audiogram.shape[1],))
    alpha = np.exp(-(1. / fft_fs) / t_scale)
    for i in range(len(audiogram)):
        state = np.maximum(alpha * state, audiogram[i])
        fbg[i] = state

    sf_to_a = np.sum(f_to_a, axis=0)
    E = np.diag(1. / (sf_to_a + (sf_to_a == 0)))
    E = E.dot(f_to_a.T)
    E = fbg.dot(E.T)
    E[E <= 0] = np.min(E[E > 0])
    ts = istft(X_freq / E, window_size, mean_normalize=False)
    return ts, X_freq, E


def hebbian_kmeans(X, n_clusters=10, n_epochs=10, W=None, learning_rate=0.01,
                   batch_size=100, random_state=None, verbose=True):
    """
    Modified from existing code from R. Memisevic
    See http://www.cs.toronto.edu/~rfm/code/hebbian_kmeans.py
    """
    if W is None:
        if random_state is None:
            random_state = np.random.RandomState()
        W = 0.1 * random_state.randn(n_clusters, X.shape[1])
    else:
        assert n_clusters == W.shape[0]
    X2 = (X ** 2).sum(axis=1, keepdims=True)
    last_print = 0
    for e in range(n_epochs):
        for i in range(0, X.shape[0], batch_size):
            X_i = X[i: i + batch_size]
            X2_i = X2[i: i + batch_size]
            D = -2 * np.dot(W, X_i.T)
            D += (W ** 2).sum(axis=1, keepdims=True)
            D += X2_i.T
            S = (D == D.min(axis=0)[None, :]).astype("float").T
            W += learning_rate * (
                np.dot(S.T, X_i) - S.sum(axis=0)[:, None] * W)
        if verbose:
            if e == 0 or e > (.05 * n_epochs + last_print):
                last_print = e
                print("Epoch %i of %i, cost %.4f" % (
                    e + 1, n_epochs, D.min(axis=0).sum()))
    return W


def complex_to_real_view(arr_c):
    # Inplace view from complex to r, i as separate columns
    assert arr_c.dtype in [np.complex64, np.complex128]
    shp = arr_c.shape
    dtype = np.float64 if arr_c.dtype == np.complex128 else np.float32
    arr_r = arr_c.ravel().view(dtype=dtype).reshape(shp[0], 2 * shp[1])
    return arr_r


def real_to_complex_view(arr_r):
    # Inplace view from real, image as columns to complex
    assert arr_r.dtype not in [np.complex64, np.complex128]
    shp = arr_r.shape
    dtype = np.complex128 if arr_r.dtype == np.float64 else np.complex64
    arr_c = arr_r.ravel().view(dtype=dtype).reshape(shp[0], shp[1] // 2)
    return arr_c


def complex_to_abs(arr_c):
    return np.abs(arr_c)


def complex_to_angle(arr_c):
    return np.angle(arr_c)


def abs_and_angle_to_complex(arr_abs, arr_angle):
    # abs(f_c2 - f_c) < 1E-15
    return arr_abs * np.exp(1j * arr_angle)


def angle_to_sin_cos(arr_angle):
    return np.hstack((np.sin(arr_angle), np.cos(arr_angle)))


def sin_cos_to_angle(arr_sin, arr_cos):
    return np.arctan2(arr_sin, arr_cos)


def polyphase_core(x, m, f):
    # x = input data
    # m = decimation rate
    # f = filter
    # Hack job - append zeros to match decimation rate
    if x.shape[0] % m != 0:
        x = np.append(x, np.zeros((m - x.shape[0] % m,)))
    if f.shape[0] % m != 0:
        f = np.append(f, np.zeros((m - f.shape[0] % m,)))
    polyphase = p = np.zeros((m, (x.shape[0] + f.shape[0]) / m), dtype=x.dtype)
    p[0, :-1] = np.convolve(x[::m], f[::m])
    # Invert the x values when applying filters
    for i in range(1, m):
        p[i, 1:] = np.convolve(x[m - i::m], f[i::m])
    return p


def polyphase_single_filter(x, m, f):
    return np.sum(polyphase_core(x, m, f), axis=0)


def polyphase_lowpass(arr, downsample=2, n_taps=50, filter_pad=1.1):
    filt = firwin(downsample * n_taps, 1 / (downsample * filter_pad))
    filtered = polyphase_single_filter(arr, downsample, filt)
    return filtered


def window(arr, window_size, window_step=1, axis=0):
    """
    Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>
    <http://stackoverflow.com/questions/4936620/using-strides-for-an-efficient-moving-average-filter>
    """
    if window_size < 1:
        raise ValueError("`window_size` must be at least 1.")
    if window_size > arr.shape[-1]:
        raise ValueError("`window_size` is too long.")

    orig = list(range(len(arr.shape)))
    trans = list(range(len(arr.shape)))
    trans[axis] = orig[-1]
    trans[-1] = orig[axis]
    arr = arr.transpose(trans)

    shape = arr.shape[:-1] + (arr.shape[-1] - window_size + 1, window_size)
    strides = arr.strides + (arr.strides[-1],)
    strided = as_strided(arr, shape=shape, strides=strides)

    if window_step > 1:
        strided = strided[..., ::window_step, :]

    orig = list(range(len(strided.shape)))
    trans = list(range(len(strided.shape)))
    trans[-2] = orig[-1]
    trans[-1] = orig[-2]
    trans = trans[::-1]
    strided = strided.transpose(trans)
    return strided


def unwindow(arr, window_size, window_step=1, axis=0):
    # undo windows by broadcast
    if axis != 0:
        raise ValueError("axis != 0 currently unsupported")
    shp = arr.shape
    unwindowed = np.tile(arr[:, None, ...], (1, window_step, 1, 1))
    unwindowed = unwindowed.reshape(shp[0] * window_step, *shp[1:])
    return unwindowed.mean(axis=1)
