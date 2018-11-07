# License: BSD 3-clause
# Authors: Kyle Kastner
# LTSD routine from jfsantos (Joao Felipe Santos)
# Harvest, Cheaptrick, D4C, WORLD routines based on MATLAB code from M. Morise
# http://ml.cs.yamanashi.ac.jp/world/english/
# MGC code based on r9y9 (Ryuichi Yamamoto) MelGeneralizedCepstrums.jl
# Pieces also adapted from SPTK
from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.signal as sg
from scipy.interpolate import interp1d
from scipy.cluster.vq import vq
from scipy import linalg, fftpack
from numpy.testing import assert_almost_equal
from scipy.linalg import svd
from scipy.io import wavfile
from scipy.signal import firwin
from multiprocessing import Pool


def lpc_analysis(X, order=8, window_step=128, window_size=2 * 128,
                 emphasis=0.9, voiced_start_threshold=.9,
                 voiced_stop_threshold=.6, truncate=False, copy=True):
    """
    Extract LPC coefficients from a signal
    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/
    _rParameters
    ----------
    X : ndarray
        Signals to extract LPC coefficients from
    order : int, optional (default=8)
        Order of the LPC coefficients. For speech, use the general rule that the
        order is two times the expected number of formants plus 2.
        This can be formulated as 2 + 2 * (fs // 2000). For approx. signals
        with fs = 7000, this is 8 coefficients - 2 + 2 * (7000 // 2000).
    window_step : int, optional (default=128)
        The size (in samples) of the space between each window
    window_size : int, optional (default=2 * 128)
        The size of each window (in samples) to extract coefficients over
    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering
    voiced_start_threshold : float, optional (default=0.9)
        Upper power threshold for estimating when speech has started
    voiced_stop_threshold : float, optional (default=0.6)
        Lower power threshold for estimating when speech has stopped
    truncate : bool, optional (default=False)
        Whether to cut the data at the last window or do zero padding.
    copy : bool, optional (default=True)
        Whether to copy the input X or modify in place
    Returns
    -------
    lp_coefficients : ndarray
        lp coefficients to describe the frame
    per_frame_gain : ndarray
        calculated gain for each frame
    residual_excitation : ndarray
        leftover energy which is not described by lp coefficents and gain
    voiced_frames : ndarray
        array of [0, 1] values which holds voiced/unvoiced decision for each
        frame.
    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    X = np.array(X, copy=copy)
    if len(X.shape) < 2:
        X = X[None]

    n_points = X.shape[1]
    n_windows = int(n_points // window_step)
    if not truncate:
        pad_sizes = [(window_size - window_step) // 2,
                     window_size - window_step // 2]
        # TODO: Handling for odd window sizes / steps
        X = np.hstack((np.zeros((X.shape[0], int(pad_sizes[0]))), X,
                       np.zeros((X.shape[0], int(pad_sizes[1])))))
    else:
        pad_sizes = [0, 0]
        X = X[0, :n_windows * window_step]

    lp_coefficients = np.zeros((n_windows, order + 1))
    per_frame_gain = np.zeros((n_windows, 1))
    residual_excitation = np.zeros(
        int(((n_windows - 1) * window_step + window_size)))
    # Pre-emphasis high-pass filter
    X = sg.lfilter([1, -emphasis], 1, X)
    # stride_tricks.as_strided?
    autocorr_X = np.zeros((n_windows, int(2 * window_size - 1)))
    for window in range(max(n_windows - 1, 1)):
        wtws = int(window * window_step)
        XX = X.ravel()[wtws + np.arange(window_size, dtype="int32")]
        WXX = XX * sg.windows.hann(window_size)
        autocorr_X[window] = np.correlate(WXX, WXX, mode='full')
        center = np.argmax(autocorr_X[window])
        RXX = autocorr_X[window,
                         np.arange(center, window_size + order, dtype="int32")]
        R = linalg.toeplitz(RXX[:-1])
        solved_R = linalg.pinv(R).dot(RXX[1:])
        filter_coefs = np.hstack((1, -solved_R))
        residual_signal = sg.lfilter(filter_coefs, 1, WXX)
        gain = np.sqrt(np.mean(residual_signal ** 2))
        lp_coefficients[window] = filter_coefs
        per_frame_gain[window] = gain
        assign_range = wtws + np.arange(window_size, dtype="int32")
        residual_excitation[assign_range] += residual_signal / gain
    # Throw away first part in overlap mode for proper synthesis
    residual_excitation = residual_excitation[int(pad_sizes[0]):]
    return lp_coefficients, per_frame_gain, residual_excitation


def lpc_to_frequency(lp_coefficients, per_frame_gain):
    """
    Extract resonant frequencies and magnitudes from LPC coefficients and gains.
    Parameters
    ----------
    lp_coefficients : ndarray
        LPC coefficients, such as those calculated by ``lpc_analysis``
    per_frame_gain : ndarray
       Gain calculated for each frame, such as those calculated
       by ``lpc_analysis``
    Returns
    -------
    frequencies : ndarray
       Resonant frequencies calculated from LPC coefficients and gain. Returned
       frequencies are from 0 to 2 * pi
    magnitudes : ndarray
       Magnitudes of resonant frequencies
    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    n_windows, order = lp_coefficients.shape

    frame_frequencies = np.zeros((n_windows, (order - 1) // 2))
    frame_magnitudes = np.zeros_like(frame_frequencies)

    for window in range(n_windows):
        w_coefs = lp_coefficients[window]
        g_coefs = per_frame_gain[window]
        roots = np.roots(np.hstack(([1], w_coefs[1:])))
        # Roots doesn't return the same thing as MATLAB... agh
        frequencies, index = np.unique(
            np.abs(np.angle(roots)), return_index=True)
        # Make sure 0 doesn't show up...
        gtz = np.where(frequencies > 0)[0]
        frequencies = frequencies[gtz]
        index = index[gtz]
        magnitudes = g_coefs / (1. - np.abs(roots))
        sort_index = np.argsort(frequencies)
        frame_frequencies[window, :len(sort_index)] = frequencies[sort_index]
        frame_magnitudes[window, :len(sort_index)] = magnitudes[sort_index]
    return frame_frequencies, frame_magnitudes


def lpc_to_lsf(all_lpc):
    if len(all_lpc.shape) < 2:
        all_lpc = all_lpc[None]
    order = all_lpc.shape[1] - 1
    all_lsf = np.zeros((len(all_lpc), order))
    for i in range(len(all_lpc)):
        lpc = all_lpc[i]
        lpc1 = np.append(lpc, 0)
        lpc2 = lpc1[::-1]
        sum_filt = lpc1 + lpc2
        diff_filt = lpc1 - lpc2

        if order % 2 != 0:
            deconv_diff, _ = sg.deconvolve(diff_filt, [1, 0, -1])
            deconv_sum = sum_filt
        else:
            deconv_diff, _ = sg.deconvolve(diff_filt, [1, -1])
            deconv_sum, _ = sg.deconvolve(sum_filt, [1, 1])

        roots_diff = np.roots(deconv_diff)
        roots_sum = np.roots(deconv_sum)
        angle_diff = np.angle(roots_diff[::2])
        angle_sum = np.angle(roots_sum[::2])
        lsf = np.sort(np.hstack((angle_diff, angle_sum)))
        if len(lsf) != 0:
            all_lsf[i] = lsf
    return np.squeeze(all_lsf)


def lsf_to_lpc(all_lsf):
    if len(all_lsf.shape) < 2:
        all_lsf = all_lsf[None]
    order = all_lsf.shape[1]
    all_lpc = np.zeros((len(all_lsf), order + 1))
    for i in range(len(all_lsf)):
        lsf = all_lsf[i]
        zeros = np.exp(1j * lsf)
        sum_zeros = zeros[::2]
        diff_zeros = zeros[1::2]
        sum_zeros = np.hstack((sum_zeros, np.conj(sum_zeros)))
        diff_zeros = np.hstack((diff_zeros, np.conj(diff_zeros)))
        sum_filt = np.poly(sum_zeros)
        diff_filt = np.poly(diff_zeros)

        if order % 2 != 0:
            deconv_diff = sg.convolve(diff_filt, [1, 0, -1])
            deconv_sum = sum_filt
        else:
            deconv_diff = sg.convolve(diff_filt, [1, -1])
            deconv_sum = sg.convolve(sum_filt, [1, 1])

        lpc = .5 * (deconv_sum + deconv_diff)
        # Last coefficient is 0 and not returned
        all_lpc[i] = lpc[:-1]
    return np.squeeze(all_lpc)


def lpc_synthesis(lp_coefficients, per_frame_gain, residual_excitation=None,
                  voiced_frames=None, window_step=128, emphasis=0.9):
    """
    Synthesize a signal from LPC coefficients
    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/
        http://web.uvic.ca/~tyoon/resource/auditorytoolbox/auditorytoolbox/synlpc.html
    Parameters
    ----------
    lp_coefficients : ndarray
        Linear prediction coefficients
    per_frame_gain : ndarray
        Gain coefficients
    residual_excitation : ndarray or None, optional (default=None)
        Residual excitations. If None, this will be synthesized with white noise
    voiced_frames : ndarray or None, optional (default=None)
        Voiced frames. If None, all frames assumed to be voiced.
    window_step : int, optional (default=128)
        The size (in samples) of the space between each window
    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering
    overlap_add : bool, optional (default=True)
        What type of processing to use when joining windows
    copy : bool, optional (default=True)
       Whether to copy the input X or modify in place
    Returns
    -------
    synthesized : ndarray
        Sound vector synthesized from input arguments
    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    # TODO: Incorporate better synthesis from
    # http://eecs.oregonstate.edu/education/docs/ece352/CompleteManual.pdf
    window_size = 2 * window_step
    [n_windows, order] = lp_coefficients.shape

    n_points = (n_windows + 1) * window_step
    n_excitation_points = n_points + window_step + window_step // 2

    random_state = np.random.RandomState(1999)
    if residual_excitation is None:
        # Need to generate excitation
        if voiced_frames is None:
            # No voiced/unvoiced info
            voiced_frames = np.ones((lp_coefficients.shape[0], 1))
        residual_excitation = np.zeros((n_excitation_points))
        f, m = lpc_to_frequency(lp_coefficients, per_frame_gain)
        t = np.linspace(0, 1, window_size, endpoint=False)
        hanning = sg.windows.hann(window_size)
        for window in range(n_windows):
            window_base = window * window_step
            index = window_base + np.arange(window_size)
            if voiced_frames[window]:
                sig = np.zeros_like(t)
                cycles = np.cumsum(f[window][0] * t)
                sig += sg.sawtooth(cycles, 0.001)
                residual_excitation[index] += hanning * sig
            residual_excitation[index] += hanning * 0.01 * random_state.randn(
                window_size)
    else:
        n_excitation_points = residual_excitation.shape[0]
        n_points = n_excitation_points + window_step + window_step // 2
    residual_excitation = np.hstack((residual_excitation,
                                     np.zeros(window_size)))
    if voiced_frames is None:
        voiced_frames = np.ones_like(per_frame_gain)

    synthesized = np.zeros((n_points))
    for window in range(n_windows):
        window_base = window * window_step
        oldbit = synthesized[window_base + np.arange(window_step)]
        w_coefs = lp_coefficients[window]
        if not np.all(w_coefs):
            # Hack to make lfilter avoid
            # ValueError: BUG: filter coefficient a[0] == 0 not supported yet
            # when all coeffs are 0
            w_coefs = [1]
        g_coefs = voiced_frames[window] * per_frame_gain[window]
        index = window_base + np.arange(window_size)
        newbit = g_coefs * sg.lfilter([1], w_coefs,
                                      residual_excitation[index])
        synthesized[index] = np.hstack((oldbit, np.zeros(
            (window_size - window_step))))
        synthesized[index] += sg.windows.hann(window_size) * newbit
    synthesized = sg.lfilter([1], [1, -emphasis], synthesized)
    return synthesized


def run_lpc_example(sinusoid_sample_rate, num_components, overall_window_size,
                    lpc_coefficients):
    # ae.wav is from
    # http://www.linguistics.ucla.edu/people/hayes/103/Charts/VChart/ae.wav
    # Partially following the formant tutorial here
    # http://www.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html

    samplerate, X = wavfile.read('./results/LJ001-0001.wav') #fetch_sample_music()

    # compress c
    c = overlap_dct_compress(X, num_components, overall_window_size) # components, window_size
    # uncompress
    X_r = overlap_dct_uncompress(c, overall_window_size)
    # write this uncompressed form
    wavfile.write('./results/lpc_uncompress.wav', samplerate, soundsc(X_r))

    print("Calculating sinusoids")
    f_hz, m = sinusoid_analysis(X, input_sample_rate=sinusoid_sample_rate)
    Xs_sine = sinusoid_synthesis(f_hz, m)
    orig_fname = "./results/lpc_orig_{}.wav".format(overall_window_size)
    sine_fname = "./results/lpc_sine_synth_{}.wav".format(overall_window_size)
    wavfile.write(orig_fname, samplerate, soundsc(X))
    wavfile.write(sine_fname, samplerate, soundsc(Xs_sine))

    # lpc_order_list = [8, ]
    lpc_order_list = [2, 4, 6, 8] #[lpc_coefficients, ]
    dct_components_list = [50, 100, 150, 200] ##[num_components, ]
    window_size_list = [100, 200, 300, 400] #[overall_window_size, ]
    # Seems like a dct component size of ~2/3rds the step
    # (1/3rd the window for 50% overlap) works well.
    for lpc_order in lpc_order_list:
        for dct_components in dct_components_list:
            for window_size in window_size_list:
                # 50% overlap
                window_step = window_size // 2
                a, g, e = lpc_analysis(X, order=lpc_order,
                                       window_step=window_step,
                                       window_size=window_size, emphasis=0.9,
                                       copy=True)
                print("Calculating LSF")
                lsf = lpc_to_lsf(a)
                # Not window_size - window_step! Need to implement overlap
                print("Calculating compression")
                c = dct_compress(e, n_components=dct_components,
                             window_size=window_step)
                co = overlap_dct_compress(e, n_components=dct_components,
                                      window_size=window_step)
                block_excitation = dct_uncompress(c, window_size=window_step)
                overlap_excitation = overlap_dct_uncompress(co,
                                                        window_size=window_step)
                # compute lpc
                a_r = lsf_to_lpc(lsf)
                # from PIL import Image
                # img = Image.fromarray(np.transpose(a_r), 'RGB')
                # img.save('./lpc.png')
                # compute frequency
                f, m = lpc_to_frequency(a_r, g)
                # compute synthesis
                block_lpc = lpc_synthesis(a_r, g, block_excitation,
                                          emphasis=0.9,
                                          window_step=window_step)
                overlap_lpc = lpc_synthesis(a_r, g, overlap_excitation,
                                            emphasis=0.9,
                                            window_step=window_step)
                v, p = voiced_unvoiced(X, window_size=window_size,
                                       window_step=window_step)
                noisy_lpc = lpc_synthesis(a_r, g, voiced_frames=v,
                                          emphasis=0.9,
                                          window_step=window_step)
                if dct_components is None:
                    dct_components = window_size
                noisy_fname = './results/lpc_noisy_synth_%iwin_%ilpc_%idct.wav' % (
                    window_size, lpc_order, dct_components)
                block_fname = './results/lpc_block_synth_%iwin_%ilpc_%idct.wav' % (
                    window_size, lpc_order, dct_components)
                overlap_fname = './results/lpc_overlap_synth_%iwin_%ilpc_%idct.wav' % (
                    window_size, lpc_order, dct_components)
                wavfile.write(noisy_fname, samplerate, soundsc(noisy_lpc))
                wavfile.write(block_fname, samplerate,
                              soundsc(block_lpc))
                wavfile.write(overlap_fname, samplerate,
                              soundsc(overlap_lpc))

if __name__ == "__main__":
    run_lpc_example(16000, 200, 400, 8)
