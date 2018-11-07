# License: BSD 3-clause
# Authors: Kyle Kastner
# LTSD routine from jfsantos (Joao Felipe Santos)
# Harvest, Cheaptrick, D4C, WORLD routines based on MATLAB code from M. Morise
# http://ml.cs.yamanashi.ac.jp/world/english/
# MGC code based on r9y9 (Ryuichi Yamamoto) MelGeneralizedCepstrums.jl
# Pieces also adapted from SPTK
from __future__ import division
import numpy as np
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

try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib


def _mgc_b2c(wc, c, alpha):
    wc_o = np.zeros_like(wc)
    desired_order = len(wc) - 1
    for i in range(0, len(c))[::-1]:
        prev = copy.copy(wc_o)
        wc_o[0] = c[i]
        if desired_order >= 1:
            wc_o[1] = (1. - alpha ** 2) * prev[0] + alpha * prev[1]
        for m in range(2, desired_order + 1):
            wc_o[m] = prev[m - 1] + alpha * (prev[m] - wc_o[m - 1])
    return wc_o


def _mgc_ptrans(p, m, alpha):
    d = 0.
    o = 0.

    d = p[m]
    for i in range(1, m)[::-1]:
        o = p[i] + alpha * d
        d = p[i]
        p[i] = o

    o = alpha * d
    p[0] = (1. - alpha ** 2) * p[0] + 2 * o


def _mgc_qtrans(q, m, alpha):
    d = q[1]
    for i in range(2, 2 * m + 1):
        o = q[i] + alpha * d
        d = q[i]
        q[i] = o


def _mgc_gain(er, c, m, g):
    t = 0.
    if g != 0:
        for i in range(1, m + 1):
            t += er[i] * c[i]
        return er[0] + g * t
    else:
        return er[0]


def _mgc_fill_toeplitz(A, t):
    n = len(t)
    for i in range(n):
        for j in range(n):
            A[i, j] = t[i - j] if i - j >= 0 else t[j - i]


def _mgc_fill_hankel(A, t):
    n = len(t) // 2 + 1
    for i in range(n):
        for j in range(n):
            A[i, j] = t[i + j]


def _mgc_ignorm(c, gamma):
    if gamma == 0.:
        c[0] = np.log(c[0])
        return c
    gain = c[0] ** gamma
    c[1:] *= gain
    c[0] = (gain - 1.) / gamma


def _mgc_gnorm(c, gamma):
    if gamma == 0.:
        c[0] = np.exp(c[0])
        return c
    gain = 1. + gamma * c[0]
    c[1:] /= gain
    c[0] = gain ** (1. / gamma)


def _mgc_b2mc(mc, alpha):
    m = len(mc)
    o = 0.
    d = mc[m - 1]
    for i in range(m - 1)[::-1]:
        o = mc[i] + alpha * d
        d = mc[i]
        mc[i] = o


def _mgc_mc2b(mc, alpha):
    itr = range(len(mc) - 1)[::-1]
    for i in itr:
        mc[i] = mc[i] - alpha * mc[i + 1]


def _mgc_gc2gc(src_ceps, src_gamma=0., dst_order=None, dst_gamma=0.):
    if dst_order == None:
        dst_order = len(src_ceps) - 1

    dst_ceps = np.zeros((dst_order + 1,), dtype=src_ceps.dtype)
    dst_order = len(dst_ceps) - 1
    m1 = len(src_ceps) - 1
    dst_ceps[0] = copy.deepcopy(src_ceps[0])

    for m in range(2, dst_order + 2):
        ss1 = 0.
        ss2 = 0.
        min_1 = m1 if (m1 < m - 1) else m - 2
        itr = range(2, min_1 + 2)
        if len(itr) < 1:
            if min_1 + 1 == 2:
                itr = [2]
            else:
                itr = []

        """
        # old slower version
        for k in itr:
            assert k >= 1
            assert (m - k) >= 0
            cc = src_ceps[k - 1] * dst_ceps[m - k]
            ss2 += (k - 1) * cc
            ss1 += (m - k) * cc
        """

        if len(itr) > 0:
            itr = np.array(itr)
            cc_a = src_ceps[itr - 1] * dst_ceps[m - itr]
            ss2 += ((itr - 1) * cc_a).sum()
            ss1 += ((m - itr) * cc_a).sum()

        if m <= m1 + 1:
            dst_ceps[m - 1] = src_ceps[m - 1] + (dst_gamma * ss2 - src_gamma * ss1)/(m - 1.)
        else:
            dst_ceps[m - 1] = (dst_gamma * ss2 - src_gamma * ss1) / (m - 1.)
    return dst_ceps


def _mgc_newton(mgc_stored, periodogram, order, alpha, gamma,
        recursion_order, iter_number, y_fft, z_fft, cr, pr, rr, ri,
        qr, qi, Tm, Hm, Tm_plus_Hm, b):
    # a lot of inplace operations to match the Julia code
    cr[1:order + 1] = mgc_stored[1:order + 1]

    if alpha != 0:
        cr_res = _mgc_b2c(cr[:recursion_order + 1], cr[:order + 1], -alpha)
        cr[:recursion_order + 1] = cr_res[:]

    y = sp.fftpack.fft(np.cast["float64"](cr))
    c = mgc_stored
    x = periodogram
    if gamma != 0.:
        gamma_inv = 1. / gamma
    else:
        gamma_inv = np.inf

    if gamma == -1.:
        pr[:] = copy.deepcopy(x)
        new_pr = copy.deepcopy(pr)
    elif gamma == 0.:
        pr[:] = copy.deepcopy(x) / np.exp(2 * np.real(y))
        new_pr = copy.deepcopy(pr)
    else:
        tr = 1. + gamma * np.real(y)
        ti = -gamma * np.imag(y)
        trr = tr * tr
        tii = ti * ti
        s = trr + tii
        t = x * np.power(s, (-gamma_inv))
        t /= s
        pr[:] = t
        rr[:] = tr * t
        ri[:] = ti * t
        t /= s
        qr[:] = (trr - tii) * t
        s = tr * ti * t
        qi[:] = (s + s)
        new_pr = copy.deepcopy(pr)

    if gamma != -1.:
        """
        print()
        print(pr.sum())
        print(rr.sum())
        print(ri.sum())
        print(qr.sum())
        print(qi.sum())
        print()
        """
        pass

    y_fft[:] = copy.deepcopy(pr) + 0.j
    z_fft[:] = np.fft.fft(y_fft) / len(y_fft)
    pr[:] = copy.deepcopy(np.real(z_fft))
    if alpha != 0.:
        idx_1 = pr[:2 * order + 1]
        idx_2 = pr[:recursion_order + 1]
        idx_3 = _mgc_b2c(idx_1, idx_2, alpha)
        pr[:2 * order + 1] = idx_3[:]

    if gamma == 0. or gamma == -1.:
        qr[:2 * order + 1] = pr[:2 * order + 1]
        rr[:order + 1] = copy.deepcopy(pr[:order + 1])
    else:
        for i in range(len(qr)):
            y_fft[i] = qr[i] + 1j * qi[i]
        z_fft[:] = np.fft.fft(y_fft) / len(y_fft)
        qr[:] = np.real(z_fft)

        for i in range(len(rr)):
            y_fft[i] = rr[i] + 1j * ri[i]
        z_fft[:] = np.fft.fft(y_fft) / len(y_fft)
        rr[:] = np.real(z_fft)

        if alpha != 0.:
            qr_new = _mgc_b2c(qr[:recursion_order + 1], qr[:recursion_order + 1], alpha)
            qr[:recursion_order + 1] = qr_new[:]
            rr_new = _mgc_b2c(rr[:order + 1], rr[:recursion_order + 1], alpha)
            rr[:order + 1] = rr_new[:]

    if alpha != 0:
        _mgc_ptrans(pr, order, alpha)
        _mgc_qtrans(qr, order, alpha)

    eta = 0.
    if gamma != -1.:
        eta = _mgc_gain(rr, c, order, gamma)
        c[0] = np.sqrt(eta)

    if gamma == -1.:
        qr[:] = 0.
    elif gamma != 0.:
        for i in range(2, 2 * order + 1):
            qr[i] *= 1. + gamma

    te = pr[:order]
    _mgc_fill_toeplitz(Tm, te)
    he = qr[2: 2 * order + 1]
    _mgc_fill_hankel(Hm, he)

    Tm_plus_Hm[:] = Hm[:] + Tm[:]
    b[:order] = rr[1:order + 1]
    res = np.linalg.solve(Tm_plus_Hm, b)
    b[:] = res[:]

    c[1:order + 1] += res[:order]

    if gamma == -1.:
        eta = _mgc_gain(rr, c, order, gamma)
        c[0] = np.sqrt(eta)
    return np.log(eta), new_pr


def _mgc_mgcepnorm(b_gamma, alpha, gamma, otype):
    if otype != 0:
        raise ValueError("Not yet implemented for otype != 0")

    mgc = copy.deepcopy(b_gamma)
    _mgc_ignorm(mgc, gamma)
    _mgc_b2mc(mgc, alpha)
    return mgc


def _sp2mgc(sp, order=20, alpha=0.35, gamma=-0.41, miniter=2, maxiter=30, criteria=0.001, otype=0, verbose=False):
    # Based on r9y9 Julia code
    # https://github.com/r9y9/MelGeneralizedCepstrums.jl
    periodogram = np.abs(sp) ** 2
    recursion_order = len(periodogram) - 1
    slen = len(periodogram)
    iter_number = 1

    def _z():
        return np.zeros((slen,), dtype="float64")

    def _o():
        return np.zeros((order,), dtype="float64")

    def _o2():
        return np.zeros((order, order), dtype="float64")

    cr = _z()
    pr = _z()
    rr = _z()
    ri = _z().astype("float128")
    qr = _z()
    qi = _z().astype("float128")
    Tm = _o2()
    Hm = _o2()
    Tm_plus_Hm = _o2()
    b = _o()
    y = _z() + 0j
    z = _z() + 0j
    b_gamma = np.zeros((order + 1,), dtype="float64")
    # return pr_new due to oddness with Julia having different numbers
    # in pr at end of function vs back in this scope
    eta0, pr_new = _mgc_newton(b_gamma, periodogram, order, alpha, -1.,
                               recursion_order, iter_number, y, z, cr, pr, rr,
                               ri, qr, qi, Tm, Hm, Tm_plus_Hm, b)
    pr[:] = pr_new
    """
    print(eta0)
    print(sum(b_gamma))
    print(sum(periodogram))
    print(order)
    print(alpha)
    print(recursion_order)
    print(sum(y))
    print(sum(cr))
    print(sum(z))
    print(sum(pr))
    print(sum(rr))
    print(sum(qi))
    print(Tm.sum())
    print(Hm.sum())
    print(sum(b))
    raise ValueError()
    """
    if gamma != -1.:
        d = np.zeros((order + 1,), dtype="float64")
        if alpha != 0.:
            _mgc_ignorm(b_gamma, -1.)
            _mgc_b2mc(b_gamma, alpha)
            d = copy.deepcopy(b_gamma)
            _mgc_gnorm(d, -1.)
            # numbers are slightly different here - numerical diffs?
        else:
            d = copy.deepcopy(b_gamma)
        b_gamma = _mgc_gc2gc(d, -1., order, gamma)

        if alpha != 0.:
            _mgc_ignorm(b_gamma, gamma)
            _mgc_mc2b(b_gamma, alpha)
            _mgc_gnorm(b_gamma, gamma)

    if gamma != -1.:
        eta_t = eta0
        for i in range(1, maxiter + 1):
            eta, pr_new = _mgc_newton(b_gamma, periodogram, order, alpha,
                    gamma, recursion_order, i, y, z, cr, pr, rr,
                    ri, qr, qi, Tm, Hm, Tm_plus_Hm, b)
            pr[:] = pr_new
            """
            print(eta0)
            print(sum(b_gamma))
            print(sum(periodogram))
            print(order)
            print(alpha)
            print(recursion_order)
            print(sum(y))
            print(sum(cr))
            print(sum(z))
            print(sum(pr))
            print(sum(rr))
            print(sum(qi))
            print(Tm.sum())
            print(Hm.sum())
            print(sum(b))
            raise ValueError()
            """
            err = np.abs((eta_t - eta) / eta)
            if verbose:
                print("iter %i, criterion: %f" % (i, err))
            if i >= miniter:
                if err < criteria:
                    if verbose:
                        print("optimization complete at iter %i" % i)
                    break
            eta_t = eta
    mgc_arr = _mgc_mgcepnorm(b_gamma, alpha, gamma, otype)
    return mgc_arr


_sp_convert_results = []

def _sp_collect_result(result):
    _sp_convert_results.append(result)


def _sp_convert(c_i, order, alpha, gamma, miniter, maxiter, criteria,
        otype, verbose):
    i = c_i[0]
    tot_i = c_i[1]
    sp_i = c_i[2]
    r_i = (i, _sp2mgc(sp_i, order=order, alpha=alpha, gamma=gamma,
                miniter=miniter, maxiter=maxiter, criteria=criteria,
                otype=otype, verbose=verbose))
    return r_i


def sp2mgc(sp, order=20, alpha=0.35, gamma=-0.41, miniter=2,
        maxiter=30, criteria=0.001, otype=0, verbose=False):
    """
    Accepts 1D or 2D one-sided spectrum (complex or real valued).
    If 2D, assumes time is axis 0.
    Returns mel generalized cepstral coefficients.
    Based on r9y9 Julia code
    https://github.com/r9y9/MelGeneralizedCepstrums.jl
    """

    if len(sp.shape) == 1:
        sp = np.concatenate((sp, sp[:, 1:][:, ::-1]), axis=0)
        return _sp2mgc(sp, order=order, alpha=alpha, gamma=gamma,
                miniter=miniter, maxiter=maxiter, criteria=criteria,
                otype=otype, verbose=verbose)
    else:
        sp = np.concatenate((sp, sp[:, 1:][:, ::-1]), axis=1)
        # Slooow, use multiprocessing to speed up a bit
        # http://blog.shenwei.me/python-multiprocessing-pool-difference-between-map-apply-map_async-apply_async/
        # http://stackoverflow.com/questions/5666576/show-the-progress-of-a-python-multiprocessing-pool-map-call
        c = [(i + 1, sp.shape[0], sp[i]) for i in range(sp.shape[0])]
        p = Pool()
        start = time.time()
        if verbose:
            print("Starting conversion of %i frames" % sp.shape[0])
            print("This may take some time...")

        # takes ~360s for 630 frames, 1 process
        itr = p.map_async(functools.partial(_sp_convert, order=order, alpha=alpha, gamma=gamma, miniter=miniter, maxiter=maxiter, criteria=criteria, otype=otype, verbose=False), c, callback=_sp_collect_result)

        sz = len(c) // itr._chunksize
        if (sz * itr._chunksize) != len(c):
            sz += 1

        last_remaining = None
        while True:
            remaining = itr._number_left
            if verbose:
                if remaining != last_remaining:
                    last_remaining = remaining
                    print("%i chunks of %i complete" % (sz - remaining, sz))
            if itr.ready():
                break
            time.sleep(.5)

        """
        # takes ~455s for 630 frames
        itr = p.imap_unordered(functools.partial(_sp_convert, order=order, alpha=alpha, gamma=gamma, miniter=miniter, maxiter=maxiter, criteria=criteria, otype=otype, verbose=False), c)
        res = []
        # print ~every 5%
        mod = int(len(c)) // 20
        if mod < 1:
            mod = 1
        for i, res_i in enumerate(itr, 1):
            res.append(res_i)
            if i % mod == 0 or i == 1:
                print("%i of %i complete" % (i, len(c)))
        """
        p.close()
        p.join()
        stop = time.time()
        if verbose:
            print("Processed %i frames in %s seconds" % (sp.shape[0], stop - start))
        # map_async result comes in chunks
        flat = [a_i for a in _sp_convert_results for a_i in a]
        final = [o[1] for o in sorted(flat, key=lambda x: x[0])]
        for i in range(len(_sp_convert_results)):
            _sp_convert_results.pop()
        return np.array(final)


def win2mgc(windowed_signal, order=20, alpha=0.35, gamma=-0.41, miniter=2,
        maxiter=30, criteria=0.001, otype=0, verbose=False):
    """
    Accepts 1D or 2D array of windowed signal frames.
    If 2D, assumes time is axis 0.
    Returns mel generalized cepstral coefficients.
    Based on r9y9 Julia code
    https://github.com/r9y9/MelGeneralizedCepstrums.jl
    """
    if len(windowed_signal.shape) == 1:
        sp = np.fft.fft(windowed_signal)
        return _sp2mgc(sp, order=order, alpha=alpha, gamma=gamma,
                miniter=miniter, maxiter=maxiter, criteria=criteria,
                otype=otype, verbose=verbose)
    else:
        raise ValueError("2D input not yet complete for win2mgc")


def _mgc_freqt(wc, c, alpha):
    prev = np.zeros_like(wc)
    dst_order = len(wc) - 1
    wc *= 0
    m1 = len(c) - 1
    for i in range(-m1, 1, 1):
        prev[:] = wc
        if dst_order >= 0:
            wc[0] = c[-i] + alpha * prev[0]
        if dst_order >= 1:
            wc[1] = (1. - alpha * alpha) * prev[0] + alpha * prev[1]
        for m in range(2, dst_order + 1):
            wc[m] = prev[m - 1] + alpha * (prev[m] - wc[m - 1])


def _mgc_mgc2mgc(src_ceps, src_alpha, src_gamma, dst_order, dst_alpha, dst_gamma):
    dst_ceps = np.zeros((dst_order + 1,))
    alpha = (dst_alpha - src_alpha) / (1. - dst_alpha * src_alpha)
    if alpha == 0.:
        new_dst_ceps = copy.deepcopy(src_ceps)
        _mgc_gnorm(new_dst_ceps, src_gamma)
        dst_ceps = _mgc_gc2gc(new_dst_ceps, src_gamma, dst_order, dst_gamma)
        _mgc_ignorm(dst_ceps, dst_gamma)
    else:
        _mgc_freqt(dst_ceps, src_ceps, alpha)
        _mgc_gnorm(dst_ceps, src_gamma)
        new_dst_ceps = copy.deepcopy(dst_ceps)
        dst_ceps = _mgc_gc2gc(new_dst_ceps, src_gamma, dst_order, dst_gamma)
        _mgc_ignorm(dst_ceps, dst_gamma)
    return dst_ceps


_mgc_convert_results = []

def _mgc_collect_result(result):
    _mgc_convert_results.append(result)


def _mgc_convert(c_i, alpha, gamma, fftlen):
    i = c_i[0]
    tot_i = c_i[1]
    mgc_i = c_i[2]
    r_i = (i, _mgc_mgc2mgc(mgc_i, src_alpha=alpha, src_gamma=gamma,
                dst_order=fftlen // 2, dst_alpha=0., dst_gamma=0.))
    return r_i


def mgc2sp(mgc_arr, alpha=0.35, gamma=-0.41, fftlen="auto", fs=None,
        mode="world_pad", verbose=False):
    """
    Accepts 1D or 2D array of mgc
    If 2D, assume time is on axis 0
    Returns reconstructed smooth spectrum
    Based on r9y9 Julia code
    https://github.com/r9y9/MelGeneralizedCepstrums.jl
    """
    if mode != "world_pad":
        raise ValueError("Only currently supported mode is world_pad")

    if fftlen == "auto":
        if fs == None:
            raise ValueError("fs must be provided for fftlen 'auto'")
        f0_low_limit = 71
        fftlen = int(2 ** np.ceil(np.log2(3. * float(fs) / f0_low_limit + 1)))
        if verbose:
            print("setting fftlen to %i" % fftlen)

    if len(mgc_arr.shape) == 1:
        c = _mgc_mgc2mgc(mgc_arr, alpha, gamma, fftlen // 2, 0., 0.)
        buf = np.zeros((fftlen,), dtype=c.dtype)
        buf[:len(c)] = c[:]
        return np.fft.rfft(buf)
    else:
        # Slooow, use multiprocessing to speed up a bit
        # http://blog.shenwei.me/python-multiprocessing-pool-difference-between-map-apply-map_async-apply_async/
        # http://stackoverflow.com/questions/5666576/show-the-progress-of-a-python-multiprocessing-pool-map-call
        c = [(i + 1, mgc_arr.shape[0], mgc_arr[i]) for i in range(mgc_arr.shape[0])]
        p = Pool()
        start = time.time()
        if verbose:
            print("Starting conversion of %i frames" % mgc_arr.shape[0])
            print("This may take some time...")
        #itr = p.map(functools.partial(_mgc_convert, alpha=alpha, gamma=gamma, fftlen=fftlen), c)
        #raise ValueError()

        # 500.1 s for 630 frames process
        itr = p.map_async(functools.partial(_mgc_convert, alpha=alpha, gamma=gamma, fftlen=fftlen), c, callback=_mgc_collect_result)

        sz = len(c) // itr._chunksize
        if (sz * itr._chunksize) != len(c):
            sz += 1

        last_remaining = None
        while True:
            remaining = itr._number_left
            if verbose:
                if last_remaining != remaining:
                    last_remaining = remaining
                    print("%i chunks of %i complete" % (sz - remaining, sz))
            if itr.ready():
                break
            time.sleep(.5)
        p.close()
        p.join()
        stop = time.time()
        if verbose:
            print("Processed %i frames in %s seconds" % (mgc_arr.shape[0], stop - start))
        # map_async result comes in chunks
        flat = [a_i for a in _mgc_convert_results for a_i in a]
        final = [o[1] for o in sorted(flat, key=lambda x: x[0])]
        for i in range(len(_mgc_convert_results)):
            _mgc_convert_results.pop()
        c = np.array(final)
        buf = np.zeros((len(c), fftlen), dtype=c.dtype)
        buf[:, :c.shape[1]] = c[:]
        return np.exp(np.fft.rfft(buf, axis=-1).real)

def run_mgc_example():
    import matplotlib.pyplot as plt
    fs, x = wavfile.read("test16k.wav")
    pos = 3000
    fftlen = 1024
    win = np.blackman(fftlen) / np.sqrt(np.sum(np.blackman(fftlen) ** 2))
    xw = x[pos:pos + fftlen] * win
    sp = 20 * np.log10(np.abs(np.fft.rfft(xw)))
    mgc_order = 20
    mgc_alpha = 0.41
    mgc_gamma = -0.35
    mgc_arr = win2mgc(xw, order=mgc_order, alpha=mgc_alpha, gamma=mgc_gamma, verbose=True)
    xwsp = 20 * np.log10(np.abs(np.fft.rfft(xw)))
    sp = mgc2sp(mgc_arr, mgc_alpha, mgc_gamma, fftlen)
    plt.plot(xwsp)
    plt.plot(20. / np.log(10) * np.real(sp), "r")
    plt.xlim(1, len(xwsp))
    plt.show()
