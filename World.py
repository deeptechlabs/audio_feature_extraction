# License: BSD 3-clause
# Authors: Kyle Kastner
# LTSD routine from jfsantos (Joao Felipe Santos)
# Harvest, Cheaptrick, D4C, WORLD routines based on MATLAB code from M. Morise
# http://ml.cs.yamanashi.ac.jp/world/english/
# MGC code based on r9y9 (Ryuichi Yamamoto) MelGeneralizedCepstrums.jl
# Pieces also adapted from SPTK
from __future__ import division
import numpy as np
import scipy.signal as sg
from scipy.cluster.vq import vq
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import as_strided
from scipy import linalg, fftpack
from numpy.testing import assert_almost_equal
from scipy.linalg import svd
from scipy.io import wavfile
from scipy.signal import firwin


def harvest_get_downsampled_signal(x, fs, target_fs):
    decimation_ratio = np.round(fs / target_fs)
    offset = np.ceil(140. / decimation_ratio) * decimation_ratio
    start_pad = x[0] * np.ones(int(offset), dtype=np.float32)
    end_pad = x[-1] * np.ones(int(offset), dtype=np.float32)
    x = np.concatenate((start_pad, x, end_pad), axis=0)

    if fs < target_fs:
        raise ValueError("CASE NOT HANDLED IN harvest_get_downsampled_signal")
    else:
        try:
            y0 = sg.decimate(x, int(decimation_ratio), 3, zero_phase=True)
        except:
            y0 = sg.decimate(x, int(decimation_ratio), 3)
        actual_fs = fs / decimation_ratio
        y = y0[int(offset / decimation_ratio):-int(offset / decimation_ratio)]
    y = y - np.mean(y)
    return y, actual_fs


def harvest_get_raw_f0_candidates(number_of_frames, boundary_f0_list,
      y_length, temporal_positions, actual_fs, y_spectrum, f0_floor,
      f0_ceil):
    raw_f0_candidates = np.zeros((len(boundary_f0_list), number_of_frames), dtype=np.float32)
    for i in range(len(boundary_f0_list)):
        raw_f0_candidates[i, :] = harvest_get_f0_candidate_from_raw_event(
                boundary_f0_list[i], actual_fs, y_spectrum, y_length,
                temporal_positions, f0_floor, f0_ceil)
    return raw_f0_candidates


def harvest_nuttall(N):
    t = np.arange(0, N) * 2 * np.pi / (N - 1)
    coefs = np.array([0.355768, -0.487396, 0.144232, -0.012604])
    window = np.cos(t[:, None].dot(np.array([0., 1., 2., 3.])[None])).dot( coefs[:, None])
    # 1D window...
    return window.ravel()


def harvest_get_f0_candidate_from_raw_event(boundary_f0,
        fs, y_spectrum, y_length, temporal_positions, f0_floor,
        f0_ceil):
    filter_length_half = int(np.round(fs / boundary_f0 * 2))
    band_pass_filter_base = harvest_nuttall(filter_length_half * 2 + 1)
    shifter = np.cos(2 * np.pi * boundary_f0 * np.arange(-filter_length_half, filter_length_half + 1) / float(fs))
    band_pass_filter = band_pass_filter_base * shifter

    index_bias = filter_length_half
    # possible numerical issues if 32 bit
    spectrum_low_pass_filter = np.fft.fft(band_pass_filter.astype("float64"), len(y_spectrum))
    filtered_signal = np.real(np.fft.ifft(spectrum_low_pass_filter * y_spectrum))
    index_bias = filter_length_half + 1
    filtered_signal = filtered_signal[index_bias + np.arange(y_length).astype("int32")]
    negative_zero_cross = harvest_zero_crossing_engine(filtered_signal, fs)
    positive_zero_cross = harvest_zero_crossing_engine(-filtered_signal, fs)
    d_filtered_signal = filtered_signal[1:] - filtered_signal[:-1]
    peak = harvest_zero_crossing_engine(d_filtered_signal, fs)
    dip = harvest_zero_crossing_engine(-d_filtered_signal, fs)
    f0_candidate = harvest_get_f0_candidate_contour(negative_zero_cross,
            positive_zero_cross, peak, dip, temporal_positions)
    f0_candidate[f0_candidate > (boundary_f0 * 1.1)] = 0.
    f0_candidate[f0_candidate < (boundary_f0 * .9)] = 0.
    f0_candidate[f0_candidate > f0_ceil] = 0.
    f0_candidate[f0_candidate < f0_floor] = 0.
    return f0_candidate


def harvest_get_f0_candidate_contour(negative_zero_cross_tup,
        positive_zero_cross_tup, peak_tup, dip_tup, temporal_positions):
    # 0 is inteval locations
    # 1 is interval based f0
    usable_channel = max(0, len(negative_zero_cross_tup[0]) - 2)
    usable_channel *= max(0, len(positive_zero_cross_tup[0]) - 2)
    usable_channel *= max(0, len(peak_tup[0]) - 2)
    usable_channel *= max(0, len(dip_tup[0]) - 2)
    if usable_channel > 0:
        interpolated_f0_list = np.zeros((4, len(temporal_positions)))
        nz = interp1d(negative_zero_cross_tup[0], negative_zero_cross_tup[1],
                 kind="linear", bounds_error=False, fill_value="extrapolate")
        pz = interp1d(positive_zero_cross_tup[0], positive_zero_cross_tup[1],
                 kind="linear", bounds_error=False, fill_value="extrapolate")
        pkz = interp1d(peak_tup[0], peak_tup[1],
                  kind="linear", bounds_error=False, fill_value="extrapolate")
        dz = interp1d(dip_tup[0], dip_tup[1],
                  kind="linear", bounds_error=False, fill_value="extrapolate")
        interpolated_f0_list[0, :] = nz(temporal_positions)
        interpolated_f0_list[1, :] = pz(temporal_positions)
        interpolated_f0_list[2, :] = pkz(temporal_positions)
        interpolated_f0_list[3, :] = dz(temporal_positions)
        f0_candidate = np.mean(interpolated_f0_list, axis=0)
    else:
        f0_candidate = temporal_positions * 0
    return f0_candidate


def harvest_zero_crossing_engine(x, fs, debug=False):
    # negative zero crossing, going from positive to negative
    x_shift = x.copy()
    x_shift[:-1] = x_shift[1:]
    x_shift[-1] = x[-1]
    # +1 here to avoid edge case at 0
    points = np.arange(len(x)) + 1
    negative_going_points = points * ((x_shift * x < 0) * (x_shift < x))
    edge_list = negative_going_points[negative_going_points > 0]
    # -1 to correct index
    fine_edge_list = edge_list - x[edge_list - 1] / (x[edge_list] - x[edge_list - 1]).astype("float32")
    interval_locations = (fine_edge_list[:-1] + fine_edge_list[1:]) / float(2) / fs
    interval_based_f0 = float(fs) / (fine_edge_list[1:] - fine_edge_list[:-1])
    return interval_locations, interval_based_f0


def harvest_detect_official_f0_candidates(raw_f0_candidates):
    number_of_channels, number_of_frames = raw_f0_candidates.shape
    f0_candidates = np.zeros((int(np.round(number_of_channels / 10.)), number_of_frames))
    number_of_candidates = 0
    threshold = 10
    for i in range(number_of_frames):
        tmp = raw_f0_candidates[:, i].copy()
        tmp[tmp > 0] = 1.
        tmp[0] = 0
        tmp[-1] = 0
        tmp = tmp[1:] - tmp[:-1]
        st = np.where(tmp == 1)[0]
        ed = np.where(tmp == -1)[0]
        count = 0
        for j in range(len(st)):
            dif = ed[j] - st[j]
            if dif >= threshold:
                tmp_f0 = raw_f0_candidates[st[j] + 1: ed[j] + 1, i]
                f0_candidates[count, i] = np.mean(tmp_f0)
                count = count + 1
        number_of_candidates = max(number_of_candidates, count)
    return f0_candidates, number_of_candidates


def harvest_overlap_f0_candidates(f0_candidates, max_number_of_f0_candidates):
    n = 3 # this is the optimized parameter... apparently
    number_of_candidates = n * 2 + 1
    new_f0_candidates = f0_candidates[number_of_candidates, :].copy()
    new_f0_candidates = new_f0_candidates[None]
    # hack to bypass magic matlab-isms of allocating when indexing OOB
    new_f0_candidates = np.vstack([new_f0_candidates] + (new_f0_candidates.shape[-1] - 1) * [np.zeros_like(new_f0_candidates)])
    # this indexing is megagross, possible source for bugs!
    all_nonzero = []
    for i in range(number_of_candidates):
        st = max(-(i - n), 0)
        ed = min(-(i - n), 0)
        f1_b = np.arange(max_number_of_f0_candidates).astype("int32")
        f1 = f1_b + int(i * max_number_of_f0_candidates)
        all_nonzero = list(set(all_nonzero + list(f1)))
        f2 = None if ed == 0 else ed
        f3 = -ed
        f4 = None if st == 0 else -st
        new_f0_candidates[f1, st:f2] = f0_candidates[f1_b, f3:f4]
    new_f0_candidates = new_f0_candidates[all_nonzero, :]
    return new_f0_candidates


def harvest_refine_candidates(x, fs, temporal_positions, f0_candidates,
        f0_floor, f0_ceil):
    new_f0_candidates = f0_candidates.copy()
    f0_scores = f0_candidates * 0.
    for i in range(len(temporal_positions)):
        for j in range(len(f0_candidates)):
            tmp_f0 = f0_candidates[j, i]
            if tmp_f0 == 0:
                continue
            res = harvest_get_refined_f0(x, fs, temporal_positions[i],
                    tmp_f0, f0_floor, f0_ceil)
            new_f0_candidates[j, i] = res[0]
            f0_scores[j, i] = res[1]
    return new_f0_candidates, f0_scores


def harvest_get_refined_f0(x, fs, current_time, current_f0, f0_floor,
        f0_ceil):
    half_window_length = np.ceil(3. * fs / current_f0 / 2.)
    window_length_in_time = (2. * half_window_length + 1) / float(fs)
    base_time = np.arange(-half_window_length, half_window_length + 1) / float(fs)
    fft_size = int(2 ** np.ceil(np.log2((half_window_length * 2 + 1)) + 1))
    frequency_axis = np.arange(fft_size) / fft_size * float(fs)

    base_index = np.round((current_time + base_time) * fs + 0.001)
    index_time = (base_index - 1) / float(fs)
    window_time = index_time - current_time
    part1 = np.cos(2 * np.pi * window_time / window_length_in_time)
    part2 = np.cos(4 * np.pi * window_time / window_length_in_time)
    main_window = 0.42 + 0.5 * part1 + 0.08 * part2
    ext = np.zeros((len(main_window) + 2))
    ext[1:-1] = main_window
    diff_window = -((ext[1:-1] - ext[:-2]) + (ext[2:] - ext[1:-1])) / float(2)
    safe_index = np.maximum(1, np.minimum(len(x), base_index)).astype("int32") - 1
    spectrum = np.fft.fft(x[safe_index] * main_window, fft_size)
    diff_spectrum = np.fft.fft(x[safe_index] * diff_window, fft_size)
    numerator_i = np.real(spectrum) * np.imag(diff_spectrum) - np.imag(spectrum) * np.real(diff_spectrum)
    power_spectrum = np.abs(spectrum) ** 2
    instantaneous_frequency = frequency_axis + numerator_i / power_spectrum * float(fs) / 2. / np.pi

    number_of_harmonics = int(min(np.floor(float(fs) / 2. / current_f0), 6.))
    harmonics_index = np.arange(number_of_harmonics) + 1
    index_list = np.round(current_f0 * fft_size / fs * harmonics_index).astype("int32")
    instantaneous_frequency_list = instantaneous_frequency[index_list]
    amplitude_list = np.sqrt(power_spectrum[index_list])
    refined_f0 = np.sum(amplitude_list * instantaneous_frequency_list)
    refined_f0 /= np.sum(amplitude_list * harmonics_index.astype("float32"))

    variation = np.abs(((instantaneous_frequency_list / harmonics_index.astype("float32")) - current_f0) / float(current_f0))
    refined_score = 1. / (0.000000000001 + np.mean(variation))

    if (refined_f0 < f0_floor) or (refined_f0 > f0_ceil) or (refined_score < 2.5):
        refined_f0 = 0.
        redined_score = 0.
    return refined_f0, refined_score


def harvest_select_best_f0(reference_f0, f0_candidates, allowed_range):
    best_f0 = 0
    best_error = allowed_range

    for i in range(len(f0_candidates)):
        tmp = np.abs(reference_f0 - f0_candidates[i]) / reference_f0
        if tmp > best_error:
            continue
        best_f0 = f0_candidates[i]
        best_error = tmp
    return best_f0, best_error


def harvest_remove_unreliable_candidates(f0_candidates, f0_scores):
    new_f0_candidates = f0_candidates.copy()
    new_f0_scores = f0_scores.copy()
    threshold = 0.05
    f0_length = f0_candidates.shape[1]
    number_of_candidates = len(f0_candidates)

    for i in range(1, f0_length - 1):
        for j in range(number_of_candidates):
            reference_f0 = f0_candidates[j, i]
            if reference_f0 == 0:
                continue
            _, min_error1 = harvest_select_best_f0(reference_f0, f0_candidates[:, i + 1], 1)
            _, min_error2 = harvest_select_best_f0(reference_f0, f0_candidates[:, i - 1], 1)
            min_error = min([min_error1, min_error2])
            if min_error > threshold:
                new_f0_candidates[j, i] = 0
                new_f0_scores[j, i] = 0
    return new_f0_candidates, new_f0_scores


def harvest_search_f0_base(f0_candidates, f0_scores):
    f0_base = f0_candidates[0, :] * 0.
    for i in range(len(f0_base)):
        max_index = np.argmax(f0_scores[:, i])
        f0_base[i] = f0_candidates[max_index, i]
    return f0_base


def harvest_fix_step_1(f0_base, allowed_range):
    # Step 1: Rapid change of f0 contour is replaced by 0
    f0_step1 = f0_base.copy()
    f0_step1[0] = 0.
    f0_step1[1] = 0.

    for i in range(2, len(f0_base)):
        if f0_base[i] == 0:
            continue
        reference_f0 = f0_base[i - 1] * 2 - f0_base[i - 2]
        c1 = np.abs((f0_base[i] - reference_f0) / reference_f0) > allowed_range
        c2 = np.abs((f0_base[i] - f0_base[i - 1]) / f0_base[i - 1]) > allowed_range
        if c1 and c2:
            f0_step1[i] = 0.
    return f0_step1


def harvest_fix_step_2(f0_step1, voice_range_minimum):
    f0_step2 = f0_step1.copy()
    boundary_list = harvest_get_boundary_list(f0_step1)

    for i in range(1, int(len(boundary_list) / 2.) + 1):
        distance = boundary_list[(2 * i) - 1] - boundary_list[(2 * i) - 2]
        if distance < voice_range_minimum:
            # need one more due to range not including last index
            lb = boundary_list[(2 * i) - 2]
            ub = boundary_list[(2 * i) - 1] + 1
            f0_step2[lb:ub] = 0.
    return f0_step2


def harvest_fix_step_3(f0_step2, f0_candidates, allowed_range, f0_scores):
    f0_step3 = f0_step2.copy()
    boundary_list = harvest_get_boundary_list(f0_step2)
    multichannel_f0 = harvest_get_multichannel_f0(f0_step2, boundary_list)
    rrange = np.zeros((int(len(boundary_list) / 2), 2))
    threshold1 = 100
    threshold2 = 2200
    count = 0
    for i in range(1, int(len(boundary_list) / 2) + 1):
        # changed to 2 * i - 2
        extended_f0, tmp_range_1 = harvest_extend_f0(multichannel_f0[i - 1, :],
                boundary_list[(2 * i) - 1],
                min([len(f0_step2) - 1, boundary_list[(2 * i) - 1] + threshold1]),
                1, f0_candidates, allowed_range)
        tmp_f0_sequence, tmp_range_0 = harvest_extend_f0(extended_f0,
                boundary_list[(2 * i) - 2],
                max([2, boundary_list[(2 * i) - 2] - threshold1]), -1,
                f0_candidates, allowed_range)

        mean_f0 = np.mean(tmp_f0_sequence[tmp_range_0 : tmp_range_1 + 1])
        if threshold2 / mean_f0 < (tmp_range_1 - tmp_range_0):
            multichannel_f0[count, :] = tmp_f0_sequence
            rrange[count, :] = np.array([tmp_range_0, tmp_range_1])
            count = count + 1
    if count > 0:
        multichannel_f0 = multichannel_f0[:count, :]
        rrange = rrange[:count, :]
        f0_step3 = harvest_merge_f0(multichannel_f0, rrange, f0_candidates,
                f0_scores)
    return f0_step3


def harvest_merge_f0(multichannel_f0, rrange, f0_candidates, f0_scores):
    number_of_channels = len(multichannel_f0)
    sorted_order = np.argsort(rrange[:, 0])
    f0 = multichannel_f0[sorted_order[0], :]
    for i in range(1, number_of_channels):
        if rrange[sorted_order[i], 0] - rrange[sorted_order[0], 1] > 0:
            # no overlapping
            f0[int(rrange[sorted_order[i], 0]):int(rrange[sorted_order[i], 1])] = multichannel_f0[sorted_order[i], int(rrange[sorted_order[i], 0]):int(rrange[sorted_order[i], 1])]
            cp = rrange.copy()
            rrange[sorted_order[0], 0] = cp[sorted_order[i], 0]
            rrange[sorted_order[0], 1] = cp[sorted_order[i], 1]
        else:
            cp = rrange.copy()
            res = harvest_merge_f0_sub(f0, cp[sorted_order[0], 0],
                    cp[sorted_order[0], 1],
                    multichannel_f0[sorted_order[i], :],
                    cp[sorted_order[i], 0],
                    cp[sorted_order[i], 1], f0_candidates, f0_scores)
            f0 = res[0]
            rrange[sorted_order[0], 1] = res[1]
    return f0


def harvest_merge_f0_sub(f0_1, st1, ed1, f0_2, st2, ed2, f0_candidates,
        f0_scores):
    merged_f0 = f0_1
    if (st1 <= st2) and (ed1 >= ed2):
        new_ed = ed1
        return merged_f0, new_ed
    new_ed = ed2

    score1 = 0.
    score2 = 0.
    for i in range(int(st2), int(ed1) + 1):
        score1 = score1 + harvest_serach_score(f0_1[i], f0_candidates[:, i], f0_scores[:, i])
        score2 = score2 + harvest_serach_score(f0_2[i], f0_candidates[:, i], f0_scores[:, i])
    if score1 > score2:
        merged_f0[int(ed1):int(ed2) + 1] = f0_2[int(ed1):int(ed2) + 1]
    else:
        merged_f0[int(st2):int(ed2) + 1] = f0_2[int(st2):int(ed2) + 1]
    return merged_f0, new_ed


def harvest_serach_score(f0, f0_candidates, f0_scores):
    score = 0
    for i in range(len(f0_candidates)):
        if (f0 == f0_candidates[i]) and (score < f0_scores[i]):
            score = f0_scores[i]
    return score


def harvest_extend_f0(f0, origin, last_point, shift, f0_candidates,
        allowed_range):
    threshold = 4
    extended_f0 = f0.copy()
    tmp_f0 = extended_f0[origin]
    shifted_origin = origin
    count = 0

    for i in np.arange(origin, last_point + shift, shift):
        # off by 1 issues
        if (i + shift) >= f0_candidates.shape[1]:
            continue
        bf0, bs = harvest_select_best_f0(tmp_f0,
                f0_candidates[:, i + shift], allowed_range)
        extended_f0[i + shift] = bf0
        if extended_f0[i + shift] != 0:
            tmp_f0 = extended_f0[i + shift]
            count = 0
            shifted_origin = i + shift
        else:
            count = count + 1
        if count == threshold:
            break
    return extended_f0, shifted_origin


def harvest_get_multichannel_f0(f0, boundary_list):
    multichannel_f0 = np.zeros((int(len(boundary_list) / 2), len(f0)))
    for i in range(1, int(len(boundary_list) / 2) + 1):
        sl = boundary_list[(2 * i) - 2]
        el = boundary_list[(2 * i) - 1] + 1
        multichannel_f0[i - 1, sl:el] = f0[sl:el]
    return multichannel_f0


def harvest_get_boundary_list(f0):
    vuv = f0.copy()
    vuv[vuv != 0] = 1.
    vuv[0] = 0
    vuv[-1] = 0
    diff_vuv = vuv[1:] - vuv[:-1]
    boundary_list = np.where(diff_vuv != 0)[0]
    boundary_list[::2] = boundary_list[::2] + 1
    return boundary_list


def harvest_fix_step_4(f0_step3, threshold):
    f0_step4 = f0_step3.copy()
    boundary_list = harvest_get_boundary_list(f0_step3)

    for i in range(1, int(len(boundary_list) / 2.)):
        distance = boundary_list[(2 * i)] - boundary_list[(2 * i) - 1] - 1
        if distance >= threshold:
            continue
        boundary0 = f0_step3[boundary_list[(2 * i) - 1]] + 1
        boundary1 = f0_step3[boundary_list[(2 * i)]] - 1
        coefficient = (boundary1 - boundary0) / float((distance + 1))
        count = 1
        st = boundary_list[(2 * i) - 1] + 1
        ed = boundary_list[(2 * i)]
        for j in range(st, ed):
            f0_step4[j] = boundary0 + coefficient * count
            count = count + 1
    return f0_step4


def harvest_fix_f0_contour(f0_candidates, f0_scores):
    f0_base = harvest_search_f0_base(f0_candidates, f0_scores)
    f0_step1 = harvest_fix_step_1(f0_base, 0.008) # optimized?
    f0_step2 = harvest_fix_step_2(f0_step1, 6) # optimized?
    f0_step3 = harvest_fix_step_3(f0_step2, f0_candidates, 0.18, f0_scores) # optimized?
    f0 = harvest_fix_step_4(f0_step3, 9) # optimized
    vuv = f0.copy()
    vuv[vuv != 0] = 1.
    return f0, vuv


def harvest_filter_f0_contour(f0, st, ed, b, a):
    smoothed_f0 = f0.copy()
    smoothed_f0[:st] = smoothed_f0[st]
    smoothed_f0[ed + 1:] = smoothed_f0[ed]
    aaa = sg.lfilter(b, a, smoothed_f0)
    bbb = sg.lfilter(b, a, aaa[::-1])
    smoothed_f0 = bbb[::-1].copy()
    smoothed_f0[:st] = 0.
    smoothed_f0[ed + 1:] = 0.
    return smoothed_f0


def harvest_smooth_f0_contour(f0):
    b = np.array([0.0078202080334971724, 0.015640416066994345, 0.0078202080334971724])
    a = np.array([1.0, -1.7347257688092754, 0.76600660094326412])
    smoothed_f0 = np.concatenate([np.zeros(300,), f0, np.zeros(300,)])
    boundary_list = harvest_get_boundary_list(smoothed_f0)
    multichannel_f0 = harvest_get_multichannel_f0(smoothed_f0, boundary_list)
    for i in range(1, int(len(boundary_list) / 2) + 1):
        tmp_f0_contour = harvest_filter_f0_contour(multichannel_f0[i - 1, :],
                boundary_list[(2 * i) - 2], boundary_list[(2 * i) - 1], b, a)
        st = boundary_list[(2 * i) - 2]
        ed = boundary_list[(2 * i) - 1] + 1
        smoothed_f0[st:ed] = tmp_f0_contour[st:ed]
    smoothed_f0 = smoothed_f0[300:-300]
    return smoothed_f0


def _world_get_temporal_positions(x_len, fs):
    frame_period = 5
    basic_frame_period = 1
    basic_temporal_positions = np.arange(0, x_len / float(fs), basic_frame_period / float(1000))
    temporal_positions = np.arange(0,
            x_len / float(fs),
            frame_period / float(1000))
    return basic_temporal_positions, temporal_positions


def harvest(x, fs):
    f0_floor = 71
    f0_ceil = 800
    target_fs = 8000
    channels_in_octave = 40.
    basic_temporal_positions, temporal_positions = _world_get_temporal_positions(len(x), fs)
    adjusted_f0_floor = f0_floor * 0.9
    adjusted_f0_ceil = f0_ceil * 1.1
    boundary_f0_list = np.arange(1, np.ceil(np.log2(adjusted_f0_ceil / adjusted_f0_floor) * channels_in_octave) + 1) / float(channels_in_octave)
    boundary_f0_list = adjusted_f0_floor * 2.0 ** boundary_f0_list
    y, actual_fs = harvest_get_downsampled_signal(x, fs, target_fs)
    fft_size = 2. ** np.ceil(np.log2(len(y) + np.round(fs / f0_floor * 4) + 1))
    y_spectrum = np.fft.fft(y, int(fft_size))
    raw_f0_candidates = harvest_get_raw_f0_candidates(
        len(basic_temporal_positions),
        boundary_f0_list, len(y), basic_temporal_positions, actual_fs,
        y_spectrum, f0_floor, f0_ceil)

    f0_candidates, number_of_candidates = harvest_detect_official_f0_candidates(raw_f0_candidates)
    f0_candidates = harvest_overlap_f0_candidates(f0_candidates, number_of_candidates)
    f0_candidates, f0_scores = harvest_refine_candidates(y, actual_fs,
            basic_temporal_positions, f0_candidates, f0_floor, f0_ceil)

    f0_candidates, f0_scores = harvest_remove_unreliable_candidates(f0_candidates, f0_scores)

    connected_f0, vuv = harvest_fix_f0_contour(f0_candidates, f0_scores)
    smoothed_f0 = harvest_smooth_f0_contour(connected_f0)
    idx = np.minimum(len(smoothed_f0) - 1, np.round(temporal_positions * 1000)).astype("int32")
    f0 = smoothed_f0[idx]
    vuv = vuv[idx]
    f0_candidates = f0_candidates
    return temporal_positions, f0, vuv, f0_candidates

def run_world_example():
    fs, d = fetch_sample_speech_tapestry()
    d = d.astype("float32") / 2 ** 15
    temporal_positions_h, f0_h, vuv_h, f0_candidates_h = harvest(d, fs)
    temporal_positions_ct, spectrogram_ct, fs_ct = cheaptrick(d, fs,
            temporal_positions_h, f0_h, vuv_h)
    temporal_positions_d4c, f0_d4c, vuv_d4c, aper_d4c, coarse_aper_d4c = d4c(d, fs,
            temporal_positions_h, f0_h, vuv_h)
    #y = world_synthesis(f0_d4c, vuv_d4c, aper_d4c, spectrogram_ct, fs_ct)
    y = world_synthesis(f0_d4c, vuv_d4c, coarse_aper_d4c, spectrogram_ct, fs_ct)
    wavfile.write("out.wav", fs, soundsc(y))


def cheaptrick_get_windowed_waveform(x, fs, current_f0, current_position):
    half_window_length = np.round(1.5 * fs / float(current_f0))
    base_index = np.arange(-half_window_length, half_window_length + 1)
    index = np.round(current_position * fs + 0.001) + base_index + 1
    safe_index = np.minimum(len(x), np.maximum(1, np.round(index))).astype("int32")
    safe_index = safe_index - 1
    segment = x[safe_index]
    time_axis = base_index / float(fs) / 1.5
    window1 = 0.5 * np.cos(np.pi * time_axis * float(current_f0)) + 0.5
    window1 = window1 / np.sqrt(np.sum(window1 ** 2))
    waveform = segment * window1 - window1 * np.mean(segment * window1) / np.mean(window1)
    return waveform


def cheaptrick_get_power_spectrum(waveform, fs, fft_size, f0):
    power_spectrum = np.abs(np.fft.fft(waveform, fft_size)) ** 2
    frequency_axis = np.arange(fft_size) / float(fft_size) * float(fs)
    ind = frequency_axis < (f0 + fs / fft_size)
    low_frequency_axis = frequency_axis[ind]
    low_frequency_replica = interp1d(f0 - low_frequency_axis,
            power_spectrum[ind], kind="linear",
            fill_value="extrapolate")(low_frequency_axis)
    p1 = low_frequency_replica[(frequency_axis < f0)[:len(low_frequency_replica)]]
    p2 = power_spectrum[(frequency_axis < f0)[:len(power_spectrum)]]
    power_spectrum[frequency_axis < f0] = p1 + p2
    lb1 = int(fft_size / 2) + 1
    lb2 = 1
    ub2 = int(fft_size / 2)
    power_spectrum[lb1:] = power_spectrum[lb2:ub2][::-1]
    return power_spectrum


def cheaptrick_linear_smoothing(power_spectrum, f0, fs, fft_size):
    double_frequency_axis = np.arange(2 * fft_size) / float(fft_size ) * fs - fs
    double_spectrum = np.concatenate([power_spectrum, power_spectrum])

    double_segment = np.cumsum(double_spectrum * (fs / float(fft_size)))
    center_frequency = np.arange(int(fft_size / 2) + 1) / float(fft_size ) * fs
    low_levels = cheaptrick_interp1h(double_frequency_axis + fs / float(fft_size) / 2.,
            double_segment, center_frequency - f0 / 3.)
    high_levels = cheaptrick_interp1h(double_frequency_axis + fs / float(fft_size) / 2.,
            double_segment, center_frequency + f0 / 3.)
    smoothed_spectrum = (high_levels - low_levels) * 1.5 / f0
    return smoothed_spectrum


def cheaptrick_interp1h(x, y, xi):
    delta_x = float(x[1] - x[0])
    xi = np.maximum(x[0], np.minimum(x[-1], xi))
    xi_base = (np.floor((xi - x[0]) / delta_x)).astype("int32")
    xi_fraction = (xi - x[0]) / delta_x - xi_base
    delta_y = np.zeros_like(y)
    delta_y[:-1] = y[1:] - y[:-1]
    yi = y[xi_base] + delta_y[xi_base] * xi_fraction
    return yi


def cheaptrick_smoothing_with_recovery(smoothed_spectrum, f0, fs, fft_size, q1):
    quefrency_axis = np.arange(fft_size) / float(fs)
    # 0 is NaN
    smoothing_lifter = np.sin(np.pi * f0 * quefrency_axis) / (np.pi * f0 * quefrency_axis)
    p = smoothing_lifter[1:int(fft_size / 2)][::-1].copy()
    smoothing_lifter[int(fft_size / 2) + 1:] = p
    smoothing_lifter[0] = 1.
    compensation_lifter = (1 - 2. * q1) + 2. * q1 * np.cos(2 * np.pi * quefrency_axis * f0)
    p = compensation_lifter[1:int(fft_size / 2)][::-1].copy()
    compensation_lifter[int(fft_size / 2) + 1:] = p
    tandem_cepstrum = np.fft.fft(np.log(smoothed_spectrum))
    tmp_spectral_envelope = np.exp(np.real(np.fft.ifft(tandem_cepstrum * smoothing_lifter * compensation_lifter)))
    spectral_envelope = tmp_spectral_envelope[:int(fft_size / 2) + 1]
    return spectral_envelope


def cheaptrick_estimate_one_slice(x, fs, current_f0,
    current_position, fft_size, q1):
    waveform = cheaptrick_get_windowed_waveform(x, fs, current_f0,
        current_position)
    power_spectrum = cheaptrick_get_power_spectrum(waveform, fs, fft_size,
            current_f0)
    smoothed_spectrum = cheaptrick_linear_smoothing(power_spectrum, current_f0,
            fs, fft_size)
    comb_spectrum = np.concatenate([smoothed_spectrum, smoothed_spectrum[1:-1][::-1]])
    spectral_envelope = cheaptrick_smoothing_with_recovery(comb_spectrum,
            current_f0, fs, fft_size, q1)
    return spectral_envelope


def cheaptrick(x, fs, temporal_positions, f0_sequence,
        vuv, fftlen="auto", q1=-0.15):
    f0_sequence = f0_sequence.copy()
    f0_low_limit = 71
    default_f0 = 500
    if fftlen == "auto":
        fftlen = int(2 ** np.ceil(np.log2(3. * float(fs) / f0_low_limit + 1)))
    #raise ValueError("Only fftlen auto currently supported")
    fft_size = fftlen
    f0_low_limit = fs * 3.0 / (fft_size - 3.0)
    f0_sequence[vuv == 0] = default_f0
    spectrogram = np.zeros((int(fft_size / 2.) + 1, len(f0_sequence)))
    for i in range(len(f0_sequence)):
        if f0_sequence[i] < f0_low_limit:
            f0_sequence[i] = default_f0
        spectrogram[:, i] = cheaptrick_estimate_one_slice(x, fs, f0_sequence[i],
                temporal_positions[i], fft_size, q1)
    return temporal_positions, spectrogram.T, fs


def d4c_love_train(x, fs, current_f0, current_position, threshold):
    vuv = 0
    if current_f0 == 0:
        return vuv
    lowest_f0 = 40
    current_f0 = max([current_f0, lowest_f0])
    fft_size = int(2 ** np.ceil(np.log2(3. * fs / lowest_f0 + 1)))
    boundary0 = int(np.ceil(100 / (float(fs) / fft_size)))
    boundary1 = int(np.ceil(4000 / (float(fs) / fft_size)))
    boundary2 = int(np.ceil(7900 / (float(fs) / fft_size)))

    waveform = d4c_get_windowed_waveform(x, fs, current_f0, current_position,
            1.5, 2)
    power_spectrum = np.abs(np.fft.fft(waveform, int(fft_size)) ** 2)
    power_spectrum[0:boundary0 + 1] = 0.
    cumulative_spectrum = np.cumsum(power_spectrum)
    if (cumulative_spectrum[boundary1] / cumulative_spectrum[boundary2]) > threshold:
        vuv = 1
    return vuv


def d4c_get_windowed_waveform(x, fs, current_f0, current_position, half_length,
        window_type):
    half_window_length = int(np.round(half_length * fs / current_f0))
    base_index = np.arange(-half_window_length, half_window_length + 1)
    index = np.round(current_position * fs + 0.001) + base_index + 1
    safe_index = np.minimum(len(x), np.maximum(1, np.round(index))).astype("int32") - 1

    segment = x[safe_index]
    time_axis = base_index / float(fs) / float(half_length)
    if window_type == 1:
        window1 = 0.5 * np.cos(np.pi * time_axis * current_f0) + 0.5
    elif window_type == 2:
        window1 = 0.08 * np.cos(np.pi * time_axis * current_f0 * 2)
        window1 += 0.5 * np.cos(np.pi * time_axis * current_f0) + 0.42
    else:
        raise ValueError("Unknown window type")
    waveform = segment * window1 - window1 * np.mean(segment * window1) / np.mean(window1)
    return waveform


def d4c_get_static_centroid(x, fs, current_f0, current_position, fft_size):
    waveform1 = d4c_get_windowed_waveform(x, fs, current_f0,
        current_position + 1. / current_f0 / 4., 2, 2)
    waveform2 = d4c_get_windowed_waveform(x, fs, current_f0,
        current_position - 1. / current_f0 / 4., 2, 2)
    centroid1 = d4c_get_centroid(waveform1, fft_size)
    centroid2 = d4c_get_centroid(waveform2, fft_size)
    centroid = d4c_dc_correction(centroid1 + centroid2, fs, fft_size,
            current_f0)
    return centroid


def d4c_get_centroid(x, fft_size):
    fft_size = int(fft_size)
    time_axis = np.arange(1, len(x) + 1)
    x = x.copy()
    x = x / np.sqrt(np.sum(x ** 2))

    spectrum = np.fft.fft(x, fft_size)
    weighted_spectrum = np.fft.fft(-x * 1j * time_axis, fft_size)
    centroid = -(weighted_spectrum.imag) * spectrum.real + spectrum.imag * weighted_spectrum.real
    return centroid


def d4c_dc_correction(signal, fs, fft_size, f0):
    fft_size = int(fft_size)
    frequency_axis = np.arange(fft_size) / fft_size * fs
    low_frequency_axis = frequency_axis[frequency_axis < f0 + fs / fft_size]
    low_frequency_replica = interp1d(f0 - low_frequency_axis,
            signal[frequency_axis < f0 + fs / fft_size],
            kind="linear",
            fill_value="extrapolate")(low_frequency_axis)
    idx = frequency_axis < f0
    signal[idx] = low_frequency_replica[idx[:len(low_frequency_replica)]] + signal[idx]
    signal[int(fft_size / 2.) + 1:] = signal[1 : int(fft_size / 2.)][::-1]
    return signal


def d4c_linear_smoothing(group_delay, fs, fft_size, width):
    double_frequency_axis = np.arange(2 * fft_size) / float(fft_size ) * fs - fs
    double_spectrum = np.concatenate([group_delay, group_delay])

    double_segment = np.cumsum(double_spectrum * (fs / float(fft_size)))
    center_frequency = np.arange(int(fft_size / 2) + 1) / float(fft_size ) * fs
    low_levels = cheaptrick_interp1h(double_frequency_axis + fs / float(fft_size) / 2.,
            double_segment, center_frequency - width / 2.)
    high_levels = cheaptrick_interp1h(double_frequency_axis + fs / float(fft_size) / 2.,
            double_segment, center_frequency + width / 2.)
    smoothed_spectrum = (high_levels - low_levels) / width
    return smoothed_spectrum


def d4c_get_smoothed_power_spectrum(waveform, fs, f0, fft_size):
    power_spectrum = np.abs(np.fft.fft(waveform, int(fft_size))) ** 2
    spectral_envelope = d4c_dc_correction(power_spectrum, fs, fft_size, f0)
    spectral_envelope = d4c_linear_smoothing(spectral_envelope, fs, fft_size, f0)
    spectral_envelope = np.concatenate([spectral_envelope,
        spectral_envelope[1:-1][::-1]])
    return spectral_envelope


def d4c_get_static_group_delay(static_centroid, smoothed_power_spectrum, fs, f0,
        fft_size):
    group_delay = static_centroid / smoothed_power_spectrum
    group_delay = d4c_linear_smoothing(group_delay, fs, fft_size, f0 / 2.)
    group_delay = np.concatenate([group_delay, group_delay[1:-1][::-1]])
    smoothed_group_delay = d4c_linear_smoothing(group_delay, fs, fft_size, f0)
    group_delay = group_delay[:int(fft_size / 2) + 1] - smoothed_group_delay
    group_delay = np.concatenate([group_delay, group_delay[1:-1][::-1]])
    return group_delay


def d4c_get_coarse_aperiodicity(group_delay, fs, fft_size,
        frequency_interval, number_of_aperiodicities, window1):
    boundary = np.round(fft_size / len(window1) * 8)
    half_window_length = np.floor(len(window1) / 2)
    coarse_aperiodicity = np.zeros((number_of_aperiodicities, 1))
    for i in range(1, number_of_aperiodicities + 1):
        center = np.floor(frequency_interval * i / (fs / float(fft_size)))
        segment = group_delay[int(center - half_window_length):int(center + half_window_length + 1)] * window1
        power_spectrum = np.abs(np.fft.fft(segment, int(fft_size))) ** 2
        cumulative_power_spectrum = np.cumsum(np.sort(power_spectrum[:int(fft_size / 2) + 1]))
        coarse_aperiodicity[i - 1] = -10 * np.log10(cumulative_power_spectrum[int(fft_size / 2 - boundary) - 1] / cumulative_power_spectrum[-1])
    return coarse_aperiodicity


def d4c_estimate_one_slice(x, fs, current_f0, frequency_interval,
        current_position, fft_size, number_of_aperiodicities, window1):
    if current_f0 == 0:
        coarse_aperiodicity = np.zeros((number_of_aperiodicities, 1))
        return coarse_aperiodicity

    static_centroid = d4c_get_static_centroid(x, fs, current_f0,
        current_position, fft_size)
    waveform = d4c_get_windowed_waveform(x, fs, current_f0, current_position,
            2, 1)
    smoothed_power_spectrum = d4c_get_smoothed_power_spectrum(waveform, fs,
            current_f0, fft_size)
    static_group_delay = d4c_get_static_group_delay(static_centroid,
            smoothed_power_spectrum, fs, current_f0, fft_size)
    coarse_aperiodicity = d4c_get_coarse_aperiodicity(static_group_delay,
            fs, fft_size, frequency_interval, number_of_aperiodicities, window1)
    return coarse_aperiodicity


def d4c(x, fs, temporal_positions_h, f0_h, vuv_h, threshold="default",
        fft_size="auto"):
    f0_low_limit = 47
    if fft_size == "auto":
        fft_size = 2 ** np.ceil(np.log2(4. * fs / f0_low_limit + 1.))
    else:
        raise ValueError("Only fft_size auto currently supported")
    f0_low_limit_for_spectrum = 71
    fft_size_for_spectrum = 2 ** np.ceil(np.log2(3 * fs / f0_low_limit_for_spectrum + 1.))
    threshold = 0.85
    upper_limit = 15000
    frequency_interval = 3000
    f0 = f0_h.copy()
    temporal_positions = temporal_positions_h.copy()
    f0[vuv_h == 0] = 0.

    number_of_aperiodicities = int(np.floor(np.min([upper_limit, fs / 2. - frequency_interval]) / float(frequency_interval)))
    window_length = np.floor(frequency_interval / (fs / float(fft_size))) * 2 + 1
    window1 =  harvest_nuttall(window_length)
    aperiodicity = np.zeros((int(fft_size_for_spectrum / 2) + 1, len(f0)))
    coarse_ap = np.zeros((1, len(f0)))

    frequency_axis = np.arange(int(fft_size_for_spectrum / 2) + 1) * float(fs) / fft_size_for_spectrum
    coarse_axis = np.arange(number_of_aperiodicities + 2) * frequency_interval
    coarse_axis[-1] = fs / 2.

    for i in range(len(f0)):
        r = d4c_love_train(x, fs, f0[i], temporal_positions_h[i], threshold)
        if r == 0:
            aperiodicity[:, i] = 1 - 0.000000000001
            continue
        current_f0 = max([f0_low_limit, f0[i]])
        coarse_aperiodicity = d4c_estimate_one_slice(x, fs, current_f0,
            frequency_interval, temporal_positions[i], fft_size,
            number_of_aperiodicities, window1)
        coarse_ap[0, i] = coarse_aperiodicity.ravel()[0]
        coarse_aperiodicity = np.maximum(0, coarse_aperiodicity - (current_f0 - 100) * 2. / 100.)
        piece = np.concatenate([[-60], -coarse_aperiodicity.ravel(), [-0.000000000001]])
        part = interp1d(coarse_axis, piece, kind="linear")(frequency_axis) / 20.
        aperiodicity[:, i] = 10 ** part
    return temporal_positions_h, f0_h, vuv_h, aperiodicity.T, coarse_ap.squeeze()


def world_synthesis_time_base_generation(temporal_positions, f0, fs, vuv,
        time_axis, default_f0):
    f0_interpolated_raw = interp1d(temporal_positions, f0, kind="linear",
            fill_value="extrapolate")(time_axis)
    vuv_interpolated = interp1d(temporal_positions, vuv, kind="linear",
            fill_value="extrapolate")(time_axis)
    vuv_interpolated = vuv_interpolated > 0.5
    f0_interpolated = f0_interpolated_raw * vuv_interpolated.astype("float32")
    f0_interpolated[f0_interpolated == 0] = f0_interpolated[f0_interpolated == 0] + default_f0
    total_phase = np.cumsum(2 * np.pi * f0_interpolated / float(fs))

    core = np.mod(total_phase, 2 * np.pi)
    core = np.abs(core[1:] - core[:-1])
    # account for diff, avoid deprecation warning with [:-1]
    pulse_locations = time_axis[:-1][core > (np.pi / 2.)]
    pulse_locations_index = np.round(pulse_locations * fs).astype("int32")
    return pulse_locations, pulse_locations_index, vuv_interpolated


def world_synthesis_get_spectral_parameters(temporal_positions,
        temporal_position_index, spectrogram, amplitude_periodic,
        amplitude_random, pulse_locations):
    floor_index = int(np.floor(temporal_position_index) - 1)
    assert floor_index >= 0
    ceil_index = int(np.ceil(temporal_position_index) - 1)
    t1 = temporal_positions[floor_index]
    t2 = temporal_positions[ceil_index]

    if t1 == t2:
        spectrum_slice = spectrogram[:, floor_index]
        periodic_slice = amplitude_periodic[:, floor_index]
        aperiodic_slice = amplitude_random[:, floor_index]
    else:
        cs = np.concatenate([spectrogram[:, floor_index][None],
            spectrogram[:, ceil_index][None]], axis=0)
        mmm = max([t1, min([t2, pulse_locations])])
        spectrum_slice = interp1d(np.array([t1, t2]), cs,
            kind="linear", axis=0)(mmm.copy())
        cp = np.concatenate([amplitude_periodic[:, floor_index][None],
            amplitude_periodic[:, ceil_index][None]], axis=0)
        periodic_slice = interp1d(np.array([t1, t2]), cp,
            kind="linear", axis=0)(mmm.copy())
        ca = np.concatenate([amplitude_random[:, floor_index][None],
            amplitude_random[:, ceil_index][None]], axis=0)
        aperiodic_slice = interp1d(np.array([t1, t2]), ca,
            kind="linear", axis=0)(mmm.copy())
    return spectrum_slice, periodic_slice, aperiodic_slice


def world_synthesis(f0_d4c, vuv_d4c, aperiodicity_d4c,
        spectrogram_ct, fs_ct, random_seed=1999):

    # swap 0 and 1 axis
    spectrogram_ct = spectrogram_ct.T
    fs = fs_ct
    # coarse -> fine aper
    if len(aperiodicity_d4c.shape) == 1 or aperiodicity_d4c.shape[1] == 1:
        print("Coarse aperiodicity detected - interpolating to full size")
        aper = np.zeros_like(spectrogram_ct)
        if len(aperiodicity_d4c.shape) == 1:
            aperiodicity_d4c = aperiodicity_d4c[None, :]
        else:
            aperiodicity_d4c = aperiodicity_d4c.T
        coarse_aper_d4c = aperiodicity_d4c
        frequency_interval = 3000
        upper_limit = 15000
        number_of_aperiodicities = int(np.floor(np.min([upper_limit, fs / 2. - frequency_interval]) / float(frequency_interval)))
        coarse_axis = np.arange(number_of_aperiodicities + 2) * frequency_interval
        coarse_axis[-1] = fs / 2.
        f0_low_limit_for_spectrum = 71
        fft_size_for_spectrum = 2 ** np.ceil(np.log2(3 * fs / f0_low_limit_for_spectrum + 1.))

        frequency_axis = np.arange(int(fft_size_for_spectrum / 2) + 1) * float(fs) / fft_size_for_spectrum

        for i in range(len(f0_d4c)):
            ca = coarse_aper_d4c[0, i]
            cf = f0_d4c[i]
            coarse_aperiodicity = np.maximum(0, ca - (cf - 100) * 2. / 100.)
            piece = np.concatenate([[-60], -ca.ravel(), [-0.000000000001]])
            part = interp1d(coarse_axis, piece, kind="linear")(frequency_axis) / 20.
            aper[:, i] = 10 ** part
        aperiodicity_d4c = aper
    else:
        aperiodicity_d4c = aperiodicity_d4c.T

    default_f0 = 500.
    random_state = np.random.RandomState(1999)
    spectrogram = spectrogram_ct
    aperiodicity = aperiodicity_d4c
    # max 30s, if greater than thrown an error
    max_len = 5000000
    _, temporal_positions = _world_get_temporal_positions(max_len, fs)
    temporal_positions = temporal_positions[:spectrogram.shape[1]]
    #temporal_positions = temporal_positions_d4c
    #from IPython import embed; embed()
    #raise ValueError()
    vuv = vuv_d4c
    f0 = f0_d4c

    time_axis = np.arange(temporal_positions[0], temporal_positions[-1],
            1. / fs)
    y = 0. * time_axis
    r = world_synthesis_time_base_generation(temporal_positions, f0, fs, vuv,
            time_axis, default_f0)
    pulse_locations, pulse_locations_index, interpolated_vuv = r
    fft_size = int((len(spectrogram) - 1) * 2)
    base_index = np.arange(-fft_size / 2, fft_size / 2) + 1
    y_length = len(y)
    tmp_complex_cepstrum = np.zeros((fft_size,), dtype=np.complex128)
    latter_index = np.arange(int(fft_size / 2) + 1, fft_size + 1) - 1

    temporal_position_index = interp1d(temporal_positions, np.arange(1, len(temporal_positions) + 1), kind="linear", fill_value="extrapolate")(pulse_locations)
    # temporal_postion_index = np.maximum(1, np.minimum(len(temporal_positions),
        # temporal_position_index)) - 1

    amplitude_aperiodic = aperiodicity ** 2
    amplitude_periodic = np.maximum(0.001, (1. - amplitude_aperiodic))

    for i in range(len(pulse_locations_index)):
        spectrum_slice, periodic_slice, aperiodic_slice = world_synthesis_get_spectral_parameters(
            temporal_positions, temporal_position_index[i], spectrogram,
            amplitude_periodic, amplitude_aperiodic, pulse_locations[i])
        idx = min(len(pulse_locations_index), i + 2) - 1
        noise_size = pulse_locations_index[idx] - pulse_locations_index[i]
        output_buffer_index = np.maximum(1, np.minimum(y_length, pulse_locations_index[i] + 1 + base_index)).astype("int32") - 1

        if interpolated_vuv[pulse_locations_index[i]] >= 0.5:
            tmp_periodic_spectrum = spectrum_slice * periodic_slice
            # eps in matlab/octave
            tmp_periodic_spectrum[tmp_periodic_spectrum == 0] = 2.2204E-16
            periodic_spectrum = np.concatenate([tmp_periodic_spectrum,
                tmp_periodic_spectrum[1:-1][::-1]])
            tmp_cepstrum = np.real(np.fft.fft(np.log(np.abs(periodic_spectrum)) / 2.))
            tmp_complex_cepstrum[latter_index] = tmp_cepstrum[latter_index] * 2
            tmp_complex_cepstrum[0] = tmp_cepstrum[0]

            response = np.fft.fftshift(np.real(np.fft.ifft(np.exp(np.fft.ifft(
                tmp_complex_cepstrum)))))
            y[output_buffer_index] += response * np.sqrt(
                   max([1, noise_size]))
            tmp_aperiodic_spectrum = spectrum_slice * aperiodic_slice
        else:
            tmp_aperiodic_spectrum = spectrum_slice

        tmp_aperiodic_spectrum[tmp_aperiodic_spectrum == 0] = 2.2204E-16
        aperiodic_spectrum = np.concatenate([tmp_aperiodic_spectrum,
            tmp_aperiodic_spectrum[1:-1][::-1]])
        tmp_cepstrum = np.real(np.fft.fft(np.log(np.abs(aperiodic_spectrum)) / 2.))
        tmp_complex_cepstrum[latter_index] = tmp_cepstrum[latter_index] * 2
        tmp_complex_cepstrum[0] = tmp_cepstrum[0]
        rc = np.fft.ifft(tmp_complex_cepstrum)
        erc = np.exp(rc)
        response = np.fft.fftshift(np.real(np.fft.ifft(erc)))
        noise_input = random_state.randn(max([3, noise_size]),)

        y[output_buffer_index] = y[output_buffer_index] + fftfilt(noise_input - np.mean(noise_input), response)
    return y


def run_world_mgc_example():
    fs, d = fetch_sample_speech_tapestry()
    d = d.astype("float32") / 2 ** 15

    # harcoded for 16k from
    # https://github.com/CSTR-Edinburgh/merlin/blob/master/misc/scripts/vocoder/world/extract_features_for_merlin.sh
    mgc_alpha = 0.58
    #mgc_order = 59
    mgc_order = 59
    # this is actually just mcep
    mgc_gamma = 0.0

    #from sklearn.externals import joblib
    #mem = joblib.Memory("/tmp")
    #mem.clear()

    def enc():
        temporal_positions_h, f0_h, vuv_h, f0_candidates_h = harvest(d, fs)
        temporal_positions_ct, spectrogram_ct, fs_ct = cheaptrick(d, fs,
                temporal_positions_h, f0_h, vuv_h)
        temporal_positions_d4c, f0_d4c, vuv_d4c, aper_d4c, coarse_aper_d4c = d4c(d, fs,
                temporal_positions_h, f0_h, vuv_h)

        mgc_arr = sp2mgc(spectrogram_ct, mgc_order, mgc_alpha, mgc_gamma,
                verbose=True)
        return mgc_arr, spectrogram_ct, f0_d4c, vuv_d4c, coarse_aper_d4c


    mgc_arr, spectrogram_ct, f0_d4c, vuv_d4c, coarse_aper_d4c = enc()
    sp_r = mgc2sp(mgc_arr, mgc_alpha, mgc_gamma, fs=fs, verbose=True)

    """
    import matplotlib.pyplot as plt
    plt.imshow(20 * np.log10(sp_r))
    plt.figure()
    plt.imshow(20 * np.log10(spectrogram_ct))
    plt.show()
    raise ValueError()
    """

    y = world_synthesis(f0_d4c, vuv_d4c, coarse_aper_d4c, sp_r, fs)
    #y = world_synthesis(f0_d4c, vuv_d4c, aper_d4c, sp_r, fs)
    wavfile.write("out_mgc.wav", fs, soundsc(y))
