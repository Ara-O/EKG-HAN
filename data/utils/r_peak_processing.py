import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def pan_tompkins_detector(ecg_signal, fs):
    lowcut, highcut = 5.0, 15.0
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)
    diff_ecg = np.diff(filtered_ecg)
    squared_ecg = diff_ecg ** 2
    window_size = int(0.150 * fs)
    mwa_ecg = np.convolve(squared_ecg, np.ones(window_size) / window_size, mode='same')
    peaks, _ = find_peaks(mwa_ecg, distance=int(0.6 * fs))
    return peaks

def multi_lead_fusion(detected_peaks, fs, fusion_window=0.1, min_leads=None):
    n_leads = len(detected_peaks)
    if min_leads is None:
        min_leads = int(np.ceil(n_leads / 2))

    all_peaks = [(p, lead) for lead, peaks in enumerate(detected_peaks) for p in peaks]
    all_peaks.sort(key=lambda x: x[0])

    fused_peaks = []
    i = 0

    while i < len(all_peaks):
        cluster = [all_peaks[i]]
        i += 1

        while i < len(all_peaks) and all_peaks[i][0] - cluster[-1][0] <= fusion_window * fs:
            cluster.append(all_peaks[i])
            i += 1

        unique_leads = {lead for _, lead in cluster}
        if len(unique_leads) >= min_leads:
            fused_peak = int(np.median([p for (p, _) in cluster]))
            fused_peaks.append(fused_peak)

    return np.array(sorted(fused_peaks))


def detect_r_peaks(ecg_signals, fs):
    detected_peaks = []
    for lead in ecg_signals:
        peaks = pan_tompkins_detector(lead, fs)
        detected_peaks.append(peaks)

    fused_r_peaks = multi_lead_fusion(detected_peaks, fs, fusion_window=0.1, min_leads=6)
    return fused_r_peaks


def extract_segments_around_peaks(signal, r_peaks, pre_samples, post_samples):
    segments = []

    for peak in r_peaks:
        start = max(0, peak - pre_samples)
        end = min(len(signal), peak + post_samples)

        if end - start == pre_samples + post_samples:
            segment = signal[start:end]
            segments.append(segment)

    return segments

def extract_beats_multi_lead(ecg_signal, fs, pre_samples, post_samples):
    ecg_signal = np.array(ecg_signal)

    r_peaks = detect_r_peaks(ecg_signal, fs)

    if len(r_peaks) == 0:
        return None, None

    rr_intervals = np.diff(r_peaks) / fs
    if len(rr_intervals) > 0:
        mean_rr = np.mean(rr_intervals)
        rr_intervals = np.insert(rr_intervals, 0, mean_rr)
    else:
        rr_intervals = np.array([0.0])

    valid_peak_mask = []
    for peak in r_peaks:
        start = max(0, peak - pre_samples)
        end = min(ecg_signal.shape[1], peak + post_samples)
        valid_peak_mask.append(end - start == pre_samples + post_samples)

    valid_peaks = r_peaks[valid_peak_mask]
    rr_intervals = rr_intervals[valid_peak_mask]

    if len(valid_peaks) == 0:
        return None, None

    all_lead_segments = []
    for lead_idx, lead_signal in enumerate(ecg_signal):
        segments = extract_segments_around_peaks(lead_signal, valid_peaks, pre_samples, post_samples)
        all_lead_segments.append(segments)


    # Stack segments: (n_beats, segment_length, n_leads)
    beats_arr = np.stack([
        np.stack(segments, axis=0)
        for segments in all_lead_segments
    ], axis=-1)

    return beats_arr.astype(np.float32), rr_intervals.astype(np.float32)


def extract_rr_beats_multi_lead(ecg, fs, min_rr_ms=300, max_rr_ms=1500, min_beats=2):
    r_peaks = detect_r_peaks(ecg, fs)        
    if len(r_peaks) < min_beats:
        return None, None

    rr_samples = np.diff(r_peaks)
    rr_ms = (rr_samples / fs) * 1000.0
    valid = (rr_ms >= min_rr_ms) & (rr_ms <= max_rr_ms)
    
    if valid.sum() == 0:
        return None, None
    
    # RR in seconds, prepend mean for the first beat
    rr_intervals = rr_samples / fs
    mean_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else 0.0
    rr_intervals = np.insert(rr_intervals, 0, mean_rr) 

    beats = []
    kept_rr = []

    for i in range(len(r_peaks) - 1):
        if not valid[i]:
            continue
        start = r_peaks[i]
        end   = r_peaks[i+1]
        seg = ecg[:, start:end].T
        if np.isnan(seg).any() or seg.shape[0] == 0:
            continue
        beats.append(seg.astype(np.float32))
        kept_rr.append(rr_intervals[i])

    if len(beats) == 0:
        return None, None

    kept_rr = np.array(kept_rr, dtype=np.float32)
    return beats, kept_rr