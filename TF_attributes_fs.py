import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.signal import welch, stft, find_peaks
from scipy.stats import kurtosis, skew, entropy
import pywt
from scipy.signal import hilbert
import nolds

EPS = 1e-10


def safe_calc(func, default=np.nan):
    def wrapper(sig):
        try:
            res = func(sig)
            if np.isinf(res):
                return default
            return res
        except Exception:
            return default
    return wrapper


def compute_attributes(fs=20000):
    """
    81 time-frequency statistical features for vibration signal analysis in fault diagnosis.
    """
    attributes = {
        'mean': lambda sig: np.mean(sig),
        'variance': lambda sig: np.var(sig),
        'std_dev': lambda sig: np.std(sig),
        'rms': lambda sig: np.sqrt(np.mean(sig ** 2)),
        'max_value': lambda sig: np.max(sig),
        'min_value': lambda sig: np.min(sig),
        'kurtosis': lambda sig: kurtosis(sig),
        'skewness': lambda sig: skew(sig),
        'absolute_mean': lambda sig: np.mean(np.abs(sig)),
        'peak_value': lambda sig: np.max(np.abs(sig)),
        'peak_to_peak': lambda sig: np.ptp(sig),
        'waveform_index': lambda sig: np.sqrt(np.mean(sig ** 2)) / np.mean(np.abs(sig)) if np.mean(
            np.abs(sig)) > 0 else 0,
        'impulse_factor': lambda sig: np.max(np.abs(sig)) / np.mean(np.abs(sig)) if np.mean(np.abs(sig)) > 0 else 0,
        'crest_factor': lambda sig: np.max(np.abs(sig)) / np.sqrt(np.mean(sig ** 2)) if np.mean(sig ** 2) > 0 else 0,
        'clearance_factor': lambda sig: np.max(np.abs(sig)) / (np.mean(np.sqrt(np.abs(sig))) ** 2) if np.mean(
            np.sqrt(np.abs(sig))) > 0 else 0,
        'kurtosis_index': lambda sig: kurtosis(sig) / (np.std(sig) ** 4) if np.std(sig) > 0 else 0,
        'peak_index': lambda sig: np.max(np.abs(sig)) / np.sqrt(np.mean(sig ** 2)) if np.mean(sig ** 2) > 0 else 0,
        'pulse_index': lambda sig: np.ptp(sig) / np.mean(np.abs(sig)) if np.mean(np.abs(sig)) > 0 else 0,
        'mean_deviation_ratio': lambda sig: np.mean(np.abs(sig - np.mean(sig))) / np.mean(np.abs(sig)) if np.mean(
            np.abs(sig)) > 0 else 0,
        'root_mean_fourth': lambda sig: np.mean(sig ** 4) ** (1 / 4),
        'square_root_amplitude': lambda sig: np.mean(np.sqrt(np.abs(sig))),
        'fifth_statistical_moment': lambda sig: np.mean((sig - np.mean(sig)) ** 5),
        'sixth_statistical_moment': lambda sig: np.mean((sig - np.mean(sig)) ** 6),
        'kth_central_moment': lambda sig, k=4: np.mean((sig - np.mean(sig)) ** k),
        'shannon_entropy': lambda sig: entropy(np.histogram(sig, bins=10)[0] / len(sig) + 1e-10),
        'log_energy_entropy': lambda sig: -np.sum(
            (sig ** 2 / np.sum(sig ** 2)) * np.log2(sig ** 2 / np.sum(sig ** 2) + 1e-10))
        if np.sum(sig ** 2) > 0 else 0,
        'slope_sign_change': lambda sig: len(np.where(np.diff(np.sign(np.diff(sig))))[0]) if len(sig) >= 3 else 0,
        'zero_crossing_rate': lambda sig: len(np.where(np.diff(np.sign(sig)))[0]) / len(sig),
        'energy': lambda sig: np.sum(sig ** 2),
        'integrated_signal': lambda sig: np.sum(np.abs(sig)),
        'bi_segment_mav': lambda sig: np.mean(np.abs(sig[:len(sig) // 2])) if len(sig) >= 2 else 0,
        'tri_segment_mav': lambda sig: np.mean(np.abs(sig[len(sig) // 3:2 * len(sig) // 3])) if len(sig) >= 3 else 0,
        'mav_slope': lambda sig: np.mean(np.abs(np.diff(sig))) if len(sig) >= 2 else 0,
        'delta_rms': lambda sig: np.sqrt(np.mean((sig[1:] - sig[:-1]) ** 2)) if len(sig) >= 2 else 0,
        'root_sum_squares': lambda sig: np.sqrt(np.sum(sig ** 2)),
        'log_rms': lambda sig: np.log(np.sqrt(np.mean(sig ** 2)) + 1e-10),
        'average_amplitude_change': lambda sig: np.mean(np.abs(sig[1:] - sig[:-1])) if len(sig) >= 2 else 0,
        'peak_count': lambda sig: len(find_peaks(sig, height=0)[0]),

        'frequency_mean': lambda sig: np.mean(np.abs(fft(sig))),
        'frequency_variance': lambda sig: np.var(np.abs(fft(sig))),
        'frequency_skewness': lambda sig: skew(np.abs(fft(sig))),
        'frequency_kurtosis': lambda sig: kurtosis(np.abs(fft(sig))),
        'gravity_frequency': lambda sig: np.sum(fftfreq(len(sig), 1 / fs) * np.abs(fft(sig))) / np.sum(np.abs(fft(sig)))
        if np.sum(np.abs(fft(sig))) > 0 else 0,
        'frequency_std': lambda sig: np.std(np.abs(fft(sig))),
        'frequency_rms': lambda sig: np.sqrt(np.mean(np.abs(fft(sig)) ** 2)),
        'dominant_frequency': lambda sig: fftfreq(len(sig), 1 / fs)[np.argmax(np.abs(fft(sig)))],
        'fundamental_amplitude': lambda sig: np.max(np.abs(fft(sig))),
        'second_harmonic_amplitude': lambda sig: np.abs(fft(sig))[2] if len(fft(sig)) > 2 else 0,
        'third_harmonic_amplitude': lambda sig: np.abs(fft(sig))[3] if len(fft(sig)) > 3 else 0,
        'spectral_spread': safe_calc(lambda sig: (lambda s:
            (lambda fft_amp, freqs, grav_freq:
                np.sqrt(np.sum((freqs - grav_freq) ** 2 * fft_amp) / (np.sum(fft_amp) + EPS))
             )(np.abs(fft(s)), fftfreq(len(s), 1/fs),
               np.sum(fftfreq(len(s),1/fs)*np.abs(fft(s)))/(np.sum(np.abs(fft(s)))+EPS))
         )(sig)),
        'spectral_entropy': lambda sig: entropy(np.abs(fft(sig)) + 1e-10),
        'total_spectral_power': lambda sig: np.sum(np.abs(fft(sig)) ** 2),
        'first_spectral_moment': lambda sig: np.sum(fftfreq(len(sig), 1 / fs) * np.abs(fft(sig))),
        'second_spectral_moment': lambda sig: np.sum(fftfreq(len(sig), 1 / fs) ** 2 * np.abs(fft(sig))),
        'third_spectral_moment': lambda sig: np.sum(fftfreq(len(sig), 1 / fs) ** 3 * np.abs(fft(sig))),
        'fourth_spectral_moment': lambda sig: np.sum(fftfreq(len(sig), 1 / fs) ** 4 * np.abs(fft(sig))),
        'low_band_spectral_energy': lambda sig: np.sum(np.abs(fft(sig)[:len(sig) // 10]) ** 2),
        'mid_band_spectral_energy': lambda sig: np.sum(np.abs(fft(sig)[len(sig) // 10:len(sig) // 2]) ** 2),
        'high_band_spectral_energy': lambda sig: np.sum(np.abs(fft(sig)[len(sig) // 2:]) ** 2),
        'harmonic_ratio': lambda sig: np.sum(np.abs(fft(sig)[1:])) / np.abs(fft(sig)[0]) if np.abs(
            fft(sig)[0]) > 0 else 0,
        'mean_square_frequency': lambda sig: np.sum(fftfreq(len(sig), 1 / fs) ** 2 * np.abs(fft(sig))) / np.sum(
            np.abs(fft(sig)))
        if np.sum(np.abs(fft(sig))) > 0 else 0,
        'spectral_flux': lambda sig: np.sum(np.diff(np.abs(fft(sig))) ** 2),
        'spectral_flatness': lambda sig: np.exp(np.mean(np.log(np.abs(fft(sig)) + 1e-10))) / np.mean(np.abs(fft(sig)))
        if np.sum(np.abs(fft(sig))) > 0 else 0,
        'frequency_distortion': lambda sig: np.sum(np.abs(fft(sig)[1:]) ** 2) / (np.abs(fft(sig)[0]) ** 2 + 1e-10),
        'spectral_rolloff': safe_calc(lambda sig: (lambda s:
            (lambda amp, freqs:
                (freqs[np.where(np.cumsum(amp) >= 0.85 * np.sum(amp))[0][0]]
                 if len(np.where(np.cumsum(amp) >= 0.85 * np.sum(amp))[0])>0 else 0)
             )(np.abs(fft(s)), fftfreq(len(s),1/fs)))(sig)),

        'stft_mean': lambda sig: np.mean(np.abs(stft(sig, fs=fs, nperseg=min(128, len(sig)))[2])),
        'stft_variance': lambda sig: np.var(np.abs(stft(sig, fs=fs, nperseg=min(128, len(sig)))[2])),
        'stft_skewness': lambda sig: skew(np.abs(stft(sig, fs=fs, nperseg=min(128, len(sig)))[2]).flatten()),
        'stft_kurtosis': lambda sig: kurtosis(np.abs(stft(sig, fs=fs, nperseg=min(128, len(sig)))[2]).flatten()),
        'stft_entropy': lambda sig: entropy(np.abs(stft(sig, fs=fs, nperseg=min(128, len(sig)))[2]).flatten() + 1e-10),
        'wavelet_energy': lambda sig: np.sum(pywt.wavedec(sig, 'db1')[0] ** 2),
        'wavelet_entropy': lambda sig: entropy(pywt.wavedec(sig, 'db1')[0] + 1e-10),
        'cwt_mean': lambda sig: np.mean(pywt.cwt(sig, np.arange(1, 64), 'mexh')[0]),
        'cwt_variance': lambda sig: np.var(pywt.cwt(sig, np.arange(1, 64), 'mexh')[0]),
        'cwt_skewness': lambda sig: skew(pywt.cwt(sig, np.arange(1, 64), 'mexh')[0].flatten()),
        'cwt_kurtosis': lambda sig: kurtosis(pywt.cwt(sig, np.arange(1, 64), 'mexh')[0].flatten()),
        'hilbert_envelope_rms': lambda sig: np.sqrt(np.mean(np.abs(hilbert(sig)) ** 2)),

        'correlation_dimension': lambda sig: nolds.corr_dim(sig, 2),
        'sample_entropy': lambda sig: nolds.sampen(sig, emb_dim=2),
        'power_spectral_density_mean': lambda sig: np.mean(welch(sig, fs=fs, nperseg=min(128, len(sig)))[1]),
        'power_spectral_density_max': lambda sig: np.max(welch(sig, fs=fs, nperseg=min(128, len(sig)))[1])
    }

    return attributes