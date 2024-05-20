from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.ndimage import maximum_filter

def make_spectrogram(samples, sample_rate, name):
    frequencies, times, my_spectrogram = signal.spectrogram(samples, sample_rate, scaling='spectrum', window=('hann',))

    eps = np.finfo(float).eps
    my_spectrogram = np.maximum(my_spectrogram, eps)

    plt.pcolormesh(times, frequencies, np.log10(my_spectrogram), shading='auto')
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')
    plt.savefig(f'10sem/results/output/spectrogram/spectrogram_{name}.png', dpi=500)

    
    return frequencies, times, my_spectrogram

def calculate_max_min_freq(voice_path):
    y, sr = librosa.load(voice_path, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    frequencies = librosa.fft_frequencies(sr=sr)
    mean_spec = np.mean(D, axis=1)

    idx_min = np.argmax(mean_spec > -80)
    idx_max = len(mean_spec) - np.argmax(mean_spec[::-1] > -80) - 1

    min_freq = frequencies[idx_min]
    max_freq = frequencies[idx_max]

    return max_freq, min_freq

def calculate_max_tembr(filepath):
    data, sample_rate = librosa.load(filepath)
    chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate)
    f0 = librosa.piptrack(y=data, sr=sample_rate, S=chroma)[0]
    max_f0 = np.argmax(f0)
    return max_f0

def calculate_peaks(freq, t, spec):
    delta_t = int(0.1 * len(t))
    delta_freq = int(50 / (freq[1] - freq[0]))
    filtered = maximum_filter(spec, size=(delta_freq, delta_t))

    peaks_mask = (spec == filtered)
    peak_values = spec[peaks_mask]
    peak_frequencies = freq[peaks_mask.any(axis=1)]

    top_indices = np.argsort(peak_values)[-3:]
    top_frequencies = peak_frequencies[top_indices]

    return list(top_frequencies)

def main():
    sounds = ["10sem/results/input/a.wav", "10sem/results/input/i.wav", "10sem/results/input/gav.wav"]
    names = ["a", "i", "gav"]
    i = 0
    for sound in sounds:
        rate, samples = wavfile.read(sound)
        freq, t, spec = make_spectrogram(samples, rate, names[i])
        max_fr, min_fr = calculate_max_min_freq(sound)
        print(f"Max fr {names[i]}: {max_fr}")
        print(f"Min fr {names[i]}: {min_fr}")
        print(f"Max tembr {names[i]}: {calculate_max_tembr(sound)}")
        print(f"3 max format {names[i]}: {calculate_peaks(freq, t, spec)}\n")
        i+=1

if __name__ == "__main__":
    main()
