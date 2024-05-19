from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def make_spectrogram(samples, sample_rate):
    frequencies, times, my_spectrogram = signal.spectrogram(samples, sample_rate, scaling='spectrum', window=('hann',))

    eps = np.finfo(float).eps
    my_spectrogram = np.maximum(my_spectrogram, eps)

    plt.pcolormesh(times, frequencies, np.log10(my_spectrogram), shading='auto')
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')

# Низких частот
def noise_reduction(samples, sample_rate, cutoff_freuency):
    z = signal.savgol_filter(samples, 101, 3)
    b, a = signal.butter(3, cutoff_freuency / sample_rate)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, z, zi = zi * z[0])
    return z

def make_butter_filter(sample_rate, data):
    b, a = signal.butter(10, 0.1, btype='lowpass')
    filtered_signal = signal.filtfilt(b, a, data)
    wavfile.write("9sem/results/output/sounds/butter.wav", sample_rate, filtered_signal.astype(np.int16))
    make_spectrogram(filtered_signal, sample_rate)
    plt.savefig('9sem/results/output/spectrogram/spectrogram_butter_reduced.png', dpi=500)

def make_savgol_filter(sample_rate, data):
    denoised_savgol = signal.savgol_filter(data, 75, 5)
    wavfile.write("9sem/results/output/sounds/savgol.wav", sample_rate, denoised_savgol.astype(np.int16))
    make_spectrogram(denoised_savgol, sample_rate)
    plt.savefig('9sem/results/output/spectrogram/spectrogram_savgol_reduced.png', dpi=500)

def noise_reduction(samples, sample_rate, cutoff_freuency):
    z = signal.savgol_filter(samples, 101, 3)
    b, a = signal.butter(3, cutoff_freuency / sample_rate)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, z, zi = zi * z[0])
    return z

def to_pcm(y):
    return np.int16(y / np.max(np.abs(y)) * 30000)

def main():
    size = 500

    # Построение спектрограммы
    sample_rate, samples = wavfile.read("9sem/results/input/sound1.wav")

    make_butter_filter(sample_rate, samples)
    make_savgol_filter(sample_rate, samples)

    # Поиск моментов времени с наибольшей энергией
    reduced_noise = noise_reduction(samples, sample_rate, cutoff_freuency=4000)
    reduced_twice_noise = noise_reduction(reduced_noise, sample_rate, cutoff_freuency=4000)
    frequencies, times, my_spectrogram = signal.spectrogram(reduced_twice_noise, sample_rate, scaling='spectrum',
                                                            window=('hann',))
    energies = np.sum(my_spectrogram, axis=0)
    peaks, _ = signal.find_peaks(energies, distance=1)
    # Отображение моментов времени с наибольшей энергией
    plt.figure()
    plt.plot(times, energies)
    plt.plot(times[peaks], energies[peaks], "x")
    plt.xlabel('Время [с]')
    plt.ylabel('Энергия')
    plt.title('Моменты с наибольшей энергией')
    plt.savefig('9sem/results/output/high_energy_moments.png')

if __name__ == "__main__":
    main()
