from librosa import load, stft, magphase, amplitude_to_db, display
from librosa.core.audio import __audioread_load
from matplotlib import pyplot
# from scipy import signal
import numpy
import librosa
from cupyx.scipy import signal
import cupy

cupy.cuda.set_allocator(None)
PATH = r"C:\Users\tomokazu\Music\radio\01music_20240718200000.m4a"
duration = librosa.get_duration(path=PATH)
print(duration)
x, fs = __audioread_load(PATH,  offset=0, duration=duration, dtype=numpy.float32)

print(x, fs, x.shape[-1] / fs)
win = signal.windows.hann(M=2 ** 16)
f, t, Sxx = signal.stft(x, fs, window=win, nperseg=win.shape[0], scaling="spectrum", return_onesided=True)
f_host: numpy.ndarray = cupy.asnumpy(f)
t_host: numpy.ndarray = cupy.asnumpy(t)
Sxx_host: numpy.ndarray = cupy.asnumpy(Sxx)
S, phase = librosa.magphase(Sxx_host)
Sdb = librosa.amplitude_to_db(S)
del f
del t
del Sxx
print(x.shape, f_host.shape, t_host.shape, Sxx_host.shape, S.shape)
print(x.dtype, f_host.dtype, t_host.dtype, Sxx_host.dtype, S.dtype)
pyplot.imshow(Sdb)
pyplot.show()
