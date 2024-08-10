import subprocess
import cupy
import librosa
import numpy
from pydub import AudioSegment
import numpy as np
from matplotlib import pyplot
from scipy import signal, ndimage
from os import path, getcwd
from sys import argv

PATH = argv[1]

path_base = path.join(
    getcwd(),
    "separated",
    "htdemucs",
    path.basename(PATH).rsplit(sep=".", maxsplit=1)[0],
    "vocals.wav",
)
separated_path = path.join(
    path_base,
    "vocals.wav",
)

if not path.exists(separated_path):
    subprocess.run(["demucs", PATH, "-d", "cuda", "--two-stems", "vocals", "-j", "3"])

vocal = AudioSegment.from_file(separated_path)

print(f"{vocal.duration_seconds}sec")
print(f"Hz: {vocal.frame_rate}")
print(f"{vocal.channels} channel(s)")
vocal_numpy: np.ndarray = (
    np.array(vocal.get_array_of_samples())
    .astype(np.int16)
    .reshape((-1, vocal.channels))
)
# print(vocal_numpy.shape)
# print(vocal_numpy.dtype)
# vocal_0 = vocal_numpy[:, 0].astype(numpy.float32)
# vocal_1 = vocal_numpy[:, 1].astype(numpy.float32)
vocal_merge = numpy.mean(vocal_numpy, axis=-1, dtype=numpy.float32)
win = signal.windows.blackman(M=2**16)
f, t, Sxx = signal.stft(
    vocal_merge,
    vocal.frame_rate,
    window=win,
    nperseg=win.shape[0],
    scaling="spectrum",
    return_onesided=True,
)
S, phase = librosa.magphase(Sxx)
Sdb: numpy.ndarray = librosa.amplitude_to_db(S)
print(Sdb.shape)
low_pass = Sdb[200:600, :]
low_pass_norm = (low_pass - low_pass.min()) / (low_pass.max() - low_pass.min())
vocal_sum = numpy.sum(low_pass_norm, axis=0)
# pyplot.imshow(low_pass)
# pyplot.show()
ave_size = 100
# vocal_ave = np.convolve(vocal_sum, np.ones(ave_size) / ave_size, mode="same")
# vocal_ave = ndimage.gaussian_filter1d(vocal_ave, 6)
vocal_ave = np.convolve(vocal_sum, np.ones(ave_size) / ave_size, mode="same")
vocal_gaussian_norm = (vocal_ave - vocal_ave.min()) / (
    vocal_ave.max() - vocal_ave.min()
)
no_music: np.ndarray = vocal_gaussian_norm > 0.6
# print(no_music.astype(int).tolist())

pyplot.plot(vocal_gaussian_norm)
pyplot.show()
# exit()
# base_sound = AudioSegment.from_file(PATH)
base_sound = vocal
base_numpy: np.ndarray = (
    np.array(base_sound.get_array_of_samples())
    .astype(np.int16)
    .reshape((-1, base_sound.channels))
)

vocal_mute = (
    np.tile(
        no_music.reshape((-1, 1)),
        (1, int((base_sound.frame_rate / vocal.frame_rate) * (win.shape[0] / 2))),
    )
    .reshape((-1,))
    .astype(np.float16)[: base_numpy.shape[0]]
)
mute_size = int(ave_size * base_sound.frame_rate / 20)
vocal_mute_smoothed = cupy.convolve(
    cupy.array(vocal_mute), cupy.array(np.ones(mute_size) / mute_size), mode="same"
)
# pyplot.plot(cupy.asnumpy(vocal_mute_smoothed.reshape((-1, 1))))
# pyplot.show()
print(vocal_mute_smoothed.shape)
muted = cupy.asnumpy(vocal_mute_smoothed.reshape((-1, 1)) * cupy.array(base_numpy))
# pyplot.plot(muted[:, 1])
# pyplot.show()
# exit()
print(muted.dtype)
print(muted.shape)
muted_audio = AudioSegment(
    muted.astype(numpy.int16).tobytes(),
    sample_width=base_sound.sample_width,
    frame_rate=base_sound.frame_rate,
    channels=base_sound.channels,
)
muted_audio.export(
    path.join(
        getcwd(),
        "separated",
        "htdemucs",
        path.basename(PATH).rsplit(sep=".", maxsplit=1)[0],
        "muted.wav",
    ),
    "wav",
)
# pyplot.plot(muted[:, 0])
# pyplot.show()
subprocess.run(
    [
        "ffmpeg",
        "-i",
        path.join(
            path_base,
            "muted.wav",
        ),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        path.join(
            path_base,
            "for_whisper.wav",
        ),
    ]
)

print(
    f"whisper-ctranslate2 --model large-v3 --vad_filter True -f json -o {path_base} {path_base}/for_whisper.wav -p True"
)
print(
    f"python to_hiragana.py {path_base}/for_whisper.json"
)
