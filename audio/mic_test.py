import pyaudio
import wave

MIC_INDEX = 0
DURATION = 5
RATE = 48000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2

audio = pyaudio.PyAudio()

# Get this BEFORE terminating PyAudio
sample_width = audio.get_sample_size(FORMAT)

print("Opening microphone...")

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=MIC_INDEX,
    frames_per_buffer=CHUNK
)

frames = []

print("RECORDING NOW — say START GAME")

# Exactly enough chunks for ~5 seconds
for _ in range(int(RATE / CHUNK * DURATION)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

print("Recording finished.")

stream.stop_stream()
stream.close()
audio.terminate()

# Save WAV
filename = "audio/test_recording.wav"

with wave.open(filename, "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))

print("Saved:", filename)