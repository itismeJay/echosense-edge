import pyaudio
import numpy as np
from config import SAMPLE_RATE, CHUNK_SIZE, CHANNELS

def get_audio_device():
    p = pyaudio.PyAudio()
    device_index = None

    print("Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info['name'].lower()
        print(f"  [{i}] {info['name']}")
        if 'respeaker' in name or 'seeed' in name:
            device_index = i
            print(f"  → ReSpeaker found at index {i}")
        elif device_index is None and ('emeet' in name or 'm0' in name):
            device_index = i
            print(f"  → EMEET mic found at index {i}")
        elif device_index is None and 'usb' in name and info.get('maxInputChannels', 0) > 0:
            device_index = i
            print(f"  → USB mic found at index {i}")

    p.terminate()
    return device_index

def capture_audio(duration: float = 1.0, device_index=None):
    p = pyaudio.PyAudio()
    
    frames = []
    num_frames = int(SAMPLE_RATE * duration / CHUNK_SIZE)
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK_SIZE
    )
    
    for _ in range(num_frames):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    audio_bytes = b''.join(frames)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

    return audio_np, audio_bytes

def get_waveform_snapshot(audio_np, num_points=40):
    if len(audio_np) == 0:
        return [0] * num_points
    indices = np.linspace(0, len(audio_np)-1, num_points, dtype=int)
    snapshot = [int(abs(audio_np[i])) for i in indices]
    return snapshot