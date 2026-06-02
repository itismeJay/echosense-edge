import pyaudio
import numpy as np
from config import SAMPLE_RATE, CHUNK_SIZE, CHANNELS

# Global instance so PortAudio stays initialized across calls
_pa_instance = None

def get_pyaudio_instance():
    """Returns a persistent PyAudio instance."""
    global _pa_instance
    if _pa_instance is None:
        _pa_instance = pyaudio.PyAudio()
    return _pa_instance

def get_audio_device():
    p = get_pyaudio_instance()
    device_index = None

    print("Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info['name'].lower()
        input_chans = info.get('maxInputChannels', 0)
        print(f"  [{i}] {info['name']} (Available Inputs: {input_chans})")

        if input_chans > 0:
            if 'respeaker' in name or 'seeed' in name:
                if device_index is None:
                    device_index = i
                    print(f"  → Active ReSpeaker input found at index {i}")
            elif device_index is None and ('emeet' in name or 'm0' in name):
                device_index = i
                print(f"  → Active EMEET mic input found at index {i}")
            elif device_index is None and 'usb' in name:
                device_index = i
                print(f"  → Active USB mic input found at index {i}")

    if device_index is None:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            name = info['name'].lower()
            if info.get('maxInputChannels', 0) > 0 and ('default' in name or 'sysdefault' in name):
                device_index = i
                print(f"  → Falling back to ALSA default at index {i}")
                break

    print(f"[INIT] Selected hardware device index: {device_index}")
    return device_index

def capture_audio(duration: float = 1.0, device_index=None):
    p = get_pyaudio_instance()

    # Query device channels using the persistent instance
    if device_index is not None:
        dev_info = p.get_device_info_by_index(device_index)
        hw_channels = min(max(1, int(dev_info['maxInputChannels'])), 2)
    else:
        hw_channels = CHANNELS

    frames = []
    num_frames = int(SAMPLE_RATE * duration / CHUNK_SIZE)

    # Streams can be opened and closed safely inside the loop
    stream = p.open(
        format=pyaudio.paInt16,
        channels=hw_channels,
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
    # DO NOT terminate here—keep 'p' warm for the next second's capture!

    audio_bytes = b''.join(frames)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

    # Mix stereo → mono by averaging both channels
    if hw_channels > 1:
        audio_np = audio_np.reshape(-1, hw_channels).mean(axis=1).astype(np.int16)
        audio_bytes = audio_np.tobytes()

    return audio_np, audio_bytes

def get_waveform_snapshot(audio_np, num_points=40):
    if len(audio_np) == 0:
        return [0] * num_points
    indices = np.linspace(0, len(audio_np)-1, num_points, dtype=int)
    snapshot = [int(abs(audio_np[i])) for i in indices]
    return snapshot

def terminate_audio_system():
    """Call this explicitly in main.py's 'finally' block when closing the app."""
    global _pa_instance
    if _pa_instance is not None:
        _pa_instance.terminate()
        _pa_instance = None