# EchoSense Edge Device Configuration

# Backend API
API_URL = "https://echosense-backend-75h3.onrender.com"

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 1040
CHANNELS = 1

# YAMNet model
YAMNET_MODEL_PATH = "/home/echosense/echosense-edge/yamnet.tflite"
YAMNET_CLASSES_PATH = "/home/echosense/echosense-edge/yamnet_class_map.csv"

# Vosk models
VOSK_FILIPINO_MODEL = "/home/echosense/echosense-edge/vosk-model-tl-ph-generic-0.6"
VOSK_ENGLISH_MODEL = "/home/echosense/echosense-edge/vosk-model-small-en-us-0.15"

# Sensor location (included in every alert payload)
LOCATION = "Grade 6 Classroom"

# NOTE: Detection thresholds live in detection/thresholds.py (the authoritative
# file the runtime actually reads). The former YAMNET_CONFIDENCE_THRESHOLD and
# DURATION_THRESHOLD duplicates here were dead code and have been removed.
