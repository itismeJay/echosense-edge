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

# Detection thresholds
YAMNET_CONFIDENCE_THRESHOLD = 0.5
DURATION_THRESHOLD = 2.0
LOCATION = "Grade 6 Classroom"
