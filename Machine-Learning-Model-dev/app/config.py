import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

PROJECT_NAME = "RegQuest ML Microservice"
VERSION = "1.0.0"
HOST = "0.0.0.0"
PORT = 8001

