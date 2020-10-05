import os
import logging

HOST = "0.0.0.0"
PORT = "5000"
DATASET_DIR = "tweet-sentiment-extraction"
TRAIN = os.path.join(DATASET_DIR, "train.csv")
# TEST = os.path.join(DATASET_DIR, "test.csv")
MODELS = "models"

LOG_DIR = "log"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)