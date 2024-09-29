from pathlib import Path
from typing import Union
import logging
import os


class Config:
    ROOT_PATH: str = Path(__file__).resolve().parent
    MODELS_PATH: str = fr"{ROOT_PATH}/Model"
    MODEL_PATH: str = fr"{MODELS_PATH}/best10.pt"
    RAW_DATA_PATH: str = fr"{ROOT_PATH}/RawData"
    CLEANED_DATA_PATH: str = fr"{ROOT_PATH}/DataCleaned"
    VIDEOS_PATH: str = fr"{ROOT_PATH}/Videos"
    TRAIN_IMAGES_FOLDER: str = fr"{ROOT_PATH}/train_data/images/train"
    VAL_IMAGES_FOLDER: str = fr"{ROOT_PATH}/train_data/images/val"
    TRAIN_LABELS_FOLDER: str = fr"{ROOT_PATH}/train_data/labels/train"
    VAL_LABELS_FOLDER: str = fr"{ROOT_PATH}/train_data/labels/val"
    TEST_DATA_FOLDER: str = fr"{ROOT_PATH}/TestData"
    CLASSES_PATH: str = fr"{ROOT_PATH}/classes.txt"
    MIN_HEIGHT: int = 180
    MIN_WIDTH: int = 180
    MAX_HEIGHT: int = 4000
    MAX_WIDTH: int = 4000
    ALLOWED_EXTENSIONS: tuple = (".jpg", ".png", ".jpeg")
    VIDEO_CAP: Union[str, int, os.PathLike] = fr"{VIDEOS_PATH}/9780685-hd_1280_720_60fps.mp4"
    CHECK_INTERVAL_SEC: int = 300
    LOG_LEVEL: int = logging.INFO
    RETRY_INTERVAL: int = 30
    RETRY_COUNT: int = 3

