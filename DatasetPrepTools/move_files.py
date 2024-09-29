import os
import shutil
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

SOURCE_FOLDER = Config.CLEANED_DATA_PATH
TRAIN_FOLDER = Config.TRAIN_IMAGES_FOLDER
VAL_FOLDER = Config.VAL_IMAGES_FOLDER
TEST_FOLDER = Config.TEST_DATA_FOLDER
TRAIN_PERC = 99
VAL_PERC = 0
TEST_PERC = 1
CLEAR_SOURCE_FOLDER = True

all_files = os.listdir(SOURCE_FOLDER)
data_len = len(all_files)

train_range = (0, int((TRAIN_PERC * data_len)/100))
val_range = (train_range[1], int((VAL_PERC * data_len)/100) + train_range[1])
test_range = (val_range[1], int((TEST_PERC * data_len)/100) + val_range[1])

train_images = all_files[train_range[0]:train_range[1]]
val_images = all_files[val_range[0]:val_range[1]]
test_images = all_files[test_range[0]:test_range[1]]


folders = [TRAIN_FOLDER, VAL_FOLDER, TEST_FOLDER]
file_chunks = [train_images, val_images, test_images]


for ind, chunk in enumerate(file_chunks):
    for file in tqdm(chunk):
        if not CLEAR_SOURCE_FOLDER:
            shutil.copy(rf"{SOURCE_FOLDER}\{file}", folders[ind])
        else:
            shutil.move(rf"{SOURCE_FOLDER}\{file}", folders[ind])