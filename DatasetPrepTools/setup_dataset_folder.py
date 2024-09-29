import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config

MAIN_FOLDER = "train_data"
SUB_FOLDERS = {
    "images": ["train", "val"],
    "labels": ["train", "val"],
}
TEST_FOLDER = "TestData"

test_folder_path = Config.TEST_DATA_FOLDER
if not os.path.exists(test_folder_path):
    os.mkdir(test_folder_path)

main_folder_path = f"{Config.ROOT_PATH}/{MAIN_FOLDER}"
if not os.path.exists(main_folder_path):
    os.mkdir(main_folder_path)

for folder, sub_folders in SUB_FOLDERS.items():
    path_to_create = None
    for sub_folder in sub_folders:
        try:
            path_to_create = f"{Config.ROOT_PATH}/{MAIN_FOLDER}/{folder}/{sub_folder}"
            os.makedirs(path_to_create)
        except FileExistsError:
            print(f"Path: {path_to_create} already exists")


#labelimg C:\Users\table\PycharmProjects\test2\FireSmokDetection\train_data\images\train C:\Users\table\PycharmProjects\test2\FireSmokDetection\train_data\labels\train\classes.txt C:\Users\table\PycharmProjects\test2\FireSmokDetection\train_data\labels\train
