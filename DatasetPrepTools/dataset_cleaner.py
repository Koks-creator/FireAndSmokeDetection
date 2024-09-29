from dataclasses import dataclass
import glob
import sys
import shutil
from time import perf_counter
import zipfile
import os
from PIL import Image
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config


@dataclass(frozen=True)
class SizeRange:
    min_width: int
    min_height: int
    max_width: int
    max_height: int


@dataclass
class DatasetCleaner:
    dest_folder: str
    allowed_size_range: SizeRange = SizeRange(180, 180, 4000, 4000)  # min_h, min_w, max_h, max_w
    allowed_extensions: tuple = (".jpg", ".png", ".jpeg")

    def __post_init__(self) -> None:
        if not os.path.exists(self.dest_folder):
            os.mkdir(self.dest_folder)

    def clean_data(self, dataset_path: str, file_base_name: str) -> None:
        filename, ext = os.path.splitext(dataset_path)
        if ext == ".zip":
            with zipfile.ZipFile(dataset_path) as zip_ref:
                zip_ref.extractall(filename)

        all_files = glob.glob(f"{filename}/*.*")

        start = perf_counter()

        start_i = 0
        for index, file in enumerate(all_files):
            _, ext = os.path.splitext(file)
            if ext in self.allowed_extensions:
                img = Image.open(file)
                h, w = img.height, img.width
                if self.allowed_size_range.min_height < h < self.allowed_size_range.max_height and \
                        self.allowed_size_range.min_width < w < self.allowed_size_range.max_width:

                    shutil.copyfile(file, fr"{self.dest_folder}/{file_base_name}_{start_i}{ext}")
                    start_i += 1
        print(f"Done {perf_counter() - start}s")


if __name__ == '__main__':
    FOLDER = Config.RAW_DATA_PATH
    dc = DatasetCleaner(
        dest_folder=Config.CLEANED_DATA_PATH,
        allowed_size_range=SizeRange(Config.MIN_HEIGHT, Config.MIN_WIDTH, Config.MAX_HEIGHT, Config.MAX_WIDTH),
        allowed_extensions=Config.ALLOWED_EXTENSIONS
    )

    zip_files = os.listdir(FOLDER)
    for index, zip_file in enumerate(zip_files):
        dc.clean_data(dataset_path=f"{FOLDER}/{zip_file}", file_base_name=f"fire90{index}")
    print(len(os.listdir(Config.CLEANED_DATA_PATH)))
