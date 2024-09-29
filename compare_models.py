from typing import Optional, Tuple
from time import time
from glob import glob
from dataclasses import dataclass
import cv2
import numpy as np
import matplotlib.pyplot as plt

from FireSmokDetection.detector import Detector
from FireSmokDetection.config import Config
import matplotlib


@dataclass
class ModelCompare:
    models_path: str
    images_path: Optional[str] = None
    videos_path: Optional[str] = None
    conf_threshold: float = .2
    start_ind_model: Optional[int] = None
    end_ind_model: Optional[int] = None
    start_ind_images: Optional[int] = None
    end_ind_images: Optional[int] = None
    start_ind_videos: Optional[int] = None
    end_ind_videos: Optional[int] = None
    image_size: Tuple[int, int] = (640, 480)
    video_image_size: Tuple[int, int] = (640, 480)

    def __post_init__(self) -> None:
        self.model_slice = slice(self.start_ind_model, self.end_ind_model)
        self.images_slice = slice(self.start_ind_images, self.end_ind_images)
        self.videos_slice = slice(self.start_ind_videos, self.end_ind_videos)
        self.images = []
        self.videos = []

        self.models = glob(f"{Config.MODELS_PATH}/*.pt")[self.model_slice]
        b = plt.get_backend()
        self.detectors = [Detector(model_path=model) for model in self.models]
        matplotlib.use(b)
        if self.images_path:
            self.images = glob(f"{self.images_path}/*.*")[self.images_slice]

        if self.videos_path:
            self.videos = glob(f"{self.videos_path}/*.*")[self.videos_slice]

    @staticmethod
    def prepare_img(frame: np.array, detector: Detector) -> tuple[np.array, np.array]:
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_draw, _ = detector.detect(img=converted)
        return frame_draw, frame

    @staticmethod
    def display_images(images: list[np.array], titles: list[str]):
        if not images:
            print("No images to display.")
            return

        n = len(images)
        plt.figure(figsize=(15, 5))

        for i, (image, title) in enumerate(zip(images, titles)):
            plt.subplot(1, n, i + 1)
            plt.imshow(image)
            plt.title(title)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def compare_images(self) -> None:
        if self.images:
            for index, file in enumerate(self.images):
                img = cv2.imread(file)
                if img is None:
                    print(f"Failed to load image: {file}")
                    continue

                results = []
                titles = []

                for detector, model in zip(self.detectors, self.models):
                    res, _ = self.prepare_img(frame=img, detector=detector)

                    res_resized = cv2.resize(res, self.image_size)
                    cv2.putText(res_resized, f"{index + 1}/{len(self.images)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.4,
                                (255, 0, 255), 2)
                    results.append(res_resized)
                    titles.append(f"Model: {model.split('/')[-1]}")

                self.display_images(results, titles)
        else:
            print("No images")

    def compare_videos(self) -> None:
        stop = False
        if self.videos:
            for index, video_file in enumerate(self.videos):
                if stop:
                    print("Leaving...")
                    break
                cap = cv2.VideoCapture(video_file)

                if not cap.isOpened():
                    print(f"Failed to open video: {video_file}")
                    continue

                p_time = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print(f"End of video: {video_file}")
                        break

                    for detector, model in zip(self.detectors, self.models):
                        res, _ = self.prepare_img(frame=frame, detector=detector)
                        res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                        res_resized = cv2.resize(res_rgb, self.video_image_size)

                        c_time = time()
                        fps = int(1 / (c_time - p_time))
                        p_time = c_time

                        cv2.putText(res_resized, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_PLAIN,
                                    1.4, (255, 0, 255), 2)
                        cv2.putText(res_resized, f"{index + 1}/{len(self.videos)}", (10, 50), cv2.FONT_HERSHEY_PLAIN,
                                    1.4, (255, 0, 255), 2)
                        model_name = model.split('/')[-1]
                        cv2.imshow(f'Model: {model_name}', res_resized)

                    key = cv2.waitKey(1)
                    if key == 13:
                        break
                    if key == ord("q"):
                        stop = True
                        break

                cap.release()
                cv2.destroyAllWindows()
        else:
            print("No videos to process.")


if __name__ == '__main__':
    model_compare = ModelCompare(
        models_path=Config.MODELS_PATH,
        images_path=Config.VAL_IMAGES_FOLDER,
        videos_path=Config.VIDEOS_PATH,
        video_image_size=(640, 240),
        # start_ind_model=3
    )
    model_compare.compare_videos()
    # model_compare.compare_images()
