from dataclasses import dataclass
import pathlib
from glob import glob
from typing import Union, Tuple
import pandas as pd
import torch
import numpy as np
import cv2


"""
To avoid 'cannot instantiate 'PosixPath' on your system. Cache may be out of date, try `force_reload=True`' error
for some reason i get this error on this model, it didn't happened using models I've trained in the past
"""
pathlib.PosixPath = pathlib.WindowsPath


@dataclass
class Detector:
    model_path: str
    conf_threshold: float = .2
    ultralitycs_path: str = "ultralytics/yolov5"
    model_type: str = "custom"
    force_reload: bool = True

    def __post_init__(self) -> None:
        self.model = torch.hub.load(self.ultralitycs_path, self.model_type, self.model_path, self.force_reload)
        self.model.conf = self.conf_threshold

    def detect(self, img: Union[str, np.array]) -> Tuple[np.array, pd.DataFrame]:
        results = self.model([img])

        return np.squeeze(results.render()), results.pandas().xyxy[0]

#https://www.pexels.com/pl-pl/video/tlum-podrozujacych-w-terminalu-transportowym-3740034/
#https://www.pexels.com/pl-pl/video/pasazerowie-czekajacy-w-poczekalni-na-lotnisku-3723452/
#https://www.pexels.com/pl-pl/video/podrozujacy-w-terminalu-lotniska-3736783/
#https://www.pexels.com/pl-pl/video/pasazerow-chodzacych-na-korytarzu-lotniska-3740023/
#https://www.pexels.com/pl-pl/video/lot-swit-zachod-slonca-moda-8044791/
#https://www.pexels.com/pl-pl/video/droga-zachod-slonca-wschod-slonca-swiatlo-dzienne-5434560/
#https://www.pexels.com/pl-pl/video/osoba-kobieta-praca-pisanie-3695972/
# if __name__ == '__main__':
#     TEST_DATA_FOLDER = r"C:\Users\table\PycharmProjects\test2\FireSmokDetection\train_data\images\val"
#     TEST_VIDEOS_FOLDER = "Videos"
#     #
#     detector = Detector(model_path=fr"Model\best4.pt")
#     # detector2 = Detector(model_path=fr"Model\best4.pt")
#     # files = glob(f"{TEST_DATA_FOLDER}/*.*")
#     # for file in files:
#     #     image = cv2.imread(file)
#     #     image2 = image.copy()
#     #
#     #     converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     #     image_draw, res = detector.detect(img=converted)
#     #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     #
#     #     converted2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
#     #     image_draw2, res2 = detector2.detect(img=converted2)
#     #     image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
#     #
#     #     image_draw = cv2.resize(image_draw, (640, 480))
#     #     image_draw2 = cv2.resize(image_draw2, (640, 480))
#     #     cv2.imshow('MainWindow', image_draw)
#     #     cv2.imshow('MainWindow2', image_draw2)
#     #     cv2.waitKey(0)
#
#     # videos = glob(f"{TEST_VIDEOS_FOLDER}/*.mp4")
#     #
#     cap = cv2.VideoCapture(r"C:\Users\table\PycharmProjects\test2\FireSmokDetection\Videos\9780685-hd_1280_720_60fps.mp4")
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print("nara")
#             break
#
#         converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_draw, res = detector.detect(img=converted)
#         frame_draw = cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR)
#
#         cv2.imshow("xd", frame_draw)
#         cv2.waitKey(1)
#     cap.release()
#     cv2.destroyAllWindows()