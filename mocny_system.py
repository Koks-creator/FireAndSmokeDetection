from dataclasses import dataclass, field
import logging
from enum import Enum
from time import time
from copy import deepcopy
from datetime import datetime
import requests
from collections import Counter, defaultdict
from math import sqrt
from pydantic import field_validator, BaseModel
from typing import Union, Tuple, List
import os
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, retry_if_result, RetryError

load_dotenv()

from FireSmokDetection.detector import Detector
from FireSmokDetection.config import Config


@dataclass
class CustomLogger:
    format: str = "%(asctime)s - %(name)s - %(levelname)s - Line: %(lineno)s - %(message)s"
    file_handler_format: logging.Formatter = logging.Formatter(format)
    log_file_name: str = "logs.log"
    logger_name: str = __name__
    logger_log_level: int = logging.ERROR
    file_handler_log_level: int = logging.ERROR

    def create_logger(self) -> logging.Logger:
        logging.basicConfig(format=self.format)
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.logger_log_level)

        file_handler = logging.FileHandler(self.log_file_name)
        file_handler.setLevel(self.file_handler_log_level)
        file_handler.setFormatter(logging.Formatter(self.format))
        logger.addHandler(file_handler)

        return logger


logger = CustomLogger(
    logger_log_level=Config.LOG_LEVEL,
    file_handler_log_level=Config.LOG_LEVEL
).create_logger()


class SmokeSeverity(Enum):
    LOW = (0, 10)
    MODERATE = (10, 30)
    HIGH = (30, 70)
    EXTREME = (70, float('inf'))


class FireSeverity(Enum):
    LOW = (0, 1)
    MODERATE = (1, 2)
    HIGH = (2, 50)
    EXTREME = (50, float('inf'))


class DetectionData(BaseModel):
    detection_id: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    conf: float
    class_name: str

    @field_validator("conf")
    def conf_check(cls, v):
        return round(v, 3)

    @staticmethod
    def distance_between_points(x1, y1, x2, y2) -> float:
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_bbox(self) -> Tuple[int, int, int, int]:
        return self.xmin, self.ymin, self.xmax, self.ymax

    def get_area(self) -> int:
        a = self.distance_between_points(x1=self.xmin, y1=self.ymin, x2=self.xmax, y2=self.ymin)
        b = self.distance_between_points(x1=self.xmin, y1=self.ymin, x2=self.xmin, y2=self.ymax)

        return int(a * b)

    class Config:  # works like frozen=True from dataclasses, but frozen stops validator from changing rounding conf
        frozen = True


def check_status_code(status_code: int):
    # Retry for status codes 500 and 503
    if status_code not in [200, 202]:
        logger.warning(f"Retrying sending alert... Received status code {status_code}")
        return True
    return False


@dataclass
class FireAlertSystem:
    model_path: Union[str, os.PathLike]
    classes_file_path: Union[str, os.PathLike]
    conf_thr: float = .2
    force_reload: bool = True
    location: str = "Unknown"
    send_alerts: bool = True
    check_interval_sec: int = 300
    _image_w: int = field(default=None, init=False)
    _image_h: int = field(default=None, init=False)

    def __post_init__(self):
        logger.info(f"Starting with parameters:\n"
                    f"- {self.model_path}\n"
                    f"- {self.classes_file_path}\n"
                    f"- {self.conf_thr}\n"
                    f"- {self.force_reload}\n"
                    f"- {self.location}\n"
                    f"- {self.send_alerts}\n"
                    f"- {self.check_interval_sec}"
                    )
        self.detector = Detector(
            model_path=self.model_path,
            conf_threshold=self.conf_thr,
            force_reload=self.force_reload
        )
        logger.info(f"Model {self.model_path} loaded with params:\n"
                    f"- {self.conf_thr=}\n"
                    f"- {self.force_reload=}"
                    )

        if self.send_alerts:
            self.__bot_token = os.environ["BOT_TOKEN"]
            self.__chat_id = os.environ["CHAT_ID"]

    @staticmethod
    def get_severity(class_name: str, value: float) -> Union[None, str]:
        severities = {
            "smoke": SmokeSeverity,
            "fire": FireSeverity,
        }
        for severity in severities[class_name]:
            if severity.value[0] <= value <= severity.value[1]:
                return severity.name
        return None

    def get_severities(self, areas: defaultdict) -> defaultdict:
        areas_copied = deepcopy(areas)
        for class_name, area_data in areas.items():
            severity = self.get_severity(class_name=class_name, value=area_data["AreaPerc"])
            areas_copied[class_name]["Severity"] = severity

        return areas_copied

    def get_telegram_template(self, areas: defaultdict) -> str:
        logger.debug("Getting template..")
        severities = self.get_severities(areas)

        fire_data = severities['fire'] if 'fire' in list(severities.keys()) else None
        smoke_data = severities['smoke'] if 'smoke' in list(severities.keys()) else None

        message_template = f"""
        🔥 *ALERT: Fire in {self.location}* 🔥

        📍 *Location:* {self.location}
        🔥 *Fire Intensity:* {fire_data["Severity"] if fire_data else None}
        💨 *Smoke Intensity:* {smoke_data["Severity"] if smoke_data else None}
        🚒 *Percentage of Fire Containment:* {round(fire_data["AreaPerc"], 2) if fire_data else None}%
        🚒 *Percentage of Smoke Containment:* {round(smoke_data["AreaPerc"], 2) if smoke_data else None}%
        📢 *Recommendations:* Exercise extreme caution.

        🕓 *Last Update:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, 
            next update in: {self.check_interval_sec // 60}m
        """
        logger.debug(f"Template: \n {message_template}")

        return message_template

    @retry(
        stop=stop_after_attempt(Config.RETRY_INTERVAL),
        wait=wait_fixed(Config.RETRY_COUNT),
        retry=(retry_if_exception_type(requests.exceptions.RequestException) |
               retry_if_result(check_status_code)
               ))
    def send_tg_message(self, msg: str) -> int:
        logger.debug(f"Sending alert with msg: {msg}")
        url = f"https://api.telegram.org/bot{self.__bot_token}/sendMessage?" \
              f"chat_id={self.__chat_id}&parse_mode=Markdown&text={msg}"
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Error when making sending tg message: {response.status_code}, {response.content}")
            return response.status_code

        logger.debug(f"Alert sent")
        return response.status_code

    def preprocess_image(self, image: np.array) -> Tuple[np.array, np.array]:
        logger.debug("Processing image")

        converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_res, image = self.detector.detect(img=converted)
        image_res = cv2.cvtColor(image_res, cv2.COLOR_RGB2BGR)
        return image_res, image

    def analyze_results(self, detector_res: pd.DataFrame) -> Tuple[List[DetectionData], defaultdict, Counter]:
        logger.debug("Analyze results")

        detection_data = []
        total_areas = defaultdict(dict)
        counter = Counter()

        for row in detector_res.iterrows():
            row_id, row_data = row

            xmin = int(row_data.xmin)
            ymin = int(row_data.ymin)
            xmax = int(row_data.xmax)
            ymax = int(row_data.ymax)
            conf = row_data.confidence
            class_name = row_data["name"]  # row.name returns int instead of string name xd

            # cv2.line(image, (xmin, ymin), (xmax, ymin), (255, 0, 255), 3)
            # cv2.line(image, (xmin, ymin), (xmin, ymax), (255, 0, 255), 3)

            detection_obj = DetectionData(
                detection_id=row_id,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                conf=conf,
                class_name=class_name
            )
            detection_data.append(detection_obj)
            detection_area = detection_obj.get_area()
            detection_area_perc = self.compare_areas(area=detection_area)
            if ["Area", "AreaPerc"] != list(total_areas[class_name].keys()):
                total_areas[class_name] = {"Area": detection_area, "AreaPerc": detection_area_perc}
            else:
                total_areas[class_name]["Area"] += detection_area
                total_areas[class_name]["AreaPerc"] += detection_area_perc
            counter[class_name] += 1

        logger.debug(f"Results analyzed: \n"
                     f"- {detection_data=}\n"
                     f"- {total_areas=}\n"
                     f"- {counter=}"
                     )

        return detection_data, total_areas, counter

    def compare_areas(self, area: int) -> float:
        logger.debug("Compering areas")
        image_area = self._image_h * self._image_w
        perc = (area * 100) / image_area

        logger.debug(f"Areas compered: {perc=}, {image_area=}")

        return perc

    def detect_on_image(self, image: np.array, show: bool = False) -> Union[np.array, defaultdict]:
        logger.debug(f"Detecting on image {image.shape=}, {show=}")

        self._image_h, self._image_w, _ = image.shape  # h, w, c

        image_res, results = self.preprocess_image(image=image)
        detections, total_areas, counter = self.analyze_results(detector_res=results)
        logger.debug(f"{detections=}")

        severities = self.get_severities(areas=total_areas)
        logger.debug(f"{severities=}")

        start_y = 50
        for total_area, counter in zip(total_areas.items(), counter.items()):
            class_name = counter[0]
            class_count = counter[1]

            class_area_perc = round(total_area[1]["AreaPerc"], 2)
            cv2.putText(image_res, f"{class_name.capitalize()}: {class_count} ({class_area_perc}%)", (10, start_y),
                        cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 0, 255), 2)

            start_y += 25
        for class_name, row in severities.items():
            severity = row["Severity"]
            cv2.putText(image_res, f"{class_name.capitalize()} severity: {severity}", (10, start_y),
                        cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 0, 255), 2)
            start_y += 25

        if show:
            logger.debug("Showing image")
            cv2.imshow("Res", image_res)
            cv2.waitKey(0)

        logger.debug("Detecting on image finished")
        return image_res, total_areas

    def detect_on_video(self, video_cap: Union[str, os.PathLike, int]) -> None:
        logger.debug(f"Detecting video: {video_cap=}")
        cap = cv2.VideoCapture(video_cap)

        start_time = None
        p_time = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Leaving...")
                break
            frame, total_areas = self.detect_on_image(image=frame)
            if total_areas:
                if not start_time and self.send_alerts:
                    start_time = time()
                    try:
                        self.send_tg_message(
                            msg=self.get_telegram_template(areas=total_areas)
                        )
                    except RetryError:
                        logger.error("Failed to send alert!")

            if start_time:
                time_dff = int(time() - start_time)
                cv2.putText(frame, f"Next alert in: {self.check_interval_sec - time_dff}s",
                            (self._image_w-250, 25), cv2.FONT_HERSHEY_PLAIN,
                            1.4, (255, 0, 255), 2)
                if time_dff > self.check_interval_sec:
                    start_time = None

            c_time = time()
            fps = int(1 / (c_time - p_time))
            p_time = c_time

            cv2.putText(frame, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_PLAIN,
                        1.4, (255, 0, 255), 2)
            cv2.imshow("VideoRes", frame)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        fire_alert = FireAlertSystem(
            model_path=Config.MODEL_PATH,
            classes_file_path=Config.CLASSES_PATH,
            location="Mocna Lokalizacja",
            send_alerts=True,
            check_interval_sec=Config.CHECK_INTERVAL_SEC
        )
        fire_alert.detect_on_video(video_cap=Config.VIDEO_CAP)
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
