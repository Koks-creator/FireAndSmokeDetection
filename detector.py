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
