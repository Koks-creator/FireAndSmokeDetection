## Fire And Smoke Detection System


### 1. Introduction
Fire and smoke detection system which calculates severity based on area occupied by fire and smoke and sending telegram alert.
[![video](https://img.youtube.com/vi/CPuEzSP2ArQ/0.jpg)](https://www.youtube.com/CPuEzSP2ArQ) ![alert](https://github.com/user-attachments/assets/fa0cb3cd-1a85-4ef9-8899-5df586e00c00)

### 2. Setup
  - Install Python 3.9+
  - Create venv (if you want)
    ```
    python -m venv mycoolvenv
    linux: source mycoolvenv/bin/activate
    windows: mycoolvenv\Scripts\activate
    ```
  - Install requirements: 
    ```
    pip install requirements.txt
    ```
  - Create .env file (if you wanna have telegram alert, if you don't wanna don't create this file, just set TELEGRAM_ALERTS in config.py to False)
    ```
    BOT_TOKEN=
    CHAT_ID=
    ```

### 3. yolov5_tutorialwithgoogledrive.ipynb
Colab notebook for training yolov5 model

### 4. mocny_system.py
Main file when everything happens, I am not gonna explain every method and stuf cuz nobody reads it anyways and I expect you to have some python knowledge or you can use LLMs like chat gpt or claude or whatever...

### 5. detector.py
File with yolov5 detector

### 6. config.py
Config file for all other files

### 7. compare_models.py
You can compare models with this thing

### 8. class_counts.py
Simple tool for couting classes in dataset, so you can check if dataset is balanced or not

### 9. DatasetPrepTools/setup_dataset_folder.py
You can setup your dataset folder structure here

### 10. DatasetPrepTools/dataset_cleaner.py
Tool for cleaning dataset from too big/small images and non images

### 11. DatasetPrepTools/move_files.py
Moving cleaned images to specific folders - train, val, test 

### 12. Train your own model
See: https://github.com/Koks-creator/HowToTrainCustomYoloV5Model
