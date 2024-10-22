import os
import cv2
import torch
import time
import math
import json
import argparse
# import easyocr
import numpy as np

from PIL import Image
from datetime import datetime
from paddleocr import PaddleOCR
from IPython.display import display
from boxmot import DeepOCSORT
from pathlib import Path


ocr = PaddleOCR(
    use_gpu=False, lang='en', dilation=True
)


def augment_image(bgr_image):
    gray_crop_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    up_gray_crop_image = cv2.pyrUp(gray_crop_image)
    blur_img = cv2.GaussianBlur(up_gray_crop_image, (3, 3), 0)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    augment_img = cv2.filter2D(blur_img, -1, kernel)

    return augment_img


def lp_detection(input_image):
    plates = yolo_lp_detect(input_image)

    if plates is not None and len(plates) > 0:
        list_plates = plates.pandas().xyxy[0].values.tolist()
        return list_plates

    return None


def lp_recognition(crop_img):
    aug_img = augment_image(crop_img)
    result = ocr.ocr(aug_img, cls=False)

    if result[0] is None:
        pass
    else:
        result_length = sum(len(item) for item in result)
        list_text = ""

        for index in range(result_length):
            boxes = [line[index] for line in result]
            text = boxes[0][1][0]
            list_text += text

        if list_text:
            list_text = list_text.replace("-", "").replace(".", "")
            # print("text:", list_text)

        print("text:", list_text)
        now = datetime.now().strftime("%H%M%S_%d%m%Y")
        data = {
            "time": now,
            "license_plate": list_text
        }
        return list_text


# Load model detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_lp_detect = torch.hub.load(
    'yolov5',
    'custom',
    path='/home/minhthanh/Downloads/new_weights_yolov5_v6.0/v6.0_best.pt',
    force_reload=True,
    source='local'
)
yolo_lp_detect.conf = 0.5
yolo_lp_detect.eval()

# Tracking model
tracker = DeepOCSORT(
    model_weights=Path('yolov5/osnet_x0_25_msmt17.pt'),  # Which ReID model to use
    device='cpu',
    fp16=False,
)

# Save history
his_lp = "/home/minhthanh/code_python/biensoxe2/history"

# Input video
video_path = "/home/minhthanh/Downloads/20050117_001203_0001cc06a9eb1423.mp4"
vid = cv2.VideoCapture(video_path)
cnt = 0

while True:
        
    ret, frame = vid.read()

    frame_copy = frame.copy()
    frame_height, frame_width, _ = frame_copy.shape
    list_plates = lp_detection(frame_copy)

    if list_plates is not None and len(list_plates) > 0:
        # check rule detection
        # last_list_plates = rule_detection(list_plates, frame_height, frame_width)
        
        for plate in list_plates:
            left = int(plate[0])  # xmin
            top = int(plate[1])  # ymin
            width = int(plate[2] - plate[0])  # xmax - xmin
            height = int(plate[3] - plate[1])  # ymax - ymin
            print(plate)
            cv2.rectangle(frame_copy, (left, top), (int(plate[2]), int(plate[3])), (0, 0, 255), 2)
            crop_img = frame_copy[top:top+height, left:left+width]

            # PaddleOCR
            text = lp_recognition(crop_img)
            cv2.putText(
            frame_copy,
            text,
            (left,top-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

    frame_copy = cv2.resize(frame_copy, (1000, 640))
    cv2.imshow("img", frame_copy)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()