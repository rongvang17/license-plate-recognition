import cv2
import torch
import math
import os
import time
import argparse
import easyocr
import numpy as np
import json

from PIL import Image
from datetime import datetime
from paddleocr import PaddleOCR
from IPython.display import display

ocr = PaddleOCR(
    use_gpu=False, lang='vi', dilation=True,
    det_db_box_thresh=0.5, det_limit_side_len=2200, 
    use_dilation=True, det_east_nms_thresh=0.6, 
    det_sast_nms_thresh=0.6, show_log=False
)

def agument_image(bgr_image):
    gray_crop_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    up_gray_crop_image = cv2.pyrUp(gray_crop_image)
    blur_img = cv2.GaussianBlur(up_gray_crop_image,(3,3), 0)
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    agument_img = cv2.filter2D(blur_img, -1, kernel)

    return agument_img


def lp_detection(input_image):
    plates = yolo_LP_detect(input_image) # resize image with keep size

    if plates is not None and len(plates)>0:
        list_plates = plates.pandas().xyxy[0].values.tolist()
        return list_plates

    return None
    

def lp_recognition(crop_img):
    agu_img = agument_image(crop_img)
    result = ocr.ocr(agu_img, cls=False)

    if result[0] == None:
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
            print("text:", list_text)
        
        now = datetime.now().strftime("%H%M%S_%d/%m/%Y")
        data = {
            "time": now,
            "license_plate": list_text
        }

        json_data = json.dumps(data, ensure_ascii=False, indent=4)

        # save history
        try:
            if not os.path.exists(his_lp):
                os.makedirs(his_lp)

            output_path = os.path.join(his_lp, str(now) + ".txt")
            with open(output_path, "a", encoding="utf-8") as file:
                file.write(json_data + '\n')

        except OSError as e:
            pass


# def rule_recognition():

# def rule_detection():


# load model detection
yolo_LP_detect = torch.hub.load(
    'yolov5',
    'custom',
    path='/home/minhthanh/code_python/bien_so_xe/model/LP_detector_nano_61.pt',
    force_reload=True,
    source='local'
)
yolo_LP_detect.conf = 0.8
yolo_LP_detect.eval()

# save history
his_lp = "/home/minhthanh/code_python/biensoxe2/history"

# input video
video_path = "/home/minhthanh/Downloads/20240906_100008_0002e42ab6dad9cd.mp4"
vid = cv2.VideoCapture(video_path)
cnt = 0
while(True):
    
    ret, frame = vid.read()
    if cnt >= 115:
        if not ret:
            break

        frame_copy = frame.copy()
        list_plates = lp_detection(frame_copy)
        if list_plates is not None and len(list_plates)>0:

            for plate in list_plates:
                left = int(plate[0]) # xmin
                top = int(plate[1]) # ymin
                width = int(plate[2] - plate[0]) # xmax - xmin
                height = int(plate[3] - plate[1]) # ymax - ymin
                cv2.rectangle(frame, (left, top), (int(plate[2]), int(plate[3])), (0, 255, 0), 2)
                crop_img = frame_copy[top:top+height, left:left+width]

                # paddle ocr
                lp_recognition(crop_img)

        frame = cv2.resize(frame, (1000, 640))
        cv2.imshow("img", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cnt += 1

vid.release()
cv2.destroyAllWindows()
