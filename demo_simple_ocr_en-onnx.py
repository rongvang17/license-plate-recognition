#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import argparse
import os
import numpy as np
import pandas as pd
import imutils
import matplotlib.pyplot as plt
import pytesseract as pt
import easyocr
import plotly.express as px
import skimage.io

from ppocr_onnx.ppocr_onnx import PaddleOcrONNX
from shapely.geometry import Point, Polygon


# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('/home/minhthanh/code_python/bien_so_xe/model/LP_detector_nano_61.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

INPUT_WIDTH = 640
INPUT_HEIGHT = 640


def get_detections(img,net):

    # 1.CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc, max_rc, 3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(
        input_image,
        1/255,
        (INPUT_WIDTH, INPUT_HEIGHT),
        swapRB=False,
        crop=False
    )
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections


def non_maximum_supression(input_image,detections):
    
    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.5:
            class_score = row[5] # probability score of license plate
            if class_score > 0.5:
                cx, cy , w, h = row[0:4]
                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left, top, width, height])
                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # 4.2 NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.5,0.5)

    return boxes_np, confidences_np, index


def augment_image(bgr_image):
    # gray_crop_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    up_gray_crop_image = cv2.pyrUp(bgr_image)
    blur_img = cv2.GaussianBlur(up_gray_crop_image, (3, 3), 0)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    augment_img = cv2.filter2D(blur_img, -1, kernel)

    return augment_img


def extract_text(crop_image):
    global paddle_ocr_onnx
    augment_img = augment_image(crop_image)
    dt_boxes, rec_res, time_dict = paddle_ocr_onnx(augment_img)

    list_text = ""
    for rec in rec_res:
        list_text += rec[0]
    if list_text:
        list_text = list_text.replace("-", "").replace(".", "")
        print(list_text)

        return list_text


def drawings(image,boxes_np,confidences_np,index):
    # 5. Drawings
    global roi_polygon

    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]

        x_center, y_center = x + w/2, y + h/2
        point = Point(x_center, y_center)
        is_inside = roi_polygon.contains(point)

        if is_inside:
            crop_img = image[y:y+h, x:x+w]
            list_text = extract_text(crop_img)
            if list_text:
                cv2.putText(
                    image,
                    list_text,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2
                )
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)


# predictions flow with return result
def yolo_predictions(image, net):
    # step-1: detections
    input_image, detections = get_detections(image, net)

    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)

    # step-3: Extracting and Drawings
    result_img = drawings(image, boxes_np, confidences_np, index)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default='sample.jpg')

    parser.add_argument(
        "--det_model",
        type=str,
        default='./ppocr_onnx/model/det_model/en_PP-OCRv3_det_infer.onnx',
    )
    parser.add_argument(
        "--rec_model",
        type=str,
        default='./ppocr_onnx/model/rec_model/en_PP-OCRv3_rec_infer.onnx',
    )
    parser.add_argument(
        "--rec_char_dict",
        type=str,
        default='./ppocr_onnx/ppocr/utils/dict/en_dict.txt',
    )
    parser.add_argument(
        "--cls_model",
        type=str,
        default=
        './ppocr_onnx/model/cls_model/ch_ppocr_mobile_v2.0_cls_infer.onnx',
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true",
    )

    args = parser.parse_args()

    return args


class DictDotNotation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def get_paddleocr_parameter():
    paddleocr_parameter = DictDotNotation()

    # params for prediction engine
    paddleocr_parameter.use_gpu = False

    # params for text detector
    paddleocr_parameter.det_algorithm = 'DB'
    paddleocr_parameter.det_model_dir = './ppocr_onnx/model/det_model/en_PP-OCRv3_det_infer.onnx'
    paddleocr_parameter.det_limit_side_len = 960
    paddleocr_parameter.det_limit_type = 'max'
    paddleocr_parameter.det_box_type = 'quad'

    # DB parmas
    paddleocr_parameter.det_db_thresh = 0.3
    paddleocr_parameter.det_db_box_thresh = 0.6
    paddleocr_parameter.det_db_unclip_ratio = 1.5
    paddleocr_parameter.max_batch_size = 10
    paddleocr_parameter.use_dilation = True
    paddleocr_parameter.det_db_score_mode = 'fast'

    # params for text recognizer
    paddleocr_parameter.rec_algorithm = 'SVTR_LCNet'
    paddleocr_parameter.rec_model_dir = './ppocr_onnx/model/rec_model/en_PP-OCRv3_rec_infer.onnx'
    paddleocr_parameter.rec_image_shape = '3, 48, 320'
    paddleocr_parameter.rec_batch_num = 6
    paddleocr_parameter.rec_char_dict_path = './ppocr_onnx/ppocr/utils/dict/en_dict.txt'
    paddleocr_parameter.use_space_char = True
    paddleocr_parameter.drop_score = 0.5

    return paddleocr_parameter


def main():
    args = get_args()
    # image_path = args.image

    paddleocr_parameter = get_paddleocr_parameter()

    paddleocr_parameter.det_model_dir = args.det_model
    paddleocr_parameter.rec_model_dir = args.rec_model
    paddleocr_parameter.rec_char_dict_path = args.rec_char_dict
    paddleocr_parameter.cls_model_dir = args.cls_model

    paddleocr_parameter.use_gpu = args.use_gpu

    video_path = "/home/minhthanh/Downloads/20050117_001203_0001cc06a9eb1423.mp4"
    vid = cv2.VideoCapture(video_path)

    global paddle_ocr_onnx
    paddle_ocr_onnx = PaddleOcrONNX(paddleocr_parameter)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    global roi_polygon
    roi_points = [(0, height), (0,  2*height/4), (width, 2*height/4), (width, height)]
    roi_polygon = Polygon(roi_points)

    while(True):
        ret, frame = vid.read()
        # frame = cv2.imread("/home/minhthanh/Pictures/Screenshots/Screenshot from 2024-09-25 11-50-40.png")
        frame_copy = frame.copy()

        results = yolo_predictions(frame_copy, net)
        cv2.rectangle(frame_copy, (0, int(2*height/4)), (width, height), (0, 0, 255), 2)
        frame_copy = cv2.resize(frame_copy, (1000, 1000))

        cv2.imshow("img", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
