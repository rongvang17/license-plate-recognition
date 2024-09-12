from PIL import Image
import cv2
import torch
import math 
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper
import easyocr
import numpy as np
import shutil

# from paddleocr import PaddleOCR

# ocr = PaddleOCR(use_gpu=False, lang='vi', dilation=True,
#                 det_db_box_thresh=0.5, det_limit_side_len=2200, use_dilation=True, 
#                 det_east_nms_thresh=0.6, det_sast_nms_thresh=0.6) 

def agument_image(bgr_image):

    gray_crop_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    up_gray_crop_image = cv2.pyrUp(gray_crop_image)
    blur_img = cv2.GaussianBlur(up_gray_crop_image,(3,3), 0)
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    agument_img = cv2.filter2D(blur_img, -1, kernel)

    return agument_img


reader = easyocr.Reader(['en'], gpu=False) # read text by use Easyocr
# load model
# yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/best.pt', force_reload=True, source='local')
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_LP_detect.conf = 0.5
yolo_LP_detect.eval()
yolo_license_plate.conf = 0.5
yolo_license_plate.eval()

data_path = "/home/minhthanh/code_python/bien_so_xe/val_ocr/all_img"
exception_path = "/home/minhthanh/code_python/bien_so_xe/val_ocr/exception"
matching_path = "/home/minhthanh/code_python/bien_so_xe/val_ocr/matching"
# os.mkdir(exception_path)
num_img = len(os.listdir(data_path))
num_match = 0
num_except = 0

for img in os.listdir(data_path):
    if img.endswith(".png"):
        img_path = os.path.join(data_path, img)
        input_img = cv2.imread(img_path)
        flag = 0

        # east ocr model
        # agu_img = agument_image(input_img)
        # result = reader.readtext(input_img, detail=1, paragraph=True)
        # if len(result)>0 and result is not None:
        #     # print("text:", result)
        #     _, text = result[0]
        #     print("text:", text)
        #     if text==text_predict:
        #         pre_rec += 1
        #     # resize_img = cv2.resize(input_img, (1000, 1000))
        #     cv2.putText(input_img, "my:" + text, (500, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        #     cv2.putText(input_img, "other:" + text_predict, (448, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 4)

        # custom model
        lp = ""
        for cc in range(0,2):
            for ct in range(0,2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(input_img, cc, ct))
                if lp != "unknown":
                    # print("other model:", text_predict, "my model:", lp)
                    flag = 1
                    break

            if flag == 1:
                break
        
        if lp!="unknown":
            num_match += 1
            input_img = cv2.resize(input_img, (800, 800))
            cv2.putText(input_img, lp, (500, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            label = "{}_{}.png".format(num_match, lp)
            target_matching_path = os.path.join(matching_path, label)
            cv2.imwrite(target_matching_path, input_img)
            cv2.imshow('resize_img', input_img)
        elif lp=="unknown" :
            num_except += 1
            text = os.path.splitext(img)[0]
            label = "{}.png".format(text)
            target_exception_path = os.path.join(exception_path, label)
            cv2.imwrite(target_exception_path, input_img)
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("num_match:", num_match)
print("num_except:", num_except)

"""
while(True):
    ret, frame = vid.read()
    frame_copy = frame
    
    # frame = cv2.imread("/home/minhthanh/Downloads/Dataset 2 - at CTHPHN - by BinhNT (5).png")
    # frame_copy = frame
    plates = yolo_LP_detect(frame)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    # print(list_plates)
    list_read_plates = set()
    for plate in list_plates:
        flag = 0
        x = int(plate[0])
        y = int(plate[1])
        w = int(plate[2] - plate[0])
        h = int(plate[3] - plate[1]) 
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame_copy, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)

        # paddle ocr
        # agu_img = agument_image(crop_img)
        # result = ocr.ocr(agu_img, cls=False)

        # if result[0] == None:
        #     print("no find text in image")

        # else:
        #     result_length = sum(len(item) for item in result)
        #     list_box = []

        #     for index in range(result_length):
        #         boxes = [line[index] for line in result]
        #         print(boxes)
        #         text = boxes[0][1][0]
        #         cv2.putText(frame_copy, text, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 3)
                # print("text:", text)

        # custom model
        lp = ""
        for cc in range(0,2):
            for ct in range(0,2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    print("lp:", lp)
                    list_read_plates.add(lp)
                    cv2.putText(frame, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 3)
                    flag = 1
                    break
            if flag == 1:
                break

        # east ocr model
        # agu_img = agument_image(crop_img)
        # result = reader.readtext(agu_img, detail=1, paragraph=True)
        # if len(result)>0 and result is not None:
        #     print("text:", result)
        #     _, text = result[0]
        #     cv2.putText(frame_copy, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    #cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    frame_copy = cv2.resize(frame_copy, (1920, 1080))
    cv2.imshow('frame_copy', frame_copy)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
"""
# vid.release()
cv2.destroyAllWindows()