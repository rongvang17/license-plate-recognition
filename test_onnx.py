import os
import cv2
import numpy as np
import pandas as pd
import imutils
import matplotlib.pyplot as plt
import pytesseract as pt
import easyocr
import plotly.express as px
import skimage.io

INPUT_WIDTH =  300
INPUT_HEIGHT = 300

# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('/home/minhthanh/Downloads/az_ocr_ssdmobilenetv1_2.onnx')
# net = cv2.dnn.readNetFromONNX('/home/minhthanh/code_python/bien_so_xe/model/LP_detector_nano_61.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def get_detections(img,net):

    # 1.CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    print("preds:", preds)
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
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.1:
                cx, cy , w, h = row[0:4]
                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])
                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist() 
    confidences_np = np.array(confidences).tolist()
    
    # 4.2 NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.15,0.45)
    
    return boxes_np, confidences_np, index

def drawings(image,boxes_np,confidences_np,index):
    # 5. Drawings
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(0,255,0),-1)
        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        
    return image

# predictions flow with return result
def yolo_predictions(img,net):
    # step-1: detections
    input_image, detections = get_detections(img,net)
    print(detections)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    result_img = drawings(img,boxes_np,confidences_np,index)

    global box_coordinate
    global nm_index
    box_coordinate = boxes_np
    nm_index = index

    return result_img

def croptheROI(image,bbox, index):
     
    cropped = None
    for i in index:   
        x,y,w,h =  bbox[i]
        cropped = image[y:y+h, x:x+w]
        cv2.imwrite('cropped.png',cropped)

    return cropped

def preprocessing(crop):

    # crop = cv2.imread('cropped.png')
    if crop is None:
        return crop

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) # Edge detection

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour,10, True)
        if len(approx)==4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(crop, crop, mask=mask)

    (x,y) = np.where(mask==255)
    (x1,y1) = (np.min(x), np.min(y))
    (x2,y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    return cropped_image

def extract_text(cropped_image):
    
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    return result

# test
# img = skimage.io.imread('/home/minhthanh/Downloads/458499894_736778238574353_6456774566039528894_n.png')
global box_coordinate
global nm_index

video_path = "/home/minhthanh/Downloads/20240906_100008_0002e42ab6dad9cd.mp4"

vid = cv2.VideoCapture(video_path)
while(True):
    # ret, frame = vid.read()
    # frame_copy = frame
    frame = cv2.imread("/home/minhthanh/Pictures/Screenshots/Screenshot from 2024-09-10 14-00-44.png")
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = yolo_predictions(frame,net)
    frame_copy = cv2.resize(frame, (832, 832))
    cv2.imshow("img", frame_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # roi = croptheROI(img,box_coordinate,nm_index)
    # pp_image = preprocessing(roi)
    # text = extract_text(pp_image)
    # print(text)
    # fig = px.imshow(frame)
    # fig.update_layout(width=700, height=400, margin=dict(l=10, r=10, b=10, t=10))
    # fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    # fig.show()

# import plotly.io as ip
# ip.renderers.default='browser'
