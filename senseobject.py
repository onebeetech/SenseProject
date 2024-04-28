import cv2 as cv
import time
import requests
import json
import qrcode
from io import BytesIO
from datetime import datetime

Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

cap = cv.VideoCapture(0)  # 0 for the first webcam, 1 for the second, and so on

starting_time = time.time()
frame_counter = 0

# Prices per unit
price_orange = 300 / 1000  # Price per gram for orange (Rs)
price_apple = 200 / 1000   # Price per gram for apple (Rs)

api_endpoint = 'http://localhost:5000/'

def generate_qr_code(data):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img_io = BytesIO()
    img.save(img_io, format='PNG')
    img_io.seek(0)
    return img_io

def format_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

while True:
    ret, frame = cap.read()
    frame_counter += 1

    if not ret:
        break
    
    total_objects = 0  # Variable to count total objects detected in the frame
    highest_score = 0  # Variable to keep track of highest score for each fruit
    
    for (classid, score, box) in zip(*model.detect(frame, Conf_threshold, NMS_threshold)):
        # Skip processing if not a fruit
        if class_name[classid] not in ['apple', 'orange']:
            continue
        
        total_objects += 1
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid], score)
        cv.rectangle(frame, box, color, 1)
        cv.putText(frame, label, (box[0], box[1]-10),
                   cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
        
        # Check if the current score is higher than the previous one for the same fruit
        if score > highest_score:
            highest_score = score
            last_detected_fruit = class_name[classid]
            last_detected_score = float(score)  # Convert to regular Python float
            last_detected_box = box
            
    # If a fruit is detected, send data to Flask API
    if total_objects > 0:
        timestamp = time.time()
        data = {
            'data': {
                'fruit': last_detected_fruit,
                'price': 150 * price_orange if last_detected_fruit == 'orange' else 150 * price_apple,
                'timestamp': format_timestamp(timestamp),
                'score': last_detected_score
            }
        }
        response = requests.post(api_endpoint, json=data)
        print("Response from API:", response.text)
    
    ending_time = time.time() - starting_time
    fps = frame_counter / ending_time
    cv.putText(frame, f'FPS: {fps:.2f}', (20, 20),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
