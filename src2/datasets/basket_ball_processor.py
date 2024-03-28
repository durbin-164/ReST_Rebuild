import json
import os
from collections import defaultdict

import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

root = "/home/masud.rana/Documents/Learning_Project/Important/Re_ID_Basket_Ball/Other_Code/ReST/datasets/basket_ball/sequence1"

video_1_path = f"{root}/src/10sec_Sim0L.mp4"
video_2_path = f"{root}/src/10sec_Sim0R.mp4"

cap1 = cv2.VideoCapture(video_1_path)
cap2 = cv2.VideoCapture(video_2_path)

output_path = f"{root}/output"
frames_save_path = f"{output_path}/frames"
json_save_path = f"{output_path}"

frames = defaultdict(list)
frame_count = 0
cam_id0 = 0
cam_id1 = 1
while cap1.isOpened() and cap2.isOpened():
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()

    if not ret1 or not ret2:
        break

    result1 = model.predict(img1, classes=[0])[0]
    result2 = model.predict(img2, classes=[0])[0]

    boxes = result1.boxes.cpu().numpy()
    for box in boxes.xyxy:
        x1, y1, x2, y2 = tuple(map(int, box))
        frames[frame_count].append([x1, y1, x2-x1, y2-y1, cam_id0])

    boxes = result2.boxes.cpu().numpy()
    for box in boxes.xyxy:
        x1, y1, x2, y2 = tuple(map(int, box))
        frames[frame_count].append([x1, y1, x2 - x1, y2 - y1, cam_id1])

    cv2.imwrite(f"{frames_save_path}/{frame_count}_{cam_id0}.jpg", img1)
    cv2.imwrite(f"{frames_save_path}/{frame_count}_{cam_id1}.jpg", img2)

    frame_count += 1

with open(os.path.join(output_path, f'yolo_infer.json'), 'w') as fp:
    json.dump(frames, fp)

cap1.release()
cap2.release()
