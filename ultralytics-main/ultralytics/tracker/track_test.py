import sys
sys.path.append(r"D:\Internship_C\about-yolov8\ultralytics-main")
import cv2 
from ultralytics import YOLO
import time

from ultralytics import YOLO

model = YOLO(r"D:\Internship_C\about-yolov8\ultralytics-main\results\612Seg_Model\weights\best.pt")  # or a segmentation model .i.e yolov8n-seg.pt
model.track(
    source=r"C:\Users\admin\Desktop\test.mp4",
    stream=False,
    tracker="bytetrack.yaml",  # or 'bytetrack.yaml'
    show=True,
)

# cap = cv2.VideoCapture(r"C:\Users\admin\Desktop\test2.mp4")
# # model = YOLO(r"D:\Internship_C\about-yolov8\ultralytics-main\results\68results4\68results5\weights\best.pt")
# model = YOLO(r"D:\Internship_C\about-yolov8\ultralytics-main\results\612Seg_Model\weights\best.pt")
# id = 0
# while True:
#     ret, frame = cap.read()
#     print(type(frame),frame)
#     results = model.track(source=frame,tracker="bytetrack.yaml",show=True,stream=False)
