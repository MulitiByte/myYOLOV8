from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO(r"D:\Internship_C\about-yolov8\ultralytics-main\results\best.pt")
# 接受所有格式-image/dir/Path/URL/video/PIL/ndarray。0用于网络摄像头
# results = model.predict(source=r"D:\Internship_C\about-yolov8\ultralytics-main\ultralytics\assets\bus.jpg")
# results = model.predict(source="folder", show=True) # 展示预测结果

# from PIL
im1 = Image.open("D:/Internship_C/about-yolov8/ultralytics-main/ultralytics/assets/bus.jpg")
    # 實現特定的class不顯示bounding_box

results = model.predict(source=im1, save=True,boxes=True)  # 保存绘制的图像


# for result in results:
#     print('result[0].boxes.cls:',result[0].boxes.cls)
#     # print(result[0].boxes)
#     print(result[0].boxes)
# # from ndarray
# im2 = cv2.imread("bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # 将预测保存为标签

# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])
