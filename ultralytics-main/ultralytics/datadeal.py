import os
import shutil
import cv2
import tqdm
import torch
def testcuda():
    print(torch.tensor([1.0,2.0]).cuda())

def practise_gt():
    test_list = []
    gt_img = cv2.imread(r'D:\Internship_C\about-yolov8\test.png',-1)
    h,w = gt_img.shape
    for i in range(0,h):
        for j in range(0,w):
            print(gt_img[i,j])
            if gt_img[i,j] not in test_list:
                test_list.append(gt_img[i,j])
    print(test_list)


def deal_data(path):
    for root, dir, files in os.walk(path):
        for file in files:
            File = os.path.join(root, file)
            if File[-4:] != 'json':
                print(File)
                os.remove(File)


if __name__ == '__main__':
    practise_gt()
    # path = r'D:\Internship_C\about-yolov8\cityscapes\labels'
    # testcuda()
    # deal_data(path)
    # deal_data(path)
