import numpy as np
import os
from PIL import Image
import torch
import cv2
# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):  
    print('Num classes', num_classes)  
    #-----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    
    #------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    #------------------------------------------------#
    # gt_imgs     = [os.path.join(gt_dir, x + ".png") for x in png_name_list]
    gt_imgs = [r'D:\Internship_C\about-yolov8\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_labelTrainIds.png']
    # pred_imgs   = [os.path.join(pred_dir, x + ".png") for x in png_name_list]
    pred_imgs = [r'D:\Internship_C\about-yolov8\test.png']


    #------------------------------------------------#
    #   读取每一个（图片-标签）对
    #------------------------------------------------#
    for ind in range(len(gt_imgs)): 
        #------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))  

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        #------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(),num_classes)  
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if ind > 0 and ind % 10 == 0:  
            print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(ind, len(gt_imgs),
                                                    100 * np.nanmean(per_class_iu(hist)),
                                                    100 * np.nanmean(per_class_PA(hist))))
    #------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    #------------------------------------------------#
    mIoUs   = per_class_iu(hist)
    mPA     = per_class_PA(hist)
    #------------------------------------------------#
    #   逐类别输出一下mIoU值
    #------------------------------------------------#
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tmIou-' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA-' + str(round(mPA[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))  
    return mIoUs

if __name__ == '__main__':
    map_dict = {
        0:'road',
        1:'sidewalk',
        2:'building',
        3: 'wall',
        4: 'fence',
        5: 'pole',
        6: 'traffic light',
        7: 'traffic sign',
        8: 'vegetation',
        9: 'terrain',
        10: 'sky',
        11: 'person',
        12: 'rider',
        13: 'car',
        14: 'truck',
        15: 'bus',
        16: 'train',
        17: 'motorcycle',
        18: 'bicycle',
        255: 'none'
}
    name_classes = []
    gt_dir = ''
    pred_dir = ''
    png_name_list = []
    gt_imgs = [r'D:\Internship_C\about-yolov8\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_labelTrainIds.png']
    # num_classes = 19
    # for i in gt_imgs:
    num_classes = len(torch.tensor(np.array(cv2.imread(r'D:\Internship_C\about-yolov8\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_labelTrainIds.png'))).unique())
    pixel_list = torch.tensor(np.array(cv2.imread(r'D:\Internship_C\about-yolov8\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_labelTrainIds.png'))).unique()
    print('pixel_list:',list(np.array(pixel_list)))
    pixel_list = list(np.array(pixel_list))
    for i in pixel_list:
        if map_dict[i] not in name_classes:
            name_classes.append(map_dict[i])
    print(len(name_classes),name_classes)
    # print('num_classes:',num_classes)

    
    # name_classes = ['road','sidewalk','building','wall','fence','pole','traffic','light','traffic' 'sign','vegetation','terrain',
    #                 'sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
    compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes)