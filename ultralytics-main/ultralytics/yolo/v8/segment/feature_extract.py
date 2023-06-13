import sys
sys.path.append(r"D:\Internship_C\about-yolov8\ultralytics-main")
import numpy as np
from PIL import Image
from ultralytics import YOLO
from PIL import Image
from ultralytics.yolo.v8.segment import val
import cv2
import torch
import os

# gt_img_path:类别掩码图片地址 
# ori_img_path待预测图像地址
# model_path：预测模型地址
def index_val():
    val()


def evaluate(gt_img_path = r'D:\Internship_C\about-yolov8\gtFine\val\frankfurt\frankfurt_000001_031266_gtFine_labelTrainIds.png',
                ori_img_path=r"D:\Internship_C\about-yolov8\leftImg8bit\val\frankfurt\frankfurt_000001_031266_leftImg8bit.png",
                model_path=r"D:\Internship_C\about-yolov8\ultralytics-main\results\612results12\weights\best.pt"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    gt_img = cv2.imread(gt_img_path,-1)
    gt_cls = torch.tensor(gt_img)
    im1 = Image.open(ori_img_path)
    model = YOLO(model=model_path) # 19类的
    MIOU=0
    results = model.predict(source=im1, save=True,boxes=True,show=False)
    for result in results:
        #单张图片的处理
        s = 0
        PA = 0
        clsboxes= result.boxes.cls
        cls2mask={}
        for i in gt_cls.unique():
            cls2mask[int(i)]=np.array(torch.zeros((1024, 2048)),dtype='uint8')
        for i,mask in enumerate(result.masks.data):
            curclass = int(clsboxes[i])
            if curclass in list(gt_cls.unique()):  
                cls2mask[curclass]=np.bitwise_or(np.array(cls2mask[curclass]),np.array(mask.cpu(),dtype='uint8'))
            cls2mask[255]=np.bitwise_or(np.array(cls2mask[255]),np.array(mask.cpu(),dtype='uint8'))
        res_dict = {}
        cls2mask[255]=np.bitwise_not(cls2mask[255])
        for cls_,cls_mask in cls2mask.items():
            curgt=np.where(gt_img==cls_,1,0)
            PA += np.count_nonzero(np.bitwise_and(cls_mask,curgt))
            res_dict[cls_] =  np.count_nonzero(np.bitwise_and(cls_mask,curgt)) / np.count_nonzero(np.bitwise_or(cls_mask,curgt))
            s += res_dict[cls_]
        for value in list(res_dict.values()):
            MIOU+=value

        MIOU/=len(gt_cls.unique())
    
    s=s / len(gt_cls.unique())
    PA=PA / (1024 * 2048)
    print('\n----------iou per class----------:\n',res_dict)
    print('----------all_class-miou----------:\n',s )
    print('----------PA----------:\n:',PA )
    return MIOU,PA

    
if __name__ == '__main__':
    # evaluate()
    erropicture=[]
    PAbiggerthan1=[] 
    MIOUlist=[]
    MIOU=0
    PA=0
    valdir=r"D:\Internship_C\about-yolov8\gtFine\val"
    oripicdir=r"D:\Internship_C\about-yolov8\leftImg8bit\val"
    count=0
    for city in os.listdir(valdir):
        citydir=os.path.join(valdir,city)
        for file in os.listdir(citydir):
            if file.endswith('_labelTrainIds.png'):
                gtpath=os.path.join(citydir,file)
                oripicpath=os.path.join(oripicdir,city,file.replace('gtFine_labelTrainIds.png','leftImg8bit.png'))
                print('Current gtpath',gtpath)
                print('Current oripicpath',oripicpath)
                try:
                    curMIOU,curPA=evaluate(gt_img_path=gtpath,ori_img_path=oripicpath)
                    MIOU+=curMIOU
                    MIOUlist.append(curMIOU)
                    PA+=curPA
                except:
                    erropicture.append(oripicpath)
                    raise KeyError
                else:
                    count+=1
            if count==1:
                break
        if count==1:
            break
    print("Total number of the pictures has been validated:",count)
    try:
        print("\nMIOU: ",MIOU/count)
        print('\nmPA ',PA/count)
        with open('MIOU_and_PA.txt','w+') as f:
             f.write(str(MIOU/count)+' '+str(PA/count)+'\n')
    except:
        print("Error!")
    with open('ErroPicture.txt','w+') as f:
        for i in erropicture:
            f.write("\n"+i)
    print('Erro Pictures: ',erropicture)
    print('Erro Pictures length: ',len(erropicture))
    print('PAbiggerthan1: ',PAbiggerthan1)
    print('PAbiggerthan1 length: ',len(PAbiggerthan1))
    print(MIOUlist)
    index_val()
    print(r"mIOU and PA have been saved in D:\Internship_C\about-yolov8\MIOU_and_PA.txt")