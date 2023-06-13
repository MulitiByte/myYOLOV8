# 训练结果save在Internship/ultralytics-main/ultralytics/results/results2中  results中为预测结果。


# 5.30
# BUG记录
# 1、cityscapes数据集的图片和标签名称是不一样的......这是个粗心的问题  （解）
# 2、WARNING ⚠️ /home/issiyua/workspace/Datasets/images/train/aachen/aachen_000137_000019_leftImg8bit.png: ignoring corrupt image/label: negative label values [ -1   -1   -1]
# 3、train: Scanning /home/issiyua/workspace/Datasets/labels/train/aachen.cache... 2975 images, 0 backgrounds, 2065 corrupt: 100%|██████████| 2975/2975 [00:00<?, ?it/s] #这个问题没解决
# 4、RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 71 but got size 72 for tensor number 1 in the list. #这个问题为什么把batchsize调成4就解决了？
    # 用 cfg = defualt_copy.yaml就会报错 （why？？？）
#            答: save_hybrid: False 就不会报4的错误了。
# 5这个问题 是因为label里有-1 但是给出的分类 就是有-1 不知道怎么解决这个问题（我直接把-1这个屏蔽掉了，即license plate没有参与训练)
# 训练机器: Ultralytics YOLOv8.0.110 🚀 Python-3.7.15 torch-1.13.1+cu116 CUDA:0 (NVIDIA GeForce RTX 3080, 10009MiB)

# 5.31
# 对网络结构增加CBAM（结合
# Internship/ultralytics-main/ultralytics/models/v8/yolov8-seg_ATT.yaml