# -*- coding: utf-8 -*-
import json
import os
import argparse
from tqdm import tqdm


def convert_label_json(json_dir, save_dir):
    label_list = ['trailer','caravan','polegroup','tunnel','bridge','guard rail','rail track','parking','ground','dynamic','static','out of roi','rectification border','ego vehicle','unlabeled','bicyclegroup','cargroup', 'persongroup', 'ridergroup', 'motorcyclegroup', 'truckgroup','license plate']

    class_dict = {
    'road':0,
    'sidewalk':1,
    'building':2,
    'wall':3,
    'fence':4,
    'pole':5,
    'traffic light':6,
    'traffic sign':7,
    'vegetation':8,
    'terrain':9,
    'sky':10,
    'person':11,
    'rider':12,
    'car':13,
    'truck':14,
    'bus':15,
    'train':16,
    'motorcycle':17,
    'bicycle':18,
    }

    classeslist = []
    json_paths = os.listdir(json_dir)
    # classes = classes.split(',')
    PATH = 'D:/Internship_C/about-yolov8/cityscapes/labels'
    for root,dir,files in os.walk(PATH):
        for file in files:
            jsonpath = os.path.join(root,file)
            with open(jsonpath,'r') as load_f:
                json_dict = json.load(load_f)
            h, w = json_dict['imgHeight'], json_dict['imgWidth']
            txt_path = jsonpath.replace('json', 'txt')
            txt_path = txt_path.replace('gtFine_polygons', 'leftImg8bit')
            print('txt:',txt_path)
            print('json:',jsonpath)
            txt_file = open(txt_path, 'w')

            for shape_dict in json_dict['objects']:
                label = shape_dict['label']
                if label in label_list:
                    continue
                points = shape_dict['polygon']
                points_nor_list = []
                for point in points:
                    points_nor_list.append(point[0] / w)
                    points_nor_list.append(point[1] / h)

                points_nor_list = list(map(lambda x: str(x), points_nor_list))
                points_nor_str = ' '.join(points_nor_list)

                label_str = str(class_dict[label]) + ' ' + points_nor_str + '\n'
                txt_file.writelines(label_str)



    # for json_path in tqdm(json_paths):
    #     # for json_path in json_paths:
    #     path = os.path.join(json_dir, json_path)
    #     with open(path, 'r') as load_f:
    #         json_dict = json.load(load_f)
    #     h, w = json_dict['imgHeight'], json_dict['imgWidth']
    #
    #     # save txt path
    #     txt_path = os.path.join(save_dir, json_path.replace('json', 'txt'))
    #     txt_path = txt_path.replace('polygons', 'labelIds')
    #     txt_file = open(txt_path, 'w')
    #
    #     for shape_dict in json_dict['objects']:
    #         label = shape_dict['label']
    #         if label in label_list:
    #             continue
    #         points = shape_dict['polygon']
    #         points_nor_list = []
    #         for point in points:
    #             points_nor_list.append(point[0] / w)
    #             points_nor_list.append(point[1] / h)
    #
    #         points_nor_list = list(map(lambda x: str(x), points_nor_list))
    #         points_nor_str = ' '.join(points_nor_list)
    #
    #         label_str = str(class_dict[label]) + ' ' + points_nor_str + '\n'
    #         txt_file.writelines(label_str)


if __name__ == "__main__":
    """
    python json2txt_nomalize.py --json-dir my_datasets/color_rings/jsons --save-dir my_datasets/color_rings/txts --classes "cat,dogs"
    """
    parser = argparse.ArgumentParser(description='json convert to txt params')
    parser.add_argument('--json-dir', type=str, default='D:/Internship_C/about-yolov8/cityscapes/labels/train',
                         help='json path dir')
    parser.add_argument('--save-dir', type=str, default='D:/Internship_C/about-yolov8/cityscapes/labels/train',
                         help='txt save dir')
    #parser.add_argument('--classes', type=str, default='road,sky,sidewalk,parking,building,static,pole,dynamic,person,car,vegetation,bicycle,traffic light,rider,terrain,license plate,ego vehicle,out of roi,wall,fence,traffic sign,ground,bicyclegroup', 'cargroup', 'trailer', 'rectification border', 'truck', 'bus', 'motorcycle', 'rail track', 'persongroup', 'bridge', 'guard rail', 'polegroup', ridergroup, tunnel, train, motorcyclegroup, caravan, truckgroup', help='classes')
    args = parser.parse_args()
    json_dir = args.json_dir
    save_dir = args.save_dir
    # classes = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road',
    #            'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
    #            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider',
    #            'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']
    convert_label_json(json_dir, save_dir)
