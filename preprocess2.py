import os
import random
import shutil
from os import listdir
import json
import sys

# get the path of the directory that contains the images
ROOT = sys.argv[1]
if ROOT[-1] == "/":
    ROOT = ROOT[:-1]
dataset_name = 'railway'
if 'catenary' in ROOT:
    dataset_name = 'catenary'

TARGET_PATH = ROOT


width = 1920
height = 1080


# create data.yaml file in TARGET_PATH
'''
category = list()
for mode in ["train_json", "val_json", "test_json"]:
    mypath = TARGET_PATH + "/" + mode + "/"
    files = [f for f in listdir(mypath)]
    for file in files:
        with open(mypath + file, 'r') as curr_file:
            class_curr = json.load(curr_file)
        class_categories = class_curr['annotations']
        for i in range(len(class_categories)):
            class_id = class_categories[i]['category_id']
            class_status = class_categories[i]['status']
            if class_status != 'normal' and class_status != 'abnormal':
                continue
            class_name = class_curr['categories'][class_id-1]['supercategory'] + '_' + class_curr['categories'][class_id-1]['name'] + '-' + class_status
            if class_name not in category:
                category.append(class_name)

category = list(set(category))
num_classes = len(category)
'''

category_info = []
category = []
if dataset_name == 'railway':
    with open('railway_metadata.json', 'r') as curr_file:
        category_info = json.load(curr_file)['categories']
else:
    with open('catenary_metadata.json', 'r') as curr_file:
        category_info = json.load(curr_file)['categories']

for obj in category_info:
    class_name = obj['supercategory'] + '_' + obj['name']
    category.append(class_name + '-normal')
    category.append(class_name + '-abnormal')

print(category)
#category = list(set(category))
num_classes = len(category)


cat = dict()
for i in range(len(category)):
    cat[str(category[i])] = i

print(cat)

with open(os.path.join(TARGET_PATH, "data.yaml"), 'w') as f:
    f.write("path: " + TARGET_PATH)
    f.write("\n\n")
    f.write("train: train")
    f.write("\n")
    f.write("val: val")
    f.write("\n")
    f.write("test: test")
    f.write("\n\n")
    f.write("nc: " + str(num_classes))
    f.write("\n")
    f.write("names: " + str(category))

    f.close()


os.makedirs(os.path.join(TARGET_PATH, "annotations"), exist_ok=True)

for mode in ["train", "val", "test"]:

    mypath = TARGET_PATH + "/" + mode + "/"
    mypath_label = TARGET_PATH + "/" + mode + "_json/"
    files = [f for f in listdir(mypath)]
    files_without_suffix = [f.split(".")[0] for f in files]

    labeling = {}
    for i in range(len(files_without_suffix)):
        labeling[files_without_suffix[i]] = i


    categories = list()

    for key in cat:
        temp_now = {
            "id": cat[key],
            "name": key,
            "supercategory": "none"
        }
        categories.append(temp_now)

    #width = width
    #height = height
    

    images = list()
    for file in files_without_suffix:
        temp_dict = {
            "id": labeling[file],
            "file_name": file + '.jpg',
            "height": height,
            "width": width
        }

        images.append(temp_dict)


    annotations = list()
    counter = 0
    for file in files_without_suffix:
        image_id = labeling[file]
        segmentation = []
        iscrowd = 0
        with open(mypath_label + file + '.json', 'r') as curr_file:
            curr = json.load(curr_file)
        width = curr['metadata']['width']
        height = curr['metadata']['height']
        if width != 1920 or height != 1080:
            print(width, height, "wrong resolution")
        list_img = curr['annotations']
        for i in range(len(list_img)):
            curr_img = list_img[i]
            if curr_img['polygon'] == []:
                id = curr_img['category_id']
                status = curr_img['status']
                if status != 'normal' and status != 'abnormal':
                    continue
                
                #name = curr['categories'][id-1]['supercategory'] + '_' + curr['categories'][id-1]['name'] + '-' + status
                
                name = ''
                for x in curr['categories']:
                    if x['id'] == id:
                        name = x['supercategory'] + '_' + x['name'] + '-' + status
                        break


                if name not in cat:
                    print('ERROR')
                    print(file)
                    print('ERROR')

                
                
                category_id = cat[name]
                curr_bbox = curr_img['bbox']
                bbox = [curr_bbox[0], curr_bbox[1], curr_bbox[2], curr_bbox[3]]
                
                bbox[0] = max(bbox[0], 0)
                bbox[0] = min(bbox[0], width)
                bbox[1] = max(bbox[1], 0)
                bbox[1] = min(bbox[1], height)
                bbox[2] = max(bbox[2], 0)
                bbox[2] = min(bbox[2], width)
                bbox[3] = max(bbox[3], 0)
                bbox[3] = min(bbox[3], height)

                area = bbox[2]*bbox[3]

                temp_dict = {
                        "id": counter,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "segmentation": segmentation,
                        "iscrowd": iscrowd
                    }
                annotations.append(temp_dict)
            else:
                if False:
                    id = curr_img['category_id']
                    status = curr_img['status']
                    if status != 'normal' and status != 'abnormal':
                        continue
                    #name = curr['categories'][id-1]['supercategory'] + '_' + curr['categories'][id-1]['name'] + '-' + status

                    name = ''
                    for x in curr['categories']:
                        if x['id'] == id:
                            name = x['supercategory'] + '_' + x['name'] + '-' + status
                            break
                
                    if name not in cat:
                        print('ERROR')
                        print(file)
                        print('ERROR')
                    
                    category_id = cat[name]
                    curr_segm = curr_img["polygon"]

                    n = len(curr_segm)
                    coor_x = []
                    coor_y = []
                    for i in range(n):
                        if i % 2 == 0:
                            coor_x.append(curr_segm[i])
                        else:
                            coor_y.append(curr_segm[i])
                    
                    x_min = max(min(coor_x), 0)
                    x_max = min(max(coor_x), width)
                    y_min = max(min(coor_y), 0)
                    y_max = min(max(coor_y), height)

                    w = (x_max-x_min)
                    h = (y_max-y_min)
                    area = w*h
                    bbox = [x_min, y_min, w, h]


                    temp_dict = {
                        "id": counter,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "segmentation": segmentation,
                        "iscrowd": iscrowd
                    }
                    annotations.append(temp_dict)
            counter += 1

    main = {
        "info" : {}, 
        "licenses": {}, 
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(TARGET_PATH + '/annotations/instances_'+ mode +'.json', "w", encoding='utf-8') as outfile:
        json.dump(main, outfile, ensure_ascii=False)
