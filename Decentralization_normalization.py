'''
some data processing on decentralization and normalization in the field of object detection  
author:Chenlin Zhou
time:2019/12/3
'''
import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt

imput_path = '/home/zcl/PycharmProjects/So-yolov3/venv/guass_data/train/train_left4.txt'
txt_path = '/home/zcl/PycharmProjects/So-yolov3/venv/guass_data/train/weld_center.txt'
nor_path = '/home/zcl/PycharmProjects/So-yolov3/venv/guass_data/train/normallation.txt'

def parse_annotation(annotation):
    line = annotation.split()
    image_path = line[0]
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " % image_path)
    #image = np.array(cv2.imread(image_path))
    bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])
    return image_path, bboxes

def parse_annotation_1(annotation):
    line = annotation.split()
    image_path = line[0]
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " % image_path)
    #image = np.array(cv2.imread(image_path))
    bboxes = np.array([list(map(float, box.split(','))) for box in line[1:]])
    return image_path, bboxes

def weld_center():
    # 获取中心坐标
    write_list = []
    with open(imput_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        for j in range(len(annotations)):
            image_path, bboxes = parse_annotation(annotations[j])
            write_list.append(image_path + ' ')
            weld_center_x = 0
            weld_center_y = 0
            for i in range(len(bboxes)):
                write_list.append(str(int((bboxes[i][0] + bboxes[i][2]) / 2)) + ',')
                write_list.append(str(int((bboxes[i][1] + bboxes[i][3]) / 2)) + ' ')
                weld_center_x = weld_center_x + (bboxes[i][0] + bboxes[i][2]) / 2
                weld_center_y = weld_center_y + (bboxes[i][1] + bboxes[i][3]) / 2
                # write_list.append(' ')
            write_list.append(str(int(weld_center_x / len(bboxes))) + ',' + str(int(weld_center_y / len(bboxes))) + ' ')
            write_list.append('\n')
        file = open(txt_path, 'w')
        for annotation in write_list:
            file.write(str(annotation))
        file.close()
    f.close()


def normalization():
    # 归一化后的坐标
    write_List = []
    with open(txt_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        for j in range(len(annotations)):
            image_path, bboxes = parse_annotation(annotations[j])
            write_List.append(image_path + ' ')
            r_max = 0
            for i in range(len(bboxes) - 1):
                x_ = int(bboxes[i][0] - bboxes[-1][0])
                y_ = int(bboxes[i][1] - bboxes[-1][1])
                if r_max < np.sqrt(x_ * x_ + y_ * y_):
                    r_max = np.sqrt(x_ * x_ + y_ * y_)

            for i in range(len(bboxes) - 1):
                x_ = int(bboxes[i][0] - bboxes[-1][0])
                y_ = int(bboxes[i][1] - bboxes[-1][1])
                x_1 = float(x_) / r_max
                y_1 = float(y_) / r_max
                write_List.append(str(float("{0:.4f}".format(x_1))) + ',')
                write_List.append(str(float("{0:.4f}".format(y_1))) + ' ')
            write_List.append('\n')
        file = open(nor_path, 'w')
        for annotation in write_List:
            file.write(str(annotation))
        file.close()
    f.close()

def visualization():
    with open(nor_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        for j in range(len(annotations)):
            image_path, bboxes = parse_annotation_1(annotations[j])
            for i in range(len(bboxes)):
                plt.scatter(bboxes[i][0], bboxes[i][1], 10, 'r')
        plt.show()

if __name__ == '__main__':
    weld_center()
    normalization()
    visualization()
