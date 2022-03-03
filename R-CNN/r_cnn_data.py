import os
import glob
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# image와 annotation 데이터 로드
def get_data(path, cut_number):
    img_files = glob.glob(path+'/*.jpg')[:cut_number]
    ano_files = glob.glob(path+'/*.xml')[:cut_number]
    img_list, ano_list = [], []
    for img in img_files:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)
    for ano in ano_files:
        ano = ET.parse(ano).getroot()
        obj = ano.findall('object')
        bbox_list = []
        for i in range(len(obj)):
            bbox_list.append([
                int(obj[i].find('bndbox').find(tag).text) for tag in ('xmin', 'ymin', 'xmax', 'ymax')
            ])
        ano_list.append(bbox_list)
    return img_list, ano_list

# 원본 이미지와 ground truth box 보여주기 
def show_sample(ncols, images, bboxes):
    for i in range(ncols):
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        ax[0].imshow(images[i])
#         ax[0].axis('off')
        ax[0].set_title('Original Image')
        img_copy = images[i].copy()
        for j in range(len(bboxes[i])):
            x1, y1, x2, y2 = bboxes[i][j]
            cv2.rectangle(img_copy, (x1, y1), (x2, y2),
                          (0,255,0), int(img_copy.shape[0]*0.008), cv2.LINE_AA)
        ax[1].imshow(img_copy)
#         ax[1].axis('off')
        ax[1].set_title('Annotation Image')
        fig.show()