import os
import cv2
import numpy as np
from tqdm.notebook import tqdm

# selective search 수행하는 함수 정의
def selective_search(image):
    # 카피본 생성
    img_copy = image.copy()
    # selective search 생성
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_copy)
    ss.switchToSelectiveSearchFast()
    ss_result = ss.process()
    return ss_result

# IoU 계산하는 함수 정의
def get_iou(bb1, bb2):
    if bb1[0]<bb1[2] and bb1[1]<bb1[3] and bb2[0]<bb2[2] and bb2[1]<bb2[3]:
        
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # intersection 넓이 구하기
        intersection_area = (x_right-x_left)*(y_bottom-y_top)
        bb1_area = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
        bb2_area = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
        # iou 구하기
        iou = intersection_area/float(bb1_area+bb2_area-intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou
    else:
        return 0.0

# ground truth box인 경우에만 positive sample이자 라벨 1로, 
# iou가 0.3보다 작은 경우에는 negative sample이자 라벨 0으로 설정
def get_data_with_nms(train_image, train_anno):
    images_list, labels_list = [], []
    for i, image in tqdm(enumerate(train_image)):
        bboxes = train_anno[i]
        for box in bboxes:
            # ground truth box에만 라벨 1로 설정
            x1, y1, x2, y2 = box
            img_crop = image[y1:y1+y2, x1:x1+x2]
            resized = cv2.resize(img_crop, (224,224), cv2.INTER_AREA)
            images_list.append(resized)
            labels_list.append(1)
        # iou가 0.3 보다 작은 경우에는 negative sample로 간주
        negative_sample_count = 0
        s = 0
        # selective search 수행
        img_copy = image.copy()
        ss_results = selective_search(img_copy)
        for ss in ss_results:
            if s < 2000:
                if negative_sample_count < 5:
                    x1, y1, x2, y2 = ss
                    iou = get_iou(box, ss)
                    if iou < 0.3:
                        img_crop = img_copy[y1:y1+y2, x1:x1+x2]
                        resized = cv2.resize(img_crop, (224,224), cv2.INTER_AREA)
                        images_list.append(resized)
                        labels_list.append(0)
                        negative_sample_count += 1

    return images_list, labels_list
