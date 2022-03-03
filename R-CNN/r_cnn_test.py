import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
from r_cnn_utils import selective_search

def r_cnn_test(test_images, model):
    test_result = []
    for i in tqdm(range(len(test_images))):
        image = test_images[i].copy()
        # selective search 수행
        ss_results = selective_search(image)
        
        boxes = []
        for s, result in enumerate(ss_results):
            if s < 2000:
                x1, y1, x2, y2 = result
                crop_image = image[y1:y1+y2, x1:x1+x2]
                resized = cv2.resize(crop_image, (224, 224), interpolation=cv2.INTER_AREA)
                resized = np.expand_dims(resized, axis=0)
                out = model.predict(resized)
                
                if out[0][1]>0.9:
                    boxes.append([x1,y1,x2,y2])
        
        for box in boxes:
            x1, y1, x2, y2 = box
            # plot bounding box
            cv2.rectangle(image, (x1,y1), (x1+x2,y1+y2), (0,255,0), 1, cv2.LINE_AA)
        
        test_result.append(image)
    return test_result

# 이미지 결과 살펴보는 함수 정의
def compare_image(test_true_image, test_true_anno, test_predict):
    for i in range(len(test_predict)):
        # bounding box 예측 이미지
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        ax[0].imshow(test_predict[i])
        ax[0].axis('off')
        ax[0].set_title('Predicted Box')
        
        # Ground Truth Box
        img_copy = test_true_image[i].copy()
        for j in range(len(test_true_anno[i])):
            x1, y1, x2, y2 = test_true_anno[i][j]
            cv2.rectangle(img_copy, (x1, y1), (x2, y2),
                          (0,255,0), int(img_copy.shape[0]*0.008), cv2.LINE_AA)
            ax[1].imshow(img_copy)
            ax[1].axis('off')
            ax[1].set_title('Ground Truth Box')
            fig.show()