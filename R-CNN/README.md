# R-CNN (Rich feature hierarchies for accurate object detection and semantic segmentation)  
## data: kaggle  
### 논문 링크 : https://arxiv.org/abs/1311.2524  
### R-CNN Architecture  
<img src = "https://production-media.paperswithcode.com/methods/new_splash-method_NaA95zW.jpg" width=1000>
<img sec = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FbdmFi2%2FbtqAQ38E2v3%2FJMXznsWZsX3YQAuTkKtpWK%2Fimg.png" width=1000>


- step1. 약 2000개의 region proposal을 생성  
- step2. 이미지 crop 및 resize 진행
- step3. CNN에 입력해 feature vector 추출
- step4. Linear SVM에 입력해 Classification 
