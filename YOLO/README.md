## YOLOv3 implementation

**model**
- yolov3   
- references: [[yolo paper](https://arxiv.org/abs/1506.02640)], [[yolo9000 paper](https://arxiv.org/abs/1612.08242)], [[yolov3 paper](https://arxiv.org/abs/1804.02767)], [[darknet](https://pjreddie.com/darknet/imagenet/)]  
- architecture  
<img src = "https://github.com/Sangh0/Object-Detection/blob/main/YOLO/figure/yolov3_architecture1.png" width=500>

**Training**
```
$ python3 ./train.py --data_yaml_path {yaml file directory} --epochs 300 --batch_size 16 --iou_threshold 0.2
```