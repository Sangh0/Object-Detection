## Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection ([paper](https://arxiv.org/abs/2006.04388))  

### General Focal Loss
- [[paper](https://arxiv.org/abs/1708.02002)]
- [[code implementation](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/focal_loss.py)]
- form:  
$$FL\left(p_t\right)=-\alpha\left(1-p_t\right)^{\gamma}\log\left(p_t\right)$$  

### Code Implementation  
**Quality Focal Loss**  
- [[code implementation](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/qfocal.py)]
- form:  
$$QFL\left(\sigma\right)=-\vert y-\sigma\vert^{\beta}\left(\left(1-y\right)\log\left(1-\sigma\right)+y\log\left(\sigma\right)\right)$$  

**Distribution Focal Loss**  
- [[code implementation](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/dfocal.py)]  
- form:
$$DFL\left(S_i, S_{i+1}\right)=-\left(\left(y_{i+1}-y\right)\log\left(S_i\right)+\left(y-y_i\right)\log\left(S_{i+1}\right)\right)$$  

**Generalized Focal Loss**  
- general form:  
$$\mathcal{L} = \frac{1}{N_{pos}}\sum_z \mathcal{L_Q} + \frac{1}{N_{pos}} \sum_z \textbf{1}\left(\lambda_0 \mathcal{L_B} + \lambda_1 \mathcal{L_D}\right)}$$  

    - where $\mathcal{L}_Q$ is Quality focal loss and $\mathcal{L}_D$ is Distribution Focal Loss  
    - Typically, $\mathcal{L}_B$ denotes the bounding box regression loss like GIoU and CIoU Loss

    _{\left\{c_z^*>0\right\}