## Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression ([paper](https://arxiv.org/abs/1911.08287))  

### Distance IoU Loss function:
- [[code implementation](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/diou.py)]
- form:  
$$\mathcal{L_{DIoU}}=1-IoU+\frac{\rho^2\left(\textbf{b}, \textbf{b}^{gt}\right)}{c^2}$$  

### Complete IoU Loss function:
- [[code implementation](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/ciou.py)]  
- form:
$$\mathcal{L_{CIoU}}=1-IoU+\frac{\rho^2\left(\textbf{b}, \textbf{b}^{gt}\right)}{c^2}+\alpha v$$  
    - where $v=\frac{4}{\pi^2}\left(arctan \frac{w^{gt}}{h^{gt}}-arctan \frac{w}{h}\right)^2$, $\alpha=\frac{v}{\left(1-IoU\right)+v}$