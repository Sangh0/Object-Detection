## VarifocalNet: An IoU-aware Dense Object Detector ([paper](https://arxiv.org/abs/2008.13367))  

### Varifocal Loss
- [[code implementation](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/varifocal_loss.py)]
- form:  
$$VFL\left(p,q\right)=
\begin{cases}
-q\left(q log\left(p\right)+\left(1-q\right) log\left(1-p\right)\right), & \mbox{if }q>0\mbox{ (positive sample)}\\ 
-\alpha p^{\gamma}log\left(1-p\right), & \mbox{if }q=0\mbox{ (negative sample)} 
\end{cases}$$  
    - where $p$ is the predicted IACS and $q$ is the target score  
    - For a foreground point, $q$ for its ground truth class is set as the IoU between the predicted box and its ground truth and $0$ otherwise, whereas for a background point, the target $q$ for all classes is $0$  
    - The varifocal loss only reduces the loss contribution from negative examples $(q=0)$ by scaling their losses with a factor $p^{\gamma}$ and does not down-weight positive examples $(q>0)$

$$f(n)=
\begin{cases}
n/2, & \mbox{if }n\mbox{ is even} \\
3n+1, & \mbox{if }n\mbox{ is odd}
\end{cases}$$