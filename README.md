# Hybrid Ordinary-Welsch Function based Robust Matrix Completion for MIMO Radar
code for "Hybrid Ordinary-Welsch Function based Robust Matrix Completion for MIMO Radar" in IEEE Transactions on Aerospace and Electronic Systems (TAES) 2024

<img src="https://github.com/ShuDun23/robust-MIMO-MC/blob/main/Drawing1.png" width="500px">

In this paper, we consider a sub-Nyquist sampled multiple-input multiple-output (MIMO) radar scenario where the observations are contaminated by impulsive non-Gaussian clutter, which introduces outliers. To recover the missing data, we propose a robust matrix completion (MC) method with a regularizer that acts on outliers. This regularizer whose solution is unbiased, sparse and continuous, is generated by the hybrid ordinary-Welsch (HOW) function, aiming to classify each measurement as normal, semi-contaminated or contaminated, and then handle it appropriately. Then proximal block coordinate descent (BCD) is leveraged to tackle the HOW-based MC problem and the convergence property and computational cost of the developed algorithm are analyzed. Experimental results validate the superior performance of our method compared to existing approaches in terms of MC and direction-of-arrival estimation accuracies as well as runtime in the presence of Gaussian mixture noise and K-distributed clutter.

## Citation
H. N. Sheng, Z.-Y. Wang, Z. Liu, and H. C. So, “Hybrid ordinary-Welsch function based robust matrix completion for MIMO radar,” IEEE Transactions on Aerospace and Electronic Systems (TAES), 2024.

Z.-Y. Wang, H. C. So and A. M. Zoubir, “Robust low-rank matrix recovery via hybrid ordinary-Welsch function,” IEEE Transaction on Signal Processing, vol. 71, pp. 2548-2563, Jul. 2023.

