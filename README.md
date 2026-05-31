<div align="center">

<img src="figures/logo_full.png" width="260">

## TimeRadar: A Domain-Rotatable Foundation Model for Time Series Anomaly Detection

[![arXiv](https://img.shields.io/badge/arXiv-2602.19068-b31b1b)](https://arxiv.org/abs/2602.19068)
[![Project Page](https://img.shields.io/badge/Project_Page-website-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---


## Overview

We introduce **TimeRadar**, an innovative time series foundation model (TSFM) built in a **fractional time–frequency domain** to support generalist time series anomaly detection (TSAD) across diverse unseen datasets.  Our key insight is that rotating a time series into a **data-dependent fractional time–frequency representation** can adaptively differentiate normal and abnormal signals across different datasets. To this end, we propose a novel component, **Fractionally modulated Time-Frequency Reconstruction (FTFRecon)**, which leverages a **learnable fractional order** to rotate the time series to the most pronounced angle between the continuous time and frequency domains for accurate data reconstruction.   This design enables **adaptive reconstruction in an optimal time–frequency domain** for each input, effectively distinguishing **unbounded abnormal patterns** from regular ones across datasets, including previously unseen datasets.  To further capture **local abnormalities** that may not be reflected by global reconstruction, we introduce a **Contextual Deviation Learning (CDL)** component, which models the local deviation of the input relative to its contextual time series data in the rotatable domain.




<div align="center"><img src="figures/framework.png" width="92%"></div>


## Main Results


## Repository Structure


## 📖 Citation
    
If you find this work useful, please cite our paper:

```bibtex
@article{he2026timeradar,
  title={TimeRadar: A Domain-Rotatable Foundation Model for Time Series Anomaly Detection},
  author={He, Hui and Qiao, Hezhe and Chen, Yutong and Yi, Kun and Pang, Guansong},
  journal={arXiv preprint arXiv:2602.19068},
  year={2026}
}
