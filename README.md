# MADF-Net: Multi-phase Attentional Deep Fusion Network for Liver Tumor Segmentation

## Overview
MADF-Net is proposed to address the limitations of existing multiphase feature fusion in contrast-enhanced CT. It integrates arterial, portal venous, and delayed phase CT images through full-stage fusion at the input, feature, and decision levels, and introduces a boundary-aware dynamically weighted loss function (BED-Loss) to enhance segmentation of small lesions and indistinct boundaries.

## Core Meth{insert\_element\_0\_}ods
### 1. MADF-Net Architecture
- **Input-Level Fusion**: Concatenates multi-phase CT images along the channel dimension:  
  \[ I_{\text{FUSION}} = \text{Concate}(I_{\text{ART}}, I_{\text{DL}}, I_{\text{PV}}) \]  
- **Feature-{insert\_element\_1\_}Level Fusion**: Employs self-attention to dynamically weight phase features:  
  \[ F_{\text{FUSION}} = W_{\text{ART}} \ast F_{\text{ART}} + W_{\text{PV}} \ast F_{\text{PV}} + W_{\text{DL}} \ast F_{\text{DL}} \]  
  Weights are learned via shared key-query scheme:  
  \[
  \begin{aligned} 
  W_{\text{ART}} &= \text{softmax}\left(\frac{Q_{\text{ART}}^{P}(K_{\text{ART}}^{P})^{T}}{\sqrt{d_{k}}}\right), \\
  W_{\text{PV}} &= \text{softmax}\left(\frac{Q_{\text{PV}}^{P}(K_{\text{PV}}^{P})^{T}}{\sqrt{d_{k}}}\right), \\
  W_{\text{DL}} &= \text{softmax}\left(\frac{Q_{\text{DL}}^{P}(K_{\text{DL}}^{P})^{T}}{\sqrt{d_{k}}}\right)
  \end{aligned}
  \]  
- **Decision{insert\_element\_2\_}-Level Fusion**: Fuses phase predictions and fusion branch output:  
  \[ O^{\text{FINAL}} = F_d(O^{\text{ART}}, O^{\text{DL}}, O^{\text{PV}}, O^{\text{FUSION}}) \]  

### 2. BED-{insert\_element\_3\_}Loss Function
Combines Cross-Entropy, Dice, and Boundary Loss with dynamic weighting:  
\[ \text{Loss} = \alpha \cdot L_{\text{Dice}} + \beta \cdot L_{\text{CE}} + \gamma \cdot L_{\text{BL}} \]  
- **Boundary Loss** measures distance from predicted pixels to ground truth boundaries:  
  \[ \mathcal{L}_{\text{boundary}} = \int_{\Omega} y_i \cdot |\phi_G(x)| dx \]  

## Experime{insert\_element\_4\_}nts
### Datasets
- **LiTS2017**: Single-phase PV CT (131 training cases).  
- **MPLL**:{insert\_element\_5\_} Multi-phase CT (ART/PV/DL, 141 cases), registered via B-spline.  

### Evalua{insert\_element\_6\_}tion Metrics
DSC, Jaccard, HD<sub>95</sub>, ASSD.  

### Key Re{insert\_element\_7\_}sults
#### Multi-phase Fusion Performance
| Phase Combination | DSC (%) | Jaccard | HD<sub>95</sub> | ASSD  |
|---------------------|---------|---------|-----------------|-------|
| Single-phase (PV)   | 78.28   | 0.6431  | 3.4972          | 4.37  |
| Three-phase (All)   | **80.99**| **0.6805**|**2.5948**       |**4.26**|  

#### Ablati{insert\_element\_8\_}on Studies
| Components       | DSC Improvement (%) | HD<sub>95</sub> Reduction |
|------------------|---------------------|---------------------------|
| Self-Attention   | +2.00              | -2.2693                   |
| BED-Loss         | +1.63              | -1.2700                   |
| Combined         | +2.47              | -3.2693                   |  

## Pre-trai{insert\_element\_9\_}ned Weights
[Google Drive](https://drive.google.com/drive/folders/1FSgOOqEkdjfBTvYudSf9NAxIwG3CxWxWxW?usp=drive_link)  

## Citation
```bibtex
@article{madfnet_2020,
  title={MADF-Net: Multi-phase Attentional Deep Fusion Network for Liver Tumor Segmentation},
  author={First A. Author, Second B. Author, Third C. Author},
  journal={IEEE Transactions on Medical Imaging},
  year={2020}
}
