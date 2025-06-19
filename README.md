# MADF-Net: Multi-phase Attentional Deep Fusion Network for Liver Tumor Segmentation  

## Paper Information  
- **Title**: MADF-Net: Multi-phase Attentional Deep Fusion Network for Liver Tumor Segmentation  
- **Journal**: IEEE Transactions on Medical Imaging  
- **Year**: 2020  
- **File**: TMI_coperwith_JJN.pdf  

## Pre-trained Weights  
[Google Drive](https://drive.google.com/drive/folders/1FSgOOqEkdjfBTvYudSf9NAxIwG3CxWxWxW?usp=drive_link)  


## Core Innovations  
### 1. MADF-Net Architecture  
- **Input-Level Fusion**: Concatenates arterial (ART), portal venous (PV), and delayed (DL) phase CT images:  
  \[ 
  I_{\text{FUSION}} = \text{Concatenate}(I_{\text{ART}}, I_{\text{DL}}, I_{\text{PV}}) 
  \]  
- **Feature-Level Fusion**: Employs self-attention to dynamically weight phase-specific features:  
  \[ 
  F_{\text{FUSION}} = W_{\text{ART}} \cdot F_{\text{ART}} + W_{\text{PV}} \cdot F_{\text{PV}} + W_{\text{DL}} \cdot F_{\text{DL}} 
  \]  
  Weights are learned via a shared key-query mechanism:  
  \[
  \begin{aligned} 
  W_{\text{ART}} &= \text{softmax}\left(\frac{Q_{\text{ART}}^{P} \cdot (K_{\text{ART}}^{P})^{\top}}{\sqrt{d_{k}}}\right), \\
  W_{\text{PV}} &= \text{softmax}\left(\frac{Q_{\text{PV}}^{P} \cdot (K_{\text{PV}}^{P})^{\top}}{\sqrt{d_{k}}}\right), \\
  W_{\text{DL}} &= \text{softmax}\left(\frac{Q_{\text{DL}}^{P} \cdot (K_{\text{DL}}^{P})^{\top}}{\sqrt{d_{k}}}\right)
  \end{aligned}
  \]  
- **Decision-Level Fusion**: Fuses predictions from individual phases and the fusion branch:  
  \[ 
  O^{\text{FINAL}} = F_{d}(O^{\text{ART}}, O^{\text{DL}}, O^{\text{PV}}, O^{\text{FUSION}}) 
  \]  

### 2. Boundary-Enhanced Dynamic Loss (BED-Loss)  
Combines Cross-Entropy, Dice, and Boundary Loss with adaptive weighting:  
\[ 
\text{Loss} = \alpha \cdot L_{\text{Dice}} + \beta \cdot L_{\text{CE}} + \gamma \cdot L_{\text{BL}} 
\]  
- **Boundary Loss** measures pixel-wise distance to ground truth boundaries:  
  \[ 
  \mathcal{L}_{\text{boundary}} = \int_{\Omega} y_{i} \cdot \left| \phi_{G}(x) \right| \, dx 
  \]  


## Experimental Results  
### Datasets  
| Dataset | Phases       | Samples | Annotation Tool | Registration Method |  
|---------|--------------|---------|-----------------|---------------------|  
| LiTS2017| Single (PV)  | 131     | Publicly available | -                   |  
| MPLL    | Multi (ART/PV/DL) | 141   | ITK-SNAP        | B-spline            |  

### Key Metrics Comparison  
| Method       | Phases       | DSC (%) | Jaccard | HD₉₅   | ASSD    |  
|--------------|--------------|---------|---------|--------|---------|  
| Single-Phase (PV)| PV          | 78.28   | 0.6431  | 3.4972 | 4.37    |  
| MADF-Net     | All Phases   | **80.99**| **0.6805**|**2.5948**|**4.26**|  

### Ablation Studies  
| Components       | DSC Improvement (%) | HD₉₅ Reduction |  
|------------------|---------------------|----------------|  
| Self-Attention   | +2.00               | -2.2693        |  
| BED-Loss         | +1.63               | -1.2700        |  
| Combined         | +2.47               | -3.2693        |  




## Citation  
```bibtex
@article{madfnet_2020,
  title={MADF-Net: Multi-phase Attentional Deep Fusion Network for Liver Tumor Segmentation},
  author={First A. Author, Second B. Author, Third C. Author},
  journal={IEEE Transactions on Medical Imaging},
  year={2020},
  file={TMI_coperwith_JJN.pdf}
}
