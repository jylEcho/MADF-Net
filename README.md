# MADF-Net: Multi-phase Attentional Deep Fusion Network for Liver Tumor Segmentation  

## Paper Information  
- **Title**: MADF-Net: Multi-phase Attentional Deep Fusion Network for Liver Tumor Segmentation  
- **Journal**: IEEE Transactions on Medical Imaging  
- **Year**: 2025 

## Pre-trained Weights  
The weights of the pre-trained MADF-Net could be downloaded [Here](https://drive.google.com/drive/folders/1FSgOOqEkdjfBTvYudSf9NAxIwG3CxWxW?usp=drive_link)  

## Training Process

All models were trained for 450 epochs with a batch size of 8. The Stochastic Gradient Descent (SGD) optimizer was adopted with a learning rate of 0.00011 and 4 parallel data loading workers.
(The proposed method was implemented on a Linux 5.4.0 system using PyTorch 1.13.1. All experiments were conducted on two NVIDIA GeForce RTX 3090 GPUs (24 GB × 2), providing sufficient computational resources for efficient model training and evaluation.)

## Core Innovations  
### 1. MADF-Net Architecture  
- **Input-Level Fusion**: Concatenates arterial (ART), portal venous (PV), and delayed (DL) phase CT images. 
- **Feature-Level Fusion**: Employs self-attention to dynamically weight phase-specific features.
- **Decision-Level Fusion**: Fuses predictions from individual phases and the fusion branch.  

### 2. Boundary-Enhanced Dynamic Loss (BED-Loss)  
Combines Cross-Entropy, Dice, and Boundary Loss with adaptive weighting. 

The initial weights of BED-Loss in Eq. (8) are set to \(\alpha\)=0.49, \(\beta \)=0.49 and \(\gamma\)=0.02. If the training loss plateaus for 10 epochs, the weights are dynamically adjusted to a 4:4:2 ratio, and you can download the model weights from the Google Drive link above, and if the link is broken, you can contact the corresponding author to obtain and update the URL.

## Experimental Results  
### Datasets  
| Dataset | Phases       | Samples | Annotation Tool | Registration Method |  
|---------|--------------|---------|-----------------|---------------------|  
| LiTS2017| Single (PV)  | 131     | Publicly available | -                   |  
| MPLL    | Multi (ART/PV/DL) | 141   | ITK-SNAP        | B-spline            |  

### Ablation Studies  
| Components       | DSC Improvement (%) | HD₉₅ Reduction |  
|------------------|---------------------|----------------|  
| Self-Attention   | +2.00               | -2.2693        |  
| BED-Loss         | +1.63               | -1.2700        |  
| Combined         | +2.47               | -3.2693        |  





