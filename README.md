# MADF-Net: Multi-phase Attentional Deep Fusion Network for Liver Tumor Segmentation

## Overview
MADF-Net is a novel deep learning framework designed for liver tumor segmentation in multi-phase contrast-enhanced CT images. It addresses the limitations of traditional single-phase and multi-phase fusion methods by integrating **arterial (ART), portal venous (PV), and delayed (DL) phases** through full-stage fusion at the **input, feature, and decision levels**. The model also introduces a **Boundary-enhanced Dynamic Loss (BED-Loss)** to improve segmentation accuracy for small lesions and ambiguous boundaries.

## Core Methods
### 1. MADF-Net Architecture
- **Input-Level Fusion**: Concatenates multi-phase CT images along the channel dimension:  
  ```math
  I_{\text{FUSION}} = \text{Concate}(I_{\text{ART}}, I_{\text{DL}}, I_{\text{PV}})
- **Feature-Level Fusion**: Employs self-attention mechanisms to dynamically weight features from different phases:

F_{\text{FUSION}} = W_{\text{ART}} \ast F_{\text{ART}} + W_{\text{PV}} \ast F_{\text{PV}} + W_{\text{DL}} \ast F_{\text{DL}}
  
- **Decision-Level Fusion**: Fuses predictions from individual phases and the fusion branch to generate the final segmentation mask:
O^{\text{FINAL}} = F_d(O^{\text{ART}}, O^{\text{DL}}, O^{\text{PV}}, O^{\text{FUSION}})

### 2. BED-Loss Function
A dynamically weighted loss combining Cross-Entropy Loss, Dice Loss, and Boundary Loss to address class imbalance and boundary ambiguity:

\text{Loss} = \alpha \cdot L_{\text{Dice}} + \beta \cdot L_{\text{CE}} + \gamma \cdot L_{\text{BL}}

- **Boundary Loss (BL)** measures the spatial distance between predicted and ground truth boundaries using a distance map:

\mathcal{L}_{\text{boundary}} = \int_{\Omega} y_i \cdot |\phi_G(x)| dx

























