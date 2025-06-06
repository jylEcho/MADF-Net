# MADF-Net

# MADF-Net: Multi-phase Attentional Deep Fusion Network for Liver Tumor Segmentation

## Overview
MADF-Net is a novel deep learning framework designed for liver tumor segmentation in multi-phase contrast-enhanced CT images. It addresses the limitations of traditional single-phase and multi-phase fusion methods by integrating **arterial (ART), portal venous (PV), and delayed (DL) phases** through full-stage fusion at the **input, feature, and decision levels**. The model also introduces a **Boundary-enhanced Dynamic Loss (BED-Loss)** to improve segmentation accuracy for small lesions and ambiguous boundaries.

## Core Methods
### 1. MADF-Net Architecture
- **Input-Level Fusion**: Concatenates multi-phase CT images along the channel dimension:  
  ```math
  I_{\text{FUSION}} = \text{Concate}(I_{\text{ART}}, I_{\text{DL}}, I_{\text{PV}})










