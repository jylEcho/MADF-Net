# MADF-Net: Multi-phase Attentional Deep Fusion Network for Liver Tumor Segmentation  

## Pre-trained Weights  
The weights of the pre-trained MADF-Net in 1P、2P、3P comparative analysis could be downloaded [Here](https://drive.google.com/drive/folders/1FSgOOqEkdjfBTvYudSf9NAxIwG3CxWxW?usp=drive_link)  

## Installation (创建环境相关)
1、environments:Linux 5.4.0

2、Create a virtual environment: conda create -n environment_name python=3.8 -y and conda activate environment_name.

3、Install Pytorch : pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117

## Pre-process （先运行data_prepare/split.py进行划分、再运行data_prepare/generate_2D_train.py和data_prepare/generate_2D_test.py）
First run data_prepare/split.py for period division, then run data_prepare/generate_2D_train.py and data_prepare/generate_2D_test.py for period data processing.

## Generate distance map (分别运行boundary_map/liver_distance_map.py和boundary_map/tumor_distance_map.py分别为liver 和 tumor 生成boundary map，dataset/dataset_multiphase.py和dataset/dataset_multiphase_boundarymap.py分别对应训练时不加载距离图和加载距离图的数据加载器)
- 1.Then run boundary_map/liver_distance_map.py and boundary_map/tumor_distance_map.py to generate the boundary maps for liver and tumor, respectively.
- 2.Using the dataset/dataset_multiphase.py loader if you want to train without loading the distance map, or the dataset/dataset_multiphase_boundarymap.py loader if you want to load the distance map during training.
## Training Process（运行bash/train_multiphase.sh，运行中的权重保存在在model_out文件夹中）
The model is trained by running bash/evaluate.sh, and the weights of its runs are stored in the model_out folder.
## Evalution（运行bash/evaluate.sh，把训练权重和测试数据地址在evaluate.sh中进行替换，得到测试结果，结果保存在model_out中）
Next, we need to test the model, first run bash/evaluate.sh, replace the training weights and test data addresses in evaluate.sh. The test results will be saved in the model_out folder for viewing.
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






