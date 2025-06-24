### LiTS2017 datasets

The Liver Tumor Segmentation Benchmark (LiTS2017) is an open - access dataset for medical image segmentation. It was provided by several clinical sites worldwide and aims to promote the development and evaluation of liver and liver tumor segmentation algorithms.

- The LiTS2017 datasets can be downloaded here: {[LiTS2017](https://competitions.codalab.org/competitions/17094)}.
- After downloading the datasets, you should run ./data_prepare/preprocess_lits2017_png.py to convert .nii files into .png files for training. (Save the downloaded LiTS2017 datasets in the data folder in the following format.)


- './data_prepare/'
  - preprocess_lits2017_png.py
- './data/'
  - LITS2017
    - ct
      - .nii
    - label
      - .nii
  - trainImage_lits2017_png
      - .png
  - trainImage_lits2017_png
      - .png
### Other datasets
- Other datasets just similar to LiTS2017
