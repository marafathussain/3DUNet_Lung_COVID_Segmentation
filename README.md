# 3D-UNet for Lung and COVID infection Segmentation
Running this code needs [MONAI](https://github.com/Project-MONAI/MONAI) framework installed. I used 20 CT volumes from [this link](https://academictorrents.com/details/136ffddd0959108becb2b3a86630bec049fcb0ff). I used left-right and up-down flipping to multiply 20 CT volumes into 80. Then used 5-fold cross-validation for segmentation evaluation. 

## Data preprocessing
All the CT volumes were resampled to a common voxel dimension of 1.6mm X 1.6mm X 3.2mm. 

## Accuracy
Each fold was run for 100 epochs and the DICE metrics for five-fold cross-validations are 0.83, 0.88, 0.71, 0.68, and 0.74 respectively.

## Pre-trained model
The pretrained model can be downloaded from this [link](https://drive.google.com/file/d/11HzTu79q05Pr9IJZQN9fIcZk4ErapweA/view?usp=sharing).