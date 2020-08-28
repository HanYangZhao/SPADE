#!/bin/bash

conda env create -f conda_env.yml
conda activate nvidia-spade
mkdir -p checkpoints
gdown 'https://drive.google.com/u/0/uc?export=download&confirm=TM7u&id=12gvlTbMvUcJewQlSEaZdeb2CdOB-b8kQ'
tar xvf checkpoints.tar.gz -C checkpoints

cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .

#image scaler 
# Downlaod the zip
# unzip
# Copy all the files under the srmd-ncnn-vulkan folder to the SPADE root folder
curl -o srmd-ncnn.zip https://github.com/nihui/srmd-ncnn-vulkan/releases/download/20200818/srmd-ncnn-vulkan-20200818-windows.zip
Expand-Archive -LiteralPath  srmd-ncnn.zip -DestinationPath ./
mv srmd-ncnn-vulkan-20200818-windows/* .


#u-net image segmentation
# git clone the repo
# rename the repo to u2net
# download the two model files from google drive
# Create directory U-2-Net/saved_models/u2net/ 
# Create directory U-2-Net/saved_models/u2netp/
# move u2net.pth to U-2-Net/saved_models/u2net/
# move u2netp.pth to U-2-Net/saved_models/u2net/
git clone https://github.com/NathanUA/U-2-Net
mv U-2-Net u2net
gdown 'https://drive.google.com/u/1/uc?export=download&confirm=Mvjq&id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ'
gdown 'https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy'
mkdir -p  U-2-Net/saved_models/u2net/
mkdir -p  U-2-Net/saved_models/u2netp/
mv u2net.pth U-2-Net/saved_models/u2net/
mv u2netp.pth U-2-Net/saved_models/u2netp/