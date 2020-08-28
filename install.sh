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
curl -o srmd-ncnn.zip https://github.com/nihui/srmd-ncnn-vulkan/releases/download/20200818/srmd-ncnn-vulkan-20200818-windows.zip
Expand-Archive -LiteralPath  srmd-ncnn.zip -DestinationPath ./
mv srmd-ncnn-vulkan-20200818-windows/* .



