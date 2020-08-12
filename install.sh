#!/bin/bash

conda env create -f conda_env.yml

mkdir -p checkpoints
gdown 'https://drive.google.com/u/0/uc?export=download&confirm=TM7u&id=12gvlTbMvUcJewQlSEaZdeb2CdOB-b8kQ'
tar xvf checkpoints.tar.gz -C checkpoints

cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .