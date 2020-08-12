

import os
import cv2
import numpy as np
from PIL import Image
from collections import OrderedDict
import importlib
import data
from data.base_dataset import BaseDataset
from util import util
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from data.base_dataset import BaseDataset, get_params, get_transform
import argparse

def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported. 
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls
            
    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset

def main():
    #Loading the trained model
    opt = TestOptions().parse()
    opt.no_instance = True
    if opt.dataset_mode == "coco":
        opt.label_nc = 183

    img = cv2.imread(opt.test_file_path)
    height, width = img.shape[:2]
    opt.load_size = width
    opt.crop_size = width
    opt.aspect_ratio = width / height
    # print(opt)
    model = Pix2PixModel(opt)
    model.eval()

    #Loading Semantic Label
    label = Image.open(opt.test_file_path)
    params = get_params(opt, label.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0
    label_tensor[label_tensor == 255] = opt.label_nc
    print("-- Label tensor :", np.shape(label_tensor))

    #Creating data_dictionay
    data_i = {}
    data_i['label'] = label_tensor.unsqueeze(0)
    data_i['path'] = None
    data_i['image'] = None
    data_i['instance'] = None

    #Inference code
    generated = model(data_i, mode='inference')
    for b in range(generated.shape[0]):
        generated_image = generated[b]
        generated_image = util.tensor2im(generated_image)
        generated_image_path_ = opt.output_dir + '/' + os.path.splitext(os.path.basename(opt.test_file_path))[0] +"_generated"+".png"
        print('---- generated image ', generated_image_path_, np.shape(generated_image))
        cv2.imwrite(generated_image_path_, generated_image)

if __name__ == '__main__':
    main()