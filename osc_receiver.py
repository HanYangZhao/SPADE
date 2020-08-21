import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from collections import OrderedDict
import importlib
import data
from data.base_dataset import BaseDataset
from util import util
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from data.base_dataset import BaseDataset, get_params, get_transform
import argparse
from pythonosc import dispatcher as d
from pythonosc import osc_server
from pythonosc import udp_client
import time

model = None
opt = None
client = None
model = None
first_run = True

def osc_image_handler(unused_addr, args, img_path):

    
    generated_image_path = process(img_path)
    client.send_message(opt.osc_channel, generated_image_path)

def process(img_path):
    
    global opt
    global model
    global first_run
    img = Image.open(img_path)
    width, height = img.size
    opt.load_size = width
    opt.crop_size = width
    opt.aspect_ratio = width / height

    if(first_run):
        model = Pix2PixModel(opt)
        model.eval()
        first_run = False
    #Loading Semantic Label
    start_time = time.time()
    label = ImageOps.grayscale(img)
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
        generated_image_path_ = opt.output_dir + '/' + os.path.splitext(os.path.basename(img_path))[0] +"_generated"+".png"
        print('---- generated image ', generated_image_path_, np.shape(generated_image))
        im_rgb = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(generated_image_path_, im_rgb)
    print("Time to generate image: " + str(time.time() - start_time))
    return generated_image_path_


def main():
    global opt
    global client
    #Loading the trained model
    opt = TestOptions().parse()
    opt.no_instance = True
    if opt.dataset_mode == "coco":
        opt.label_nc = 183

 

    dispatcher = d.Dispatcher()
    dispatcher.map(opt.osc_channel, osc_image_handler, "path")
    client = udp_client.SimpleUDPClient(opt.osc_send_ip, opt.osc_send_port)
    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", opt.osc_receive_port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()


if __name__ == '__main__':

    main()