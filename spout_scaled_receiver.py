import os
import cv2
import numpy as np
from PIL import Image,ImageOps
from collections import OrderedDict
import importlib
import data
from data.base_dataset import BaseDataset
from util import util
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from data.base_dataset import BaseDataset, get_params, get_transform
import argparse
import time
import spout as sro
from options.test_options import BaseOptions
import queue
import threading
import subprocess
import torch
from torchvision import transforms
from torchvision import utils
import copy

# from Waifu2x.utils.prepare_images import *
# from Waifu2x.Models import *

process_queue = queue.Queue()
scaled_image_queue = queue.Queue()
isExit = False

class SpoutOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.add_argument('--spout_size', nargs = 2, type=int, choices=[256,512,1024,2048],
                                    default=[256, 256], help='Width and height of the spout receiver')
        parser.add_argument('--spout_in', type=str, default='spout_receiver_in',
                                    help='Spout receiving name - the name of the sender you want to receive')
        parser.add_argument('--spout_out', type=str, default='spout_receiver_out',
                                    help='Spout receiving name - the name of the sender you want to send')
        parser.add_argument('--window_size', nargs = 2, type=int, default=[10, 10],
                                    help='Width and height of the window')
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=1024)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.add_argument('--output_dir', type=str, default='results',
                            help='Directory name to save the generated images')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        # parser.add_argument('--scale', type=int, default='1',help='scale the output image by n factor. The final resolution should match the spout_size')
        parser.add_argument('--denoise', type=int, default='1',help='denoise image during scaling')
        self.isTrain = False
        return parser

def tensor_to_image(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    from PIL import Image
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im

def image_scaler(denoise,scale):
    global process_queue
    global scaled_image_queue
    denoise_flag =  str(denoise)
    input_file = "generated.png"
    output_file = "scaled.png"
    command = "srmd-ncnn-vulkan.exe"
    print(command)
    # use_cuda = torch.cuda.is_available()
    # device = torch.device('cuda' if use_cuda else 'cpu')
    # print("scale : " + str(scale))
    # model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
    #                         single_conv_size=3, single_conv_group=1,
    #                         scale=4, activation=nn.LeakyReLU(0.1),
    #                         SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
                            
    # model_cran_v2 = network_to_half(model_cran_v2)
    # checkpoint = "Waifu2x/model_check_points/CRAN_V2/CARN_model_checkpoint.pt"
    # model_cran_v2.load_state_dict(torch.load(checkpoint, 'cpu'))
    # model_cran_v2 = model_cran_v2.float() # if use cpu 


    while not isExit:
        image = process_queue.get(block=True)
        cv2.imwrite(output_file,image)
        # im_pil = Image.fromarray(image)
        # img_t = to_tensor(im_pil).unsqueeze(0) 
        start_time = time.time()
        # with torch.no_grad():
        #     out = model_cran_v2(img_t)
        #     print("here")
        # out = tensor_to_image(out,nrow=2)
        # open_cv_image = np.array(out) 
        # save_image(out, 'out.png', nrow=2)
        # print("Image scaling in %.3f seconds!" % (time.time() - start_time))
        # scaled_image_queue.put(open_cv_image)
        #waifu2x-ncnn-vulkan.exe -i input.jpg -o output.png -n 2 -s 2
        # subprocess.Popen(["./waifu2x/waifu2x-ncnn-vulkan.exe", input_flag, output_flag, scale_flag, denoise_flag])
        if(scale == 8):
            proc = subprocess.run([command,"-i",output_file,"-o",output_file,"-n",denoise_flag,"-s","2"],stdout=subprocess.DEVNULL,shell=True)
            scale = 4
            # print("the commandline is {}".format(proc.args))
        proc = subprocess.run([command,"-i",output_file,"-o",output_file,"-n",denoise_flag,"-s",str(scale)],stdout=subprocess.DEVNULL,shell=True)
        print("the commandline is {}".format(proc.args))
        print("Image scaling in %.3f seconds!" % (time.time() - start_time))
        scaled_image = cv2.imread(output_file)
        print(scaled_image.shape[:2])
        if scaled_image.size == 0:
            print("Unable to read scaled image")
            exit()
        else:
            scaled_image_queue.put(scaled_image)
            # os.remove(output_file)

def is_same_image(prev_frame,current_frame):
    if prev_frame is None:
        return False
    difference = cv2.subtract(prev_frame,current_frame)
    if cv2.countNonZero(difference) == 0:
        return True
    return False

def main():
    global process_queue
    global scaled_image_queue
    global isExit
    t = None
    opt = SpoutOptions().parse()
    opt.no_instance = True
    if opt.dataset_mode == "coco":
        opt.label_nc = 183
        opt.load_size = 256
        opt.crop_size = 256
    if opt.name == "landscape":
        opt.load_size = 512
        opt.crop_size = 512
    sro.InitSpout(opt)

    model = Pix2PixModel(opt)
    model.eval()

    scaling_ratio = int(opt.spout_size[0] / opt.load_size)
    print("scaling_ratio : " + str(scaling_ratio))
    if(scaling_ratio > 1):
        t = threading.Thread(target=image_scaler,args=(opt.denoise,scaling_ratio))
        t.start()
    prev_frame = None
    while cv2.waitKey(1) != 27:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            os._exit(1)
        frame = sro.GetSpoutFrame(opt.spout_in)
        if frame is None:
            print("Error : no image received")
            os._exit(1)
        # if is_same_image(frame,prev_frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_same_image(prev_frame,frame):
            continue
        prev_frame = frame
        if(opt.spout_size[0] != opt.crop_size):
            frame = cv2.resize(frame, (opt.crop_size,opt.crop_size), interpolation = cv2.INTER_LANCZOS4)
        label = Image.fromarray(frame)
        params = get_params(opt, label.size)
        transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = opt.label_nc

        #Creating data_dictionay
        data_i = {}
        data_i['label'] = label_tensor.unsqueeze(0)
        data_i['path'] = None
        data_i['image'] = None
        data_i['instance'] = None

        #Inference code
        start_time = time.time()
        generated = model(data_i, mode='inference')
        for b in range(generated.shape[0]):
            generated_image = generated[b]
            generated_image = util.tensor2im(generated_image)
            im_rgb = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
            print("Image generated in %.3f seconds!" % (time.time() - start_time))
            if(scaling_ratio > 1):
                process_queue.put(im_rgb)
                im_rgb = scaled_image_queue.get()
            if opt.spout_out:
                sro.SendSpoutFrame(im_rgb, opt)
            cv2.imshow("Generated", im_rgb)
if __name__ == '__main__':
    main()

