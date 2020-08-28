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
from spout import Spout
from options.test_options import BaseOptions
import queue
import threading
import subprocess
import torch
from torchvision import transforms
from torchvision import utils
import copy
import socket   

from torch.autograd import Variable
from torchvision import transforms

from u2net.model import U2NET # full size version 173.6 MB
from u2net.model import U2NETP # small version u2net 4.7 MB


# from Waifu2x.utils.prepare_images import *
# from Waifu2x.Models import *

class SpoutOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.add_argument('--spout_size', nargs = 2, type=int, choices=[256,512,1024,2048],
                                    default=[256, 256], help='Width and height of the spout receiver')
        parser.add_argument('--spout_in', type=str, default='spout_receiver_in',
                                    help='Spout receiving name - the name of the sender you want to receive')
        parser.add_argument('--spout_out', type=str, nargs='+',default='spout_receiver_out', help='the names of the channel you want to send the output')
        parser.add_argument('--spout_mask_out',type=str)
        parser.add_argument('--mask_substract',action='store_true')
        parser.add_argument('--mask_trained_model', default='u2net', type=str, help='Model to load')
        parser.add_argument('--window_size', nargs = 2, type=int, default=[10, 10],
                                    help='Width and height of the window')
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=1024)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.add_argument('--output_dir', type=str, default='results',
                            help='Directory name to save the generated images')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--ml_scaling', action='store_true', help='enable image scaling with machine learning')
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

def image_scaler(denoise,scale,process_queue,scaled_image_queue,isExit):
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

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def is_same_image(prev_frame,current_frame):
    if prev_frame is None:
        return False
    difference = cv2.subtract(prev_frame,current_frame)
    if cv2.countNonZero(difference) == 0:
        return True
    return False

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def mask_generator(frame,device,net,opt):
    content_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    start_time = time.time()
    resized = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LANCZOS4)
    norm_image = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    inputs_test = content_transform(norm_image)
    inputs_test = inputs_test.unsqueeze(0).to(device)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)

    mask = pred
    mask = mask.squeeze()
    mask_np = mask.cpu().data.numpy()
    im = Image.fromarray(mask_np * 255).convert('RGB')
    imr = im.resize((frame.shape[1], frame.shape[0]), resample=Image.LANCZOS)

    open_cv_image = np.array(imr)

    print("Image masked in %.3f seconds!" % (time.time() - start_time))
    return open_cv_image

def main():
    process_queue = queue.Queue()
    scaled_image_queue = queue.Queue()
    isExit = False
    t = None

    #argument parsing
    opt = SpoutOptions().parse()
    opt.spout_out = [str(item)for item in opt.spout_out.split(' ')]
    if opt.spout_mask_out:
        opt.spout_out.append(opt.spout_mask_out)
    opt.no_instance = True
    if opt.dataset_mode == "coco":
        opt.label_nc = 183
        opt.load_size = 256
        opt.crop_size = 256
    if opt.name == "landscape":
        opt.load_size = 512
        opt.crop_size = 512
    
    #start spout
    sro = Spout(opt)

    #load model for spade
    model = Pix2PixModel(opt)
    model.eval()

    #start threading for scaling if neccesary
    scaling_ratio = int(opt.spout_size[0] / opt.load_size)
    if(scaling_ratio > 1 and opt.ml_scaling):
        print("ml scaling enabled, scaling_ratio : " + str(scaling_ratio))
        t = threading.Thread(target=image_scaler,args=(opt.denoise,scaling_ratio,process_queue,scaled_image_queue,isExit))
        t.start()
    
    #load model for masking
    if opt.spout_mask_out:
        # --------- 1. get image path and name ---------
        mask_model_name = opt.mask_trained_model
        mask_model_dir = './u2net/saved_models/' + mask_model_name + '/' + mask_model_name + '.pth'

        # --------- 3. model define ---------
        if(mask_model_name=='u2net'):
            print("...load U2NET---173.6 MB")
            mask_net = U2NET(3,1)
        elif(mask_model_name=='u2netp'):
            print("...load U2NEP---4.7 MB")
            mask_net = U2NETP(3,1)
        mask_net.load_state_dict(torch.load(mask_model_dir))
        if torch.cuda.is_available():
            mask_device = torch.device("cuda")
            mask_net.cuda()
        mask_net.eval()

    #process incoming frames from spout
    prev_frame = None
    while True:
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
            if(opt.ml_scaling):
                process_queue.put(im_rgb)
                im_rgb = scaled_image_queue.get()
            else:
                im_rgb = cv2.resize(im_rgb, (opt.spout_size[0],opt.spout_size[1]), interpolation = cv2.INTER_LANCZOS4) 
                im_rgb = unsharp_mask(im_rgb)
            cv2_display_frame = im_rgb
            if opt.spout_out:
                if opt.spout_mask_out:
                    mask = mask_generator(im_rgb,mask_device,mask_net,opt)
                    sro.SendSpoutFrame([im_rgb,mask], opt)
                    if(opt.mask_substract):
                        mask = cv2.bitwise_and(im_rgb, mask)
                    cv2_display_frame = np.hstack((im_rgb, mask))
                else:
                    sro.SendSpoutFrame([im_rgb], opt)
            cv2.putText(cv2_display_frame, "fps: " + str(int(1.0 / (time.time() - start_time))),
                        (int(frame.shape[1] / 2 - 50), frame.shape[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Generated", cv2_display_frame)
                


            
    print("Error : no image received")
    os._exit(1)
if __name__ == '__main__':
    main()

