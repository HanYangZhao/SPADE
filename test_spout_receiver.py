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

class SpoutOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.add_argument('--spout_size', nargs = 2, type=int, default=[256, 256],
                                    help='Width and height of the spout receiver')
        parser.add_argument('--spout_in', type=str, default='spout_receiver_in',
                                    help='Spout receiving name - the name of the sender you want to receive')
        parser.add_argument('--spout_out', type=str, nargs='+',default='spout_receiver_out', help='the names of the channels you want to send, quoted and seperated by space')
        parser.add_argument('--window_size', nargs = 2, type=int, default=[10, 10],
                                    help='Width and height of the window')
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=1024)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.add_argument('--output_dir', type=str, default='results',
                            help='Directory name to save the generated images')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.isTrain = False
        return parser

def is_same_image(prev_frame,current_frame):
    if prev_frame is None:
        return False
    difference = cv2.subtract(prev_frame,current_frame)
    if cv2.countNonZero(difference) == 0:
        return True
    return False

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


def main():
    opt = SpoutOptions().parse()
    opt.spout_out = [str(item)for item in opt.spout_out.split(' ')]
    opt.no_instance = True
    if opt.dataset_mode == "coco":
        opt.label_nc = 183
        opt.load_size = 256
        opt.crop_size = 256
    if opt.name == "landscape":
        opt.load_size = 512
        opt.crop_size = 512
    sro.InitSpout(opt)

    # width, height = opt.spout_size
    # opt.load_size = width
    # opt.crop_size = width
    model = Pix2PixModel(opt)
    model.eval()
    prev_frame = None
    while sro.receiving:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame = sro.GetSpoutFrame(opt.spout_in)
        if frame is None:
            print("Error : no image received")
            break
        # image = Image.fromarray(frame)
        # label = ImageOps.grayscale(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_same_image(prev_frame,frame):
            continue
        prev_frame = frame
        if(opt.spout_size[0] != opt.crop_size):
                print("resizing input")
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
            im_rgb = cv2.resize(im_rgb, (opt.spout_size[0],opt.spout_size[1]), interpolation = cv2.INTER_LANCZOS4) 
            im_rgb = unsharp_mask(im_rgb)
            if opt.spout_out:
                sro.SendSpoutFrame(im_rgb, opt)
            cv2.putText(im_rgb, "fps: " + str(int(1.0 / (time.time() - start_time))),
            (int(frame.shape[1] / 2 - 50), frame.shape[0]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Generated", im_rgb)
    print("Error : no image received")
    os._exit(1)

if __name__ == '__main__':
    main()

