import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms
import cv2
import time
from spout import Spout

import numpy as np
from PIL import Image

from u2net.model import U2NET # full size version 173.6 MB
from u2net.model import U2NETP # small version u2net 4.7 MB


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Unet Background substraction')
    parser.add_argument('--trained_model', default='u2net', type=str, help='Model to load')
    parser.add_argument('--mask_video', default=False, dest='mask_video', action='store_true',
                        help='Give a feed with the mask apply the video without the background')
    parser.add_argument('--spout_out', default=False, dest='spout_out', action='store_true',
                        help='Send spout as a video feed')
    parser.add_argument('--spout_size', nargs = 2, type=int, default=[640, 480],
                        help='Width and height of the spout receiver')
    parser.add_argument('--spout_in', type=str, default='TDSyphonSpoutOut',
                        help='Spout receiving name - the name of the sender you want to receive')
    parser.add_argument('--window_size', nargs = 2, type=int, default=[640, 480],
                        help='Width and height of the window')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')

    global args
    args = parser.parse_args(argv)


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def main():
    sro = Spout(args)

    # --------- 1. get image path and name ---------
    model_name = args.trained_model
    model_dir = './u2net/saved_models/' + model_name + '/' + model_name + '.pth'

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        device = torch.device("cuda")
        net.cuda()
    net.eval()

    content_transform = transforms.Compose([
        transforms.ToTensor()
    ])


    # --------- 4. inference for each frame ---------
    while sro.receiving:
        start_time = time.time()
        frame = sro.GetSpoutFrame(args.spout_in)

        resized = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)
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
        imr = im.resize((frame.shape[1], frame.shape[0]), resample=Image.BILINEAR)

        open_cv_image = np.array(imr)

        if args.mask_video:
            res = cv2.bitwise_and(frame, open_cv_image)
            result = res
        else:
            result = open_cv_image

        if args.display_fps:
            cv2.putText(result, "fps: " + str(int(1.0 / (time.time() - start_time))),
                        (int(frame.shape[1] / 2 - 50), frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.6, (205, 92, 92), 2)

        cv2.imshow("Unet", result)
        if args.spout_out:
            sro.SendSpoutFrame(result, args)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        del d1,d2,d3,d4,d5,d6,d7


if __name__ == "__main__":
    parse_args()
    main()
