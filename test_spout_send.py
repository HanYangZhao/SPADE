from spout import Spout
import cv2
import numpy as np
from PIL import Image,ImageOps
import argparse
import time


"""parsing and configuration"""
def parse_args():
    desc = "Spout for Python texture receiving example"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--spout_size', nargs = 2, type=int, choices=[256,512,1024,2048],default=[256, 256], help='Width and height of the spout receiver')
    parser.add_argument('--spout_in', type=str, default='spout_mask', help='Spout receiving name - the name of the sender you want to receive')
    parser.add_argument('--spout_out', type=str, nargs='+',default='spout_receiver_in', help='the names of the channels you want to send, quoted and seperated by space')
    parser.add_argument('--window_size', nargs = 2, type=int, default=[10, 10], help='Width and height of the window')
    parser.add_argument('--off_by_one_fix', action='store_true',  help='increase all rgb by 1')
    parser.add_argument('input', type=str, help="input file or stream")
    parser.add_argument('mode', type=str, choices=["video","image"], help="input source type")
    parser = parser.parse_args()
    parser.spout_out = [str(item)for item in parser.spout_out.split(' ')]
    return parser

def main():
    # parse arguments
    args = parse_args()
    sro = Spout(args)
    if args.mode == "image":
        frame = cv2.imread(args.input)
        frame = cv2.resize(frame, (args.spout_size[0],args.spout_size[1])) 
    else:
        cap = cv2.VideoCapture(args.input) 
        fps = cap.get(cv2.CAP_PROP_FPS)
    while(True):
        if args.mode == "video":
            now = time.time()
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (args.spout_size[0],args.spout_size[1])) 
                if args.off_by_one_fix:
                    frame = frame + 1
                cv2.imshow(args.spout_out[0], frame)
                sro.SendSpoutFrame([frame], args)
            else:
                print("error, could not read frame")
                break;

            timeDiff = time.time() - now
            if (timeDiff < 1.0/(fps)):
                time.sleep(1.0/(fps) - timeDiff)
        else:
            cv2.imshow(args.spout_out[0], frame)
            sro.SendSpoutFrame([frame], args)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if args.mode == "video":
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()