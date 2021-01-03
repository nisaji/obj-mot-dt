from vidgear.gears import CamGear
import cv2
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl
import time


def prep_image(img, inp_dim):
    #Process images for CNN
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim
 
def write(x, img):
    # Display results on frame
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img
 
def arg_parse():
    parser = argparse.ArgumentParser(description='Youtube Live object detection')
    parser.add_argument('arg1', help='Youtube video URL')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()


cfgfile = "cfg/yolov3.cfg" # config file
weightsfile = "yolov3.weights" # weight file
num_classes = 80

args = arg_parse()
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

bbox_attrs = 5 + num_classes

model = Darknet(cfgfile) #Create model.
model.load_weights(weightsfile) # Load weight in model.

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])

assert inp_dim % 32 == 0
assert inp_dim > 32

if CUDA:
    model.cuda() #If CUDA is available, start-up cuda.

model.eval()

url = args.arg1

stream = CamGear(source=url).start() # YouTube Video input
avg = None
frames = 0
start = time.time()

# Infinite loop
while True:
    frame = stream.read()
    if frame is None:
        break
    else:
        img, orig_im, dim = prep_image(frame, inp_dim)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        
        # Convert frame to gray-scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Get framer for comparing with next frame
        if avg is None:
            avg = gray.copy().astype("float")
            continue

        # Calcurate frame delta
        cv2.accumulateWeighted(gray, avg, 0.8)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        # frameDelta_sum = frameDelta.sum()
        thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        cv2.imshow("frame", frame)

        # Discplay FPS
        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", orig_im)


        output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
        output[:,[1,3]] *= frame.shape[1]
        output[:,[2,4]] *= frame.shape[0]

        classes = load_classes('data/coco.names')
        colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x: write(x, orig_im), output))

        cv2.imshow("frame", orig_im)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
        frames += 1
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):

        break

cv2.destroyAllWindows()
# close output window

stream.stop()
# safely close video stream.
