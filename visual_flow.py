#!/usr/bin/env python2.7

from __future__ import print_function

import os, sys, numpy as np
import cv2
import argparse
from scipy import misc
sys.path.insert(0, '/home/szf/flownet2/python')  
import caffe
import tempfile
from math import ceil
from lib import flowlib
import time

def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)

def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)


def generateFlow(args):
    sub_folder = args.video.split('/')[-2]
    folder = args.root_folder + sub_folder + "/"
    
    if args.write_flow and not os.path.exists(folder):
        os.mkdir(folder)
    index = 0
    img0 = img1 = None

    cap = cv2.VideoCapture(args.video)
    ret0, img0 = cap.read()
    if not ret0:
        print('no frame')
        cap.release()
        return
        
    img0 = cv2.resize(img0, (width, height))

    while(cap.isOpened()):
        start = time.time()

        # skip some frame for faster inferance
        sk = 0
        while sk < args.step:
            sk = sk + 1
            ret1, img1 = cap.read()
            if not ret1:
                break;
        if sk < args.step:
            print("no frame")
            break;

        img1 = cv2.resize(img1, (width, height))
         
        num_blobs = 2
        input_data = []
        if len(img0.shape) < 3: 
            input_data.append(img0[np.newaxis, np.newaxis, :, :])
        else:                   
            input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        if len(img1.shape) < 3: 
            input_data.append(img1[np.newaxis, np.newaxis, :, :])
        else:                   
            input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    
        input_dict = {}
        for blob_idx in range(num_blobs):
            input_dict[net.inputs[blob_idx]] = input_data[blob_idx]
    
        #print('Network forward pass using %s.' % args.caffemodel)
        i = 1
        while i<=5:
            i+=1
            net.forward(**input_dict)
            containsNaN = False
            for name in net.blobs:
                blob = net.blobs[name]
                has_nan = np.isnan(blob.data[...]).any()
    
                if has_nan:
                    print('blob %s contains nan' % name)
                    containsNaN = True
    
            if not containsNaN:
                print('Succeeded.')
                break
            else:
                print('**************** FOUND NANs, RETRYING ****************')
    
        blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        if args.write_flow:
            flow_name = folder + "%(idx)05d.flo"%{'idx':index}
            #print(flow_name)
            writeFlow(flow_name, blob)
            index += 1

        visual_flow = flowlib.flow_to_image(blob)
        cv2.imshow('flow', visual_flow)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
        img0 = img1
        end = time.time()
        print(end-start)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caffemodel', help='path to model')
    parser.add_argument('--deployproto', help='path to deploy prototxt template')
    parser.add_argument('--video', help='unflowed video')
    parser.add_argument('--root-folder', help='unflowed video', default="./")
    parser.add_argument('--step', type=int, help='step of frame', default=4)
    parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
    parser.add_argument('--write-flow',  help='whether to write flow to disk', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if(not os.path.exists(args.caffemodel)): raise BaseException('caffemodel does not exist: '+args.caffemodel)
    if(not os.path.exists(args.deployproto)): raise BaseException('deploy-proto does not exist: '+args.deployproto)
    if(not os.path.exists(args.video)): raise BaseException('video does not exist: '+args.visual_video)
    width = 320 
    height = 240
    
    vars = {}
    vars['TARGET_WIDTH'] = width
    vars['TARGET_HEIGHT'] = height
    
    divisor = 64.
    vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
    vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)
    
    vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
    vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);
    
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    
    proto = open(args.deployproto).readlines()
    for line in proto:
        for key, value in vars.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))
    
        tmp.write(line)
    
    tmp.flush()
    
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)
    
    generateFlow(args)
   

