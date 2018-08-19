#! /usr/bin/env python3
import numpy as np
# import chainer
# from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
# from chainer import Link, Chain, ChainList
# import chainer.functions as F
# import chainer.links as L
# from chainer import training
# from chainer.training import extensions
import argparse
#from lib.utils import *
#from lib.image_generator import *
# from yolov2_orig import *

from devmemX import *

parser = argparse.ArgumentParser(description="parse")
parser.add_argument('file', default="yolov2-tiny-voc_352_288_final.weights", help="path")
args = parser.parse_args()

print("loading", args.file)
file = open(args.file, "rb")
#dat=np.fromfile(file, dtype=np.float32)[5:] # skip header(4xint)
dat=np.fromfile(file, dtype=np.float32)[4:] # skip header(4xint)

# load model
print("loading initial model...")
n_classes = 20
n_boxes = 5
last_out = (n_classes + 5) * n_boxes

#yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
#yolov2.train = True
#yolov2.finetune = False

param_adr = 0xc0000000  # param address DDR
layers=[
    [3, 16, 3], 
    [16, 32, 3], 
    [32, 64, 3], 
    [64, 128, 3], 
    [128, 256, 3], 
    [256, 512, 3], 
    [512, 1024, 3], 
    [1024,1024, 1], 
]

offset = 0x0
for i, l in enumerate(layers):
    in_ch, out_ch, ksize = l
    print("[ Layer",i,":", in_ch, out_ch, ksize, ksize, "]")

    # load bias
    # txt = "yolov2.bias%d.b.data = dat[%d:%d]" % (i+1, offset, offset+out_ch)
    bias_buff = np.zeros((out_ch, in_ch), dtype=np.float32)
    bias_oCiC = dat[offset: offset+out_ch].reshape((out_ch))
    for o in range(out_ch):
        for i in range(in_ch):
            bias_buff[o][i] = bias_oCiC[o]
    bias_buff = bias_buff.transpose((1, 0))
    d = bias_buff.tostring()
    print("0x{:08x} : write Bytes {:14d} bias {}".format(param_adr,len(d),bias_buff.shape))
    devmem(param_adr,len(d)).write(d).close()
    param_adr+=len(d)
    offset+=out_ch

    # load gamma
    # txt = "yolov2.bn%d.gamma.data = dat[%d:%d]" % (i+1, offset, offset+out_ch)
    gamma_buff = np.zeros((out_ch, in_ch), dtype=np.float32)
    gamma_oCiC = dat[offset: offset+out_ch].reshape((out_ch))
    for o in range(out_ch):
        for i in range(in_ch):
            gamma_buff[o][i] = gamma_oCiC[o]
    gamma_buff = gamma_buff.transpose((1, 0))
    d = gamma_buff.tostring()
    print("0x{:08x} : write Bytes {:14d} gamma {}".format(param_adr,len(d),gamma_buff.shape))
    offset+=out_ch

    # load mean
    # txt = "yolov2.bn%d.avg_mean = dat[%d:%d]" % (i+1, offset, offset+out_ch)
    mean_buff = np.zeros((out_ch, in_ch), dtype=np.float32)
    mean_oCiC = dat[offset: offset+out_ch].reshape((out_ch))
    for o in range(out_ch):
        for i in range(in_ch):
            mean_buff[o][i] = mean_oCiC[o]
    mean_buff = mean_buff.transpose((1, 0))
    d = mean_buff.tostring()
    print("0x{:08x} : write Bytes {:14d} mean {}".format(param_adr,len(d),mean_buff.shape))
    offset+=out_ch

    # load variance
    # txt = "yolov2.bn%d.avg_var = dat[%d:%d]" % (i+1, offset, offset+out_ch)
    variance_buff = np.zeros((out_ch, in_ch), dtype=np.float32)
    variance_oCiC = dat[offset: offset+out_ch].reshape((out_ch))
    for o in range(out_ch):
        for i in range(in_ch):
            variance_buff[o][i] = variance_oCiC[o]
    variance_buff = variance_buff.transpose((1, 0))
    d = variance_buff.tostring()
    print("0x{:08x} : write Bytes {:14d} variance {}".format(param_adr,len(d),variance_buff.shape))
    offset+=out_ch

    #S = gamma_iCoC / np.sqrt( variance_iCoC )
    #B = gamma_iCoC * mean_iCoC / np.sqrt( variance_iCoC )

    # load Weight
    # txt = "yolov2.conv%d.W.data = dat[%d:%d].reshape(%d, %d, %d, %d)" % (i+1, offset, offset+(out_ch*in_ch*ksize*ksize), out_ch, in_ch, ksize, ksize)
    weight_oCiCkSkS = dat[offset: offset+out_ch*in_ch*ksize*ksize].reshape((out_ch, in_ch, ksize, ksize))
    weight_iCoCkSkS = weight_oCiCkSkS.transpose((1, 0, 2, 3))
    d = weight_iCoCkSkS.tostring()
    print("0x{:08x} : write Bytes {:14d} weight {}".format(param_adr,len(d),weight_iCoCkSkS.shape))
    offset+= (out_ch*in_ch*ksize*ksize)

    #print(i+1, offset)

# load last convolution weight
in_ch = 1024
out_ch = last_out
ksize = 1
print("[ Last Layer",":",in_ch, out_ch, ksize, ksize, "]")

# txt = "yolov2.bias%d.b.data = dat[%d:%d]" % (i+2, offset, offset+out_ch)
bias_buff = np.zeros((out_ch, in_ch), dtype=np.float32)
bias_oCiC = dat[offset: offset+out_ch].reshape((out_ch))
for o in range(out_ch):
    for i in range(in_ch):
        bias_buff[o][i] = bias_oCiC[o]
bias_buff = bias_buff.transpose((1, 0))
d = bias_buff.tostring()
print("0x{:08x} : write Bytes {:14d} bias {}".format(param_adr,len(d),bias_buff.shape))
offset+=out_ch

# txt = "yolov2.conv%d.W.data = dat[%d:%d].reshape(%d, %d, %d, %d)" % (i+2, offset, offset+(out_ch*in_ch*ksize*ksize), out_ch, in_ch, ksize, ksize)
weight_oCiCkSkS = dat[offset: offset+out_ch*in_ch*ksize*ksize].reshape((out_ch, in_ch, ksize, ksize))
weight_iCoCkSkS = weight_oCiCkSkS.transpose((1, 0, 2, 3))   # IOKK
d = weight_iCoCkSkS.tostring()
print("0x{:08x} : write Bytes {:14d} weight {}".format(param_adr,len(d),weight_iCoCkSkS.shape))
devmem(0xe018c000,len(d)).write(d).close()
offset+=out_ch*in_ch*ksize*ksize

#print(i+2, offset)

#print("save weights file to yolov2_darknet_hdf5.model")
#serializers.save_hdf5("yolov2_darknet_hdf5.model", yolov2)
#print("save weights file to yolov2_darknet.model")
#serializers.save_npz("yolov2_darknet.model", yolov2)

