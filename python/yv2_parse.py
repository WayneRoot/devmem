#! /usr/bin/env python3
import numpy as np
# import chainer
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
parser.add_argument("-f", "--file", default="yolov2-tiny-voc_352_288_final.weights", help="path")
parser.add_argument("-A", "--address",type=int, default=0xc0000000,  help="start address default 0xc0000000")
parser.add_argument("-S", "--scale",  type=np.float32, default=None, help="force scale value")
parser.add_argument("-B", "--bias",   type=np.float32, default=None, help="force bias  value")
parser.add_argument("-W", "--weight", type=np.float32, default=None, help="force weight value")
parser.add_argument("-wo","--wonly",action="store_true", help="write weight only")
args = parser.parse_args()

print("start address to memory:",args.address)
if args.bias   is not None:print("force bias to",   args.bias)
if args.scale  is not None:print("force scane to",  args.scale)
if args.weight is not None:print("force weight to", args.weight)

print("loading #1", args.file)
infile = open(args.file, "rb")
dat_org=np.fromfile(infile, dtype=np.int32)
infile.close()
(major, minor, revision)= dat_org[:3]
if major*10+minor >= 2 and major < 1000 and minor < 1000:
    skipB = int((4+4+4+8)/4)
    print("Training 64bit",skipB)
else:
    skipB = int((4+4+4+4)/4)
    print("Training 32bit",skipB)

# load model
print("loading initial model...")
n_classes = 20
n_boxes = 5
last_out = (n_classes + 5) * n_boxes

param_adr_start = param_adr = args.address  # param address DDR
layers=[
    [3, 16, 3], 
    [16, 32, 3], 
    [32, 64, 3], 
    [64, 128, 3], 
    [128, 256, 3], 
    [256, 512, 3], 
    [512, 1024, 3], 
    [1024,1024, 3], 
]

# for a convolutional layer
for loadNo in range(2):
    offset = 0x0
    print("loading #2", args.file)
    infile = open(args.file, "rb")
    dat=np.fromfile(infile, dtype=np.float32)[skipB:] # skip header(4xint)
    infile.close()

    for i, l in enumerate(layers):
        in_ch, out_ch, ksize = l
        print("[ Layer",i,": IOKK %5d%5d%5d%5d ]"%(in_ch, out_ch, ksize, ksize))

        # load bias
    #    bias_buff = np.zeros((out_ch, in_ch), dtype=np.float32)
        bias_buff = np.zeros((out_ch), dtype=np.float32)
        bias_oCiC = dat[offset: offset+out_ch].reshape((out_ch))
    #    for o in range(out_ch):
    #        for i in range(in_ch):
    #            bias_buff[o][i] = bias_oCiC[o]
    #    bias_buff = bias_buff.transpose((1, 0)) # IO
        bias_buff = bias_oCiC
        d = bias_buff.tostring()
        if loadNo==1:print("  {} : read  Bytes {:14d} bias        {}".format(' '*10,len(d),bias_buff.shape))
        offset+=out_ch

        # load gamma
    #    gamma_buff = np.zeros((out_ch, in_ch), dtype=np.float32)
        gamma_buff = np.zeros((out_ch), dtype=np.float32)
        gamma_oCiC = dat[offset: offset+out_ch].reshape((out_ch))
    #    for o in range(out_ch):
    #        for i in range(in_ch):
    #            gamma_buff[o][i] = gamma_oCiC[o]
    #    gamma_buff = gamma_buff.transpose((1, 0))   # IO
        gamma_buff = gamma_oCiC
        d = gamma_buff.tostring()
        if loadNo==1:print("  {} : read  Bytes {:14d} gamma       {}".format(' '*10,len(d),gamma_buff.shape))
        offset+=out_ch

        # load mean
    #    mean_buff = np.zeros((out_ch, in_ch), dtype=np.float32)
        mean_buff = np.zeros((out_ch), dtype=np.float32)
        mean_oCiC = dat[offset: offset+out_ch].reshape((out_ch))
    #    for o in range(out_ch):
    #        for i in range(in_ch):
    #            mean_buff[o][i] = mean_oCiC[o]
    #    mean_buff = mean_buff.transpose((1, 0)) # IO
        mean_buff = mean_oCiC
        d = mean_buff.tostring()
        if loadNo==1:print("  {} : read  Bytes {:14d} mean        {}".format(' '*10,len(d),mean_buff.shape))
        offset+=out_ch

        # load variance
    #    variance_buff = np.zeros((out_ch, in_ch), dtype=np.float32)
        variance_buff = np.zeros((out_ch), dtype=np.float32)
        variance_oCiC = dat[offset: offset+out_ch].reshape((out_ch))
    #    for o in range(out_ch):
    #        for i in range(in_ch):
    #            variance_buff[o][i] = variance_oCiC[o]
    #    variance_buff = variance_buff.transpose((1, 0)) # IO
        variance_buff = variance_oCiC
        d = variance_buff.tostring()
        if loadNo==1:print("  {} : read  Bytes {:14d} variance    {}".format(' '*10,len(d),variance_buff.shape))
        offset+=out_ch

        # foldint Scale and Bias
        Scaling = gamma_buff / (np.sqrt( variance_buff ) + 0.000001)
        Biassed = -1. * gamma_buff * mean_buff / (np.sqrt( variance_buff ) + 0.000001) + bias_buff
    #    SB_buff = np.zeros((in_ch, out_ch, 2), dtype=np.float32)
        SB_buff = np.zeros((out_ch, 2), dtype=np.float32)
        if loadNo == 1:
            if args.scale is not None:
                Scaling = np.full(Scaling.shape, args.scale, dtype=np.float32)
            if args.bias is not None:
                Biassed = np.full(Biassed.shape, args.bias,  dtype=np.float32)
    #        for i in range(in_ch):
            for o in range(out_ch):
                SB_buff[o][0] = Scaling[o]
                SB_buff[o][1] = Biassed[o]
            d = SB_buff.tostring()
            print("  0x{:08x} : write Bytes {:14d} scale and bias {}".format(param_adr,len(d),SB_buff.shape))
            devmem(param_adr,len(d)).write(d).close()
            param_adr+=len(d)

        # load Weight
        if loadNo == 0:
            weight_oCiCkSkS = dat[offset: offset+out_ch*in_ch*ksize*ksize].reshape((out_ch, in_ch, ksize, ksize))
            weight_iCoCkSkS = weight_oCiCkSkS.transpose((1, 0, 2, 3))   # IOKK
            if args.weight is not None:
                weight_iCoCkSkS = np.full(weight_iCoCkSkS.shape, args.weight, dtype=np.float32)
            d = weight_iCoCkSkS.tostring()
            print("  0x{:08x} : write Bytes {:14d} weight {}".format(param_adr,len(d),weight_iCoCkSkS.shape))
            devmem(param_adr,len(d)).write(d).close()
            param_adr+=len(d)
        offset+= (out_ch*in_ch*ksize*ksize)

        #print(i+1, offset)

    # load last bias
    in_ch = 1024
    out_ch = last_out
    ksize = 1
    print("[ Last Layer",": IOKK %5d%5d%5d%5d ]"%(in_ch, out_ch, ksize, ksize))

    bias_oCiC = dat[offset: offset+out_ch].reshape((out_ch))
    if args.bias is not None:
        bias_oCiC = np.full(bias_oCiC.shape, args.bias, dtype=np.float32)
    d = bias_oCiC.tostring()
    if loadNo==1:
        print("  0x{:08x} : write Bytes {:14d} bias {}".format(param_adr,len(d),bias_oCiC.shape))
        devmem(param_adr,len(d)).write(d).close()
        param_adr+=len(d)
    offset+=out_ch

    # load last convolution weight
    weight_oCiCkSkS = dat[offset: offset+out_ch*in_ch*ksize*ksize].reshape((out_ch, in_ch, ksize, ksize))
    weight_iCoCkSkS = weight_oCiCkSkS.transpose((1, 0, 2, 3))   # IOKK
    if args.weight is not None:
        weight_iCoCkSkS = np.full(weight_iCoCkSkS.shape, args.weight, dtype=np.float32)
    d = weight_iCoCkSkS.tostring()
    if loadNo==0:
        print("  0x{:08x} : write Bytes {:14d} weight {}".format(param_adr,len(d),weight_iCoCkSkS.shape))
        devmem(param_adr,len(d)).write(d).close()
        param_adr+=len(d)
    offset+=out_ch*in_ch*ksize*ksize

    print("* Last Address 0x%x"%param_adr)

    if args.wonly:break

print("*\n* Summary")
print("* parametor : %dB / infile size : %dB"%(param_adr-param_adr_start, (len(dat)+skipB)*4))

