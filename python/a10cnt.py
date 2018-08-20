#! /usr/bin/env python3
import sys,os
import numpy as np
import argparse

from devmemX import *
import cv2


def fex(file):
    assert os.path.exists(file), "not found specified file"
    return file
parser = argparse.ArgumentParser(description="parse")
parser.add_argument("-A", "--address",type=int, default=0xe0c00000,  help="start address default 0xe0c00000")
parser.add_argument("-w", "--words",  type=int, default=10,    help="register words default 10")
parser.add_argument("-W", "--width",  type=int, default=352,   help="Width NN input")
parser.add_argument("-H", "--height", type=int, default=288,   help="Height NN input")
parser.add_argument("-i", "--image",  type=fex, default=None,  help="Load image data")
parser.add_argument("-c", "--control",action='store_true',     help="Show control reg")
parser.add_argument("-s", "--start",  action='store_true',     help="On start flag")
parser.add_argument("-C", "--compare",action='store_true',     help="Comarison btn write and read")
parser.add_argument("-S", "--compare_size",type=int,default=8, help="Comarison btn write and read")
args = parser.parse_args()

if args.image is None and not args.control: args.control = True
if args.start: args.control = True

if args.image is not None:
    # Image data
    print("* Loading image data on memory")
    start_addr = 0xe018c000             # image area
    H,W   =(args.height, args.width)    # input size of NN
    image = cv2.imread(args.image)      # HWC inputed
    assert image is not None, "Cannot understand image format"
    image = cv2.resize(image,(W,H))     # resize image size
    image = image.astype(np.uint8)      # unsigned int 8bits
    image = image.transpose((2,0,1))    # CWH transposed
    d = image.tostring(None)

    print("0x{:8x} : loading {} {} Bytes {}".format(start_addr,args.image,len(d),image.shape))
    devmem(start_addr,len(d)).write(d).close()
    if args.compare:
        print("* Check image data and data on memory")
        reg = devmem(start_addr, len(d)).read(np.uint8)
        comp= args.compare_size
        print(" image data in file:",image.reshape(-1)[:comp],"...")
        print(" image data on mem :",reg[:comp],"...")

if args.start:
    print("* Start flag on")
    d = np.asarray([0x1],dtype=np.uint32).tostring()
    devmem(0xe0c00004,len(d)).write(d).close()

if args.control:
    # Show Control registers
    print("* Control registers area: 0x%8x"%args.address)
    start_addr = args.address   # control reg

    for w in range(args.words):
        addr = start_addr + w * 4
        reg = devmem(addr, 4).read(np.uint32)
        print("0x{:08x} : 0x{:08x}".format(addr,reg[0]))

