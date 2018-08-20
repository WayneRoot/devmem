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
parser.add_argument("-i", "--image",  type=fex, default=None,  help="Load image data")
parser.add_argument("-c", "--control",action='store_true',     help="Show control reg")
parser.add_argument("-s", "--start",  action='store_true',     help="On start flag")
args = parser.parse_args()

if args.image is None and not args.control: args.control = True
if args.start: args.control = True

if args.image is not None:
    # Image data
    start_addr = 0xe018c000             # image area
    image = cv2.imread(args.image)      # HWC inputed
    assert image is not None, "Cannot understand image format"
    image = image.astype(np.uint8)      # unsigned int 8bits
    image = image.transpose((2,0,1))    # CWH transposed
    d = image.tostring(None)

    print("0x{:8x} : loading {} {} Bytes {}".format(start_addr,args.image,len(d),image.shape))
    devmem(start_addr,len(d)).write(d).close()
    #reg = devmem(start_addr, len(d)).read(np.uint8)

if args.start:
    d = np.asarray([0x1],dtype=np.uint32).tostring()
    devmem(0xe0c00004,len(d)).write(d).close()

if args.control:
    # Show Control registers
    print("start address to memory: 0x%8x"%args.address)
    start_addr = args.address   # control reg

    for w in range(10):
        addr = start_addr + w * 4
        reg = devmem(addr, 4).read(np.uint32)
        print("0x{:08x} : 0x{:08x}".format(addr,reg[0]))

