#! /usr/bin/env python3
import sys,os
import numpy as np
import argparse

from devmemX import *
import cv2


def dump_image_CHW(image,filename):
    print("Dump imag bin", filename,image.shape)
    with open(filename,"w") as f:
        for c in range(image.shape[0]):
            for y in range(image.shape[1]):
                for x in range(image.shape[2]):
                    ui8 = image[c][y][x]
                    f.write("%02x\n"%(ui8))
def dump_image_HWC(image,filename):
    print("Dump imag bin", filename,image.shape)
    with open(filename,"w") as f:
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    ui8 = image[y][x][c]
                    f.write("%02x\n"%(ui8))
def fex(file):
    assert os.path.exists(file), "not found specified file"
    return file
parser = argparse.ArgumentParser(description="parse")
parser.add_argument("-A", "--cntl_addr", type=int, default=0xe0c00000,  help="start cntl_addr default 0xe0c00000")
parser.add_argument("-I", "--image_addr",type=int, default=0xe018c000,  help="start image_addr default 0xe018c000")
parser.add_argument("-w", "--words",  type=int, default=10,    help="register words default 10")
parser.add_argument("-C", "--channels",type=int, default=3,     help="channels NN input")
parser.add_argument("-W", "--width",   type=int, default=352,   help="Width NN input")
parser.add_argument("-H", "--height",  type=int, default=288,   help="Height NN input")
igroup = parser.add_mutually_exclusive_group()
igroup.add_argument("-i", "--image",  type=fex, default=None,  help="Load image format data")
igroup.add_argument("-b", "--bin",    type=fex, default=None,  help="Load .bin text data")
parser.add_argument("-c", "--control",action='store_true',     help="Show control reg")
parser.add_argument("-s", "--start",  action='store_true',     help="On start flag")
parser.add_argument("-X", "--compare",action='store_true',     help="Comarison btn write and read")
parser.add_argument("-S", "--compare_size",type=int,default=8, help="Comarison btn write and read")
parser.add_argument("-d", "--debug",  action='store_true',     help="Debug Flag")
args = parser.parse_args()

if args.image is None and not args.control: args.control = True
if args.start: args.control = True

if args.image is not None or args.bin is not None:
    start_addr = 0xe018c000         # Image area
    start_addr = args.image_addr    # Image area

if args.bin is not None:
    with open(args.bin) as b:
        bin_data = b.read().strip().split()
        bin_data = [ int(i,16) for i in bin_data ]
        bin_data = np.asarray(bin_data,dtype=np.uint8)
    d = bin_data.tostring(None)
    C,H,W   =(args.channels, args.height, args.width)    # input size of NN
    assert C*H*W == len(d), "bin text size is difference H*W(%d) != d(%d)"%(C*H*W,len(d))
    print("from .bin file", args.bin, bin_data.shape, len(d), C*H*W)
    print("0x{:8x} : loading {} {} Bytes {}".format(start_addr,args.bin,len(d),bin_data.shape))

if args.image is not None:
    basename = os.path.basename(os.path.splitext(args.image)[0])
    def rgbgr_image(image):
        image_buf = image.copy()
        swap = image[0,:,:]
        image_buf[0,:,:] = image[2,:,:]
        image_buf[2,:,:] = swap
        return image_buf
    # Image data
    print("* Loading image data on memory")
    H,W   =(args.height, args.width)    # input size of NN
    image = cv2.imread(args.image)      # HWC inputed
    assert image is not None, "Cannot understand image format"
    if args.debug:
        dump_image_HWC(image,                    "image_%s_original.bin"%(basename))
        dump_image_CHW(image.transpose((2,0,1)), "image_%s_transepose.bin"%(basename))
        rgbgr = rgbgr_image(image.transpose((2,0,1)).copy());   # reproducting rgbgr swapping in darknet
        dump_image_HWC(rgbgr,                    "image_%s_rgbgr.bin"%(basename))
    image = cv2.resize(image,(W,H))     # resize image size
    image = image.astype(np.uint8)      # unsigned int 8bits
    image = image.transpose((2,0,1))    # CHW transposed
    d = image.tostring(None)

    if args.debug: dump_image_CHW(image, "image_%s_resize.bin"%(basename))
    print("0x{:8x} : loading {} {} Bytes {}".format(start_addr,args.image,len(d),image.shape))

if args.image is not None or args.bin is not None:

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
    print("* Control registers area: 0x%8x"%args.cntl_addr)
    start_addr = args.cntl_addr   # control reg

    for w in range(args.words):
        addr = start_addr + w * 4
        reg = devmem(addr, 4).read(np.uint32)
        print("0x{:08x} : 0x{:08x}".format(addr,reg[0]))

