import sys,os,argparse
from pdb import *
import mmap
import numpy as np

a = np.asarray([1,2,3,4],dtype=np.uint8)
a.tostring()

target_adr = 0xe0c00008
def devmem(target_adr,length=1):
    page_size= os.sysconf("SC_PAGE_SIZE")
    reg_base = int(target_adr // page_size) * page_size
    seek_size= int(target_adr %  page_size)
    map_size = seek_size+length
    print("base adr:%s seek:%s"%(hex(reg_base), hex(seek_size)))

    fd = os.open("/dev/mem", os.O_RDWR|os.O_SYNC)
    mem = mmap.mmap(fd, map_size, mmap.MAP_SHARED, mmap.PROT_READ|mmap.PROT_WRITE, offset=reg_base)
    mem.seek(seek_size, os.SEEK_SET)
    xx=mem.read(length)
    #ii=int.from_bytes(xx, 'little')
    #print("Value at address %s : %s"%(hex(target_adr),hex(ii)))
    print("Value at address %s : "%(hex(target_adr)),np.fromstring(xx,dtype=np.float32))


def s2i(s):return int(s,16)
args = argparse.ArgumentParser('devmem')
args.add_argument("target_adr", type=s2i, default=0x0)
args = args.parse_args()
devmem(args.target_adr,4)
