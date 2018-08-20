import sys,os
from pdb import *
import mmap
import numpy as np

a = np.asarray([1,2,3,4],dtype=np.uint8)
a.tostring()

page_size= 0x1000
reg_base = page_size*0xe0c00
map_size = 32
print("%x"%reg_base)

fd = os.open("/dev/mem", os.O_RDWR|os.O_SYNC)
mem = mmap.mmap(fd, map_size, mmap.MAP_SHARED, mmap.PROT_READ|mmap.PROT_WRITE, offset=reg_base)
mem.seek(0x8, os.SEEK_SET)
xx=mem.read(4)
for x in xx:print("%x"%x)
