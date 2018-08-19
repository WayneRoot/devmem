import sys,os,argparse
from pdb import *
import mmap
import numpy as np

a = np.asarray([1,2,3,4],dtype=np.uint8)
a.tostring()

class devmem():
    def __init__(self, target_adr, length, verbose=True):
        self.verbose   = verbose
        self.target_adr= target_adr
        self.length    = length
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        self.reg_base  = int(target_adr // self.page_size) * self.page_size
        self.seek_size = int(target_adr %  self.page_size)
        if self.verbose:print("base adr:%s seek:%s"%(hex(self.reg_base), hex(self.seek_size)))

        self.fd  = os.open("/dev/mem", os.O_RDWR|os.O_SYNC)
        self.mem = mmap.mmap(
            self.fd,
            self.length,
            mmap.MAP_SHARED,
            mmap.PROT_READ|mmap.PROT_WRITE,
            offset=self.reg_base
        )
        #self.mem.seek(self.seek_size, os.SEEK_SET)
        self.mem.seek(self.seek_size)

    def write(self, datas):
        self.mem.write(datas)
        self.mem.flush(0,0)
        return self

    def read(self, type):
        assert self.length<=4, 'length > 4 causes system freeze'
        datas = self.mem.read(self.length)
        array = np.fromstring(datas,dtype=type)
        #if self.verbose:print("Value at address %s : %s"%(hex(self.target_adr),hex(ii)))
        #if self.verbose:print("Value at address %s : %s"%(hex(self.target_adr),float(ii)))
        if self.verbose:print("Value at address %s :"%(hex(self.target_adr)),array)
    def close(self):
        self.mem.close()

if __name__=='__main__':
    def s2i(s):return int(s,16)
    args = argparse.ArgumentParser('devmem')
    args.add_argument("target_adr", type=s2i, default=0x0)
    args = args.parse_args()
    d = np.asarray([0.5, 1],dtype=np.float32)
    d = d.tostring()
    #print(len(d))
    #d = bytes(d)
    #devmem(args.target_adr,len(d)).write(d).close()
    devmem(args.target_adr,4).read(np.float32)
