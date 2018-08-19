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
            self.length+self.seek_size,
            mmap.MAP_SHARED,
            mmap.PROT_READ|mmap.PROT_WRITE,
            offset=self.reg_base
        )
        self.mem.seek(self.seek_size, os.SEEK_SET)

    def write(self, datas):
        self.mem.write(datas)
        self.mem.flush(0,0)
        return self

    def read(self, types):
        type_bytes = len(np.asarray(0,dtype=type).tostring())
        #assert self.length<=4, 'length > 4 causes system freeze'
        for i in range(0, self.length, type_bytes):
            datas = self.mem.read(type_bytes)
            array = np.fromstring(datas,dtype=types)
            #if self.verbose:print("Value at address %s : %s"%(hex(self.target_adr),hex(ii)))
            #if self.verbose:print("Value at address %s : %s"%(hex(self.target_adr),float(ii)))
            if self.verbose:print("Value at address {} : {}".format(hex(self.target_adr+i),array))
    def close(self):
        self.mem.close()

if __name__=='__main__':
    def s2i(s):return int(s,16)
    args = argparse.ArgumentParser('devmem')
    args.add_argument("target_adr",     type=s2i, default=0xe018c000,    help="0xe018c000 for Image")
    args.add_argument("-s", "--size",   type=int, default=4,             help="bytes default 4")
    args.add_argument("-t", "--type",   type=str, default=np.float32,    help="type default numpy.float32")
    args.add_argument("-f", "--weights",type=str, default="yolo.weights",help="weights default yolo.weights")
    args.add_argument("-w", "--write",  action='store_true',             help="write default read")
    args = args.parse_args()

    if args.write:
        d = np.arange(0,1024*2).astype(np.float32).reshape(4,256,2)
        d = d.tostring()
        print("write Bytes",len(d))
        devmem(args.target_adr,len(d)).write(d).close()
    else:
        print("read Bytes",args.size)
        str = "devmem(0x{:08x},{}).read({})".format(args.target_adr, args.size, args.type)
        print(str)
        exec(str)

