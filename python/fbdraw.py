import os,sys
import cv2
import numpy as np

class fb():
    def __init__(self,dev_fb='/dev/fb0',shrink=3):
        self.fb = open(dev_fb,"w")
        virtual_size='/sys/class/graphics/fb0/virtual_size'
        assert os.path.exists(virtual_size)
        with open(virtual_size) as f:
            vw,vh = f.read().strip().split(',')
            self.vw,self.vh = int(vw),int(vh)
        self.alpha = None
        self.shrink= shrink
        self.canvas = np.zeros((int(self.vh/self.shrink), int(self.vw),4), dtype=np.uint8)

    def imshow(self,title,img):
        assert img is not None
        assert len(img.shape) == 3
        img      = cv2.resize(img, (int(self.vw/self.shrink), int(self.vh/self.shrink)))
        if self.alpha is None: self.alpha = np.zeros((img.shape[0],img.shape[1],1),dtype=np.uint8)
        bgra_win = np.concatenate([img,self.alpha],axis=2)
        self.canvas[:,:int(img.shape[1]),:] = bgra_win
        bgra_str = (self.canvas.reshape(-1)).tostring()
        self.fb.seek(0)
        self.fb.write(bgra_str)

    def blank(self):
        bgra = np.zeros((self.vh, self.vw, 4), dtype=np.uint8)
        bgra_str = (bgra.reshape(-1)).tostring()
        self.fb.seek(0)
        self.fb.write(bgra_str)

    def close(self):
        self.fb.close()

if __name__ == '__main__':
    img = cv2.imread('dog.jpg')
    fb = fb()
    while True:
        fb.imshow('images',img)
    fb.close()
