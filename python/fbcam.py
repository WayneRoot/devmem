import numpy as np
import cv2
import os,sys
from time import time
from fbdraw import fb

fb0 = fb()
cap = cv2.VideoCapture(0)

cnt = 0
start = time()
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame,0)
    if ret==True:
        frame = cv2.flip(frame,0)

        fb0.imshow('frame',frame)
        #cv2.imshow('frame',frame)
        cnt+=1
        elapsed = time() - start
        sys.stdout.write('\b'*30)
        sys.stdout.write("%.3fFPS"%(cnt/elapsed))
        sys.stdout.flush()
    else:
        break

# Release everything if job is finished
fb0.close()
cap.release()
cv2.destroyAllWindows()
