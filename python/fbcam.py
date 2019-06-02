import numpy as np
import cv2
import os,sys
import argparse
from time import time
from fbdraw import fb

args=argparse.ArgumentParser()
args.add_argument('-c', '--cv',action='store_true')
args.add_argument('-s', '--shrink',type=int,default=3,choices=[1,2,3])
args.add_argument('-bg','--background',type=str,default='debian2.jpg')
args=args.parse_args()
video_fb = True if args.cv is not True else False
print(video_fb)

if video_fb: fb0 = fb(shrink=args.shrink)
if video_fb: fbB = fb(shrink=1)
if video_fb:
    os.system('clear')
    fbB = fb(shrink=1)
    assert os.path.exists(args.background)
    background = cv2.imread(args.background)
    fbB.imshow('back',background)
    fbB.close()
cap = cv2.VideoCapture(0)
print "Hitachi Solutions Technology"
print("cam.property-default:",cap.get(3),cap.get(4))
cap.set(3,640)  # 3:width
cap.set(4,480)  # 4:height
print("cam.property-set:",cap.get(3),cap.get(4))

cnt = 0
start = time()
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame,0)
    if ret==True:
        frame = cv2.flip(frame,0)

        if video_fb is True :fb0.imshow('frame',frame)
        if video_fb is False:cv2.imshow('frame',frame)
        cnt+=1
        elapsed = time() - start
        sys.stdout.write('\b'*30)
        sys.stdout.write("%.3fFPS"%(cnt/elapsed))
        sys.stdout.flush()
        if video_fb is False and cv2.waitKey(1) != -1:break
    else:
        break

# Release everything if job is finished
if video_fb: fb0.close()
cap.release()
cv2.destroyAllWindows()
