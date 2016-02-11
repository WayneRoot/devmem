import numpy as np
import cv2
from fbdraw import fb

fb0 = fb()
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame,0)
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        #out.write(frame)

        fb0.imshow('frame',frame)
        #cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
fb0.close()
cap.release()
cv2.destroyAllWindows()
