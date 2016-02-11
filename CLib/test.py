import os,sys,re
import numpy as np
from   devmemX import devmem
import cv2
from time import sleep
from   pdb import *
import dn

n_classes = 20
grid_h    =  9
grid_w    = 11
box_coord =  4
n_b_boxes =  5
n_info_per_grid = box_coord + 1 + n_classes

classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
colors = [(254.0, 254.0, 254), (239.8, 211.6, 127), 
          (225.7, 169.3, 0),   (211.6, 127.0, 254),
          (197.5, 84.6, 127),  (183.4, 42.3, 0),
          (169.3, 0.0, 254),   (155.2, -42.3, 127),
          (141.1, -84.6, 0),   (127.0, 254.0, 254), 
          (112.8, 211.6, 127), (98.7, 169.3, 0),
          (56.4, 42.3, 0),     (42.3, 0.0, 254), 
          (84.6, 127.0, 254),  (70.5, 84.6, 127),
          (28.2, -42.3, 127),  (14.1, -84.6, 0),
          (0.0, 254.0, 254),   (-14.1, 211.6, 127)]

# YOLOv2 anchor of Bounding-Boxes
anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]

def preprocessing(input_img,ph_height,ph_width):

  input_image = input_img                           # HWC BGR
  #input_image    = cv2.imread(input_img_path)        # HWC BGR

  #resized_image  = cv2.resize(input_image,(ph_width, ph_height), interpolation = cv2.INTER_CUBIC)
  resized_image  = cv2.resize(input_image,(ph_width, ph_height))

  resized_image  = cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB)

  resized_chwRGB = resized_image.transpose((2,0,1))  # CHW RGB

  #resized_chwRGB /= 255.

  image_nchwRGB  = np.expand_dims(resized_chwRGB, 0) # NCHW BGR

  #return input_image
  return image_nchwRGB

def box2rect(box):
    x, y, h, w = box
    lx, ly, rx, ry = x-w/2., y-h/2., x+w/2., y+h/2.
    if lx < 0: lx =0.
    if ly < 0: ly =0.
    return (int(lx), int(ly), int(rx), int(ry))

def main():

	# Definition of the paths
    input_img_path    = './dog.jpg'
    input_img_path    = './horses.jpg'
    input_img_path    = './person.jpg'
    output_image_path = './result.jpg'

    # Definition of the parameters
    ph_height = 288 # placeholder height
    ph_width  = 352 # placeholder width
    score_threshold = 0.3
    iou_threshold = 0.3

    verbose=True
    # Preprocess the input image
    print('Preprocessing...')
    input_image = cv2.imread(input_img_path)        # HWC BGR
    preprocessed_nchwRGB = preprocessing(input_image, ph_height, ph_width)
    cnt=0
    d = preprocessed_nchwRGB.reshape(-1).astype(np.uint8).tostring()
    devmem(0xe018c000,len(d),verbose=verbose).write(d).close()

    print("start FPGA accelerator")
    s = np.asarray([0x1],dtype=np.uint32).tostring()
    devmem(0xe0c00004,len(s)).write(s).close()
    #sleep(1)
    for i in range(10000):
        mem = devmem(0xe0c00008,0x4)
        status = mem.read(np.uint32)
        mem.close()
        if status[0] == 0x2000:break
    print("fpga status:0x%08x"%(status[0]))
    print("preprocessing to NCHW-RGB",preprocessed_nchwRGB.shape)

    # Compute the predictions on the input image
    print('Computing predictions...')
    mem = devmem(0xe0000000,0xc15c)
    predictions = mem.read(np.float32)
    mem.close()
    assert predictions[0]==predictions[0],"invalid mem values:{}".format(predictions[:8])
    print("inference from FPGA",predictions.shape)

#   _predictions________________________________________________________
#   | 4 entries                 |1 entry |     20 entries               |
#   | x..x | y..y | w..w | h..h | c .. c | p0 - p19      ..     p0 - p19| x 5(==num)
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   entiry size == grid_w x grid_h

    print("predictions.shape",predictions.shape)

    # Postprocess the predictions and save the output image
    print('Postprocessing...')
    im_h, im_w = input_image.shape[:2]
    res = dn.postprocessing(predictions,im_w,im_h,0.5,0.5)
    # res: object_name, confidence, ( centerX, centerY, boxW, boxH )
    for r in res:
        name, conf, bbox = r[:3]
        obj_col = colors[classes.index(r[0])]
        rect = box2rect(bbox)
        print("{}".format(bbox))
        print("{}".format(rect))
        cv2.rectangle(
            input_image,
            ( rect[0], rect[1] ),
            ( rect[2], rect[3] ),
            #(255,255,255)
            obj_col
        )
        cv2.putText(
            input_image,
            name,
            (int(bbox[0]), int(bbox[1])),
            cv2.FONT_HERSHEY_SIMPLEX,1,
            #(255,255,255)
            obj_col,
            2)
    cv2.imwrite('result.jpg',input_image)

if __name__ == '__main__':
     main() 

