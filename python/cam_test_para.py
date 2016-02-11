import os,sys,re
import numpy as np
from   devmemX import devmem
import cv2
from time import sleep,time
from fbdraw import fb
from   pdb import *

assert os.path.exists('/dev/fb0') and os.path.exists('/dev/video0')
ph_height = 288 # placeholder height
ph_width  = 352 # placeholder width
ph_chann  = 3
fb0 = fb()
devmem_image = devmem(0xe018c000,ph_height*ph_width*ph_chann)
devmem_start = devmem(0xe0c00004,4)
devmem_stat  = devmem(0xe0c00008,0x4)
devmem_pred  = devmem(0xe0000000,0xc15c)

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
          (225.7, 169.3, 0), (211.6, 127.0, 254),
          (197.5, 84.6, 127), (183.4, 42.3, 0),
          (169.3, 0.0, 254), (155.2, -42.3, 127),
          (141.1, -84.6, 0), (127.0, 254.0, 254), 
          (112.8, 211.6, 127), (98.7, 169.3, 0),
          (84.6, 127.0, 254), (70.5, 84.6, 127),
          (56.4, 42.3, 0), (42.3, 0.0, 254), 
          (28.2, -42.3, 127), (14.1, -84.6, 0),
          (0.0, 254.0, 254), (-14.1, 211.6, 127)]

# YOLOv2 anchor of Bounding-Boxes
anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def iou(boxA,boxB):
  # boxA = boxB = [x1,y1,x2,y2]

  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  # Compute the area of intersection
  intersection_area = (xB - xA + 1) * (yB - yA + 1)
 
  # Compute the area of both rectangles
  boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
  # Compute the IOU
  iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

  return iou



def non_maximal_suppression(thresholded_predictions,iou_threshold):

  nms_predictions = []

  # Add the best B-Box because it will never be deleted
  if len(thresholded_predictions)<=0: return nms_predictions
  nms_predictions.append(thresholded_predictions[0])

  # For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
  # thresholded_predictions[i][0] = [x1,y1,x2,y2]
  i = 1
  while i < len(thresholded_predictions):
    n_boxes_to_check = len(nms_predictions)
    to_delete = False

    j = 0
    while j < n_boxes_to_check:
        curr_iou = iou(thresholded_predictions[i][0],nms_predictions[j][0])
        if(curr_iou > iou_threshold ):
            to_delete = True
        j = j+1

    if to_delete == False:
        nms_predictions.append(thresholded_predictions[i])
    i = i+1

  return nms_predictions



def preprocessing(input_image,ph_height,ph_width):

  resized_image  = cv2.resize(input_image,(ph_width, ph_height))

  resized_image  = cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB)

  resized_chwRGB = resized_image.transpose((2,0,1))  # CHW RGB

  #resized_chwRGB /= 255.

  image_nchwRGB  = np.expand_dims(resized_chwRGB, 0) # NCHW BGR

  #return input_image
  return image_nchwRGB



def postprocessing(predictions,input_image,score_threshold,iou_threshold,ph_height,ph_width):

  input_image = cv2.resize(input_image,(ph_width, ph_height), interpolation = cv2.INTER_CUBIC)

  thresholded_predictions = []

  predictions = np.reshape(predictions,(grid_h, grid_w, n_b_boxes, n_info_per_grid))

  for row in range(grid_h):
    for col in range(grid_w):
      for b in range(n_b_boxes):

        tx, ty, tw, th, tc = predictions[row, col, b, :5]

        center_x = (float(col) + sigmoid(tx)) * 32.0
        center_y = (float(row) + sigmoid(ty)) * 32.0

        roi_w = np.exp(tw) * anchors[2*b + 0] * 32.0
        roi_h = np.exp(th) * anchors[2*b + 1] * 32.0

        final_confidence = sigmoid(tc)

        class_predictions = predictions[row, col, b, 5:]
        class_predictions = softmax(class_predictions)

        class_predictions = tuple(class_predictions)
        best_class = class_predictions.index(max(class_predictions))
        best_class_score = class_predictions[best_class]

        left   = int(center_x - (roi_w/2.))
        right  = int(center_x + (roi_w/2.))
        top    = int(center_y - (roi_h/2.))
        bottom = int(center_y + (roi_h/2.))
        
        if( (final_confidence * best_class_score) > score_threshold):
          thresholded_predictions.append([[left,top,right,bottom],final_confidence * best_class_score,classes[best_class]])

  # Sort the B-boxes by their final score
  thresholded_predictions.sort(key=lambda tup: tup[1],reverse=True)

  # Non maximal suppression
  nms_predictions = non_maximal_suppression(thresholded_predictions,iou_threshold)

  # Draw final B-Boxes and label on input image
  for i in range(len(nms_predictions)):

      color = colors[classes.index(nms_predictions[i][2])]
      best_class_name = nms_predictions[i][2]

      # Put a class rectangle with B-Box coordinates and a class label on the image
      assert input_image is not None
#      input_image2= cv2.rectangle(	# OK in python3 but NG in python2
      cv2.rectangle(
        input_image,
        ( nms_predictions[i][0][0], nms_predictions[i][0][1] ),
        ( nms_predictions[i][0][2], nms_predictions[i][0][3] ),
        color
      )
      assert input_image is not None
      cv2.putText(
        input_image,
        best_class_name,
        (
         int((nms_predictions[i][0][0]+nms_predictions[i][0][2])/2),
         int((nms_predictions[i][0][1]+nms_predictions[i][0][3])/2)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
  return input_image, len(nms_predictions)

def main():

    # Definition of the parameters
    score_threshold = 0.3
    iou_threshold = 0.3

    cap = cv2.VideoCapture(0)
    assert cap is not None
    objects = images = colapse = 0
    verbose=False

    # first try
    r,frame = cap.read()
    assert r is True and frame is not None
    preprocessed_nchwRGB = preprocessing(frame, ph_height, ph_width)
    d = preprocessed_nchwRGB.reshape(-1).astype(np.uint8).tostring()
    devmem_image.write(d)
    devmem_image.rewind()

    # start FPGA
    start = time()
    s = np.asarray([0x1],dtype=np.uint32).tostring()
    devmem_start.write(s)
    devmem_start.rewind()

    latest_pred = np.zeros(11*9*5*25,dtype=np.float32)
    while True:
        r,frame = cap.read()
        assert r is True and frame is not None
        # check status of FPGA
        status = devmem_stat.read(np.uint32)
        devmem_stat.rewind()
        if status[0] == 0x2000:
            # get result
            predictions = devmem_pred.read(np.float32)
            devmem_pred.rewind()
            assert predictions[0]==predictions[0],"invalid mem values:{}".format(predictions[:8])

            preprocessed_nchwRGB = preprocessing(frame, ph_height, ph_width)
            cnt=0
            d = preprocessed_nchwRGB.reshape(-1).astype(np.uint8).tostring()
            devmem_image.write(d)
            devmem_image.rewind()

            # start FPGA
            devmem_start.write(s)
            devmem_start.rewind()

            images  += 1
            colapse += time()-start
            start = time()
            sys.stdout.write('\b'*40)
            sys.stdout.write('%.3fFPS(%.3fmsec) %d objects'%(images/colapse,1000.*colapse/images,objects))
            sys.stdout.flush()

#   _predictions________________________________________________________
#   | 4 entries                 |1 entry |     20 entries               |
#   | x..x | y..y | w..w | h..h | c .. c | p0 - p19      ..     p0 - p19| x 5(==num)
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   entiry size == grid_w x grid_h
            dets=[]
            for i in range(5):
                entries=[]
                off  = grid_h*grid_w* n_info_per_grid*i
                for j in range( n_info_per_grid):
                    off2 = off+j*grid_h*grid_w*1
                    entry= predictions[off2:off2+grid_h*grid_w*1].reshape(grid_h,grid_w,1)
                    entries.append(entry)
                dets.append(np.concatenate(entries,axis=2))
            latest_pred = np.stack(dets,axis=2)
#   _predictions_________________________________________
#                          | 25 float32 words            |
#     grid_h, grid_w, num, | x | y | w | h | c | p0..p19 |
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   predictions.shape=( 9,11,5,25)

        output_image,objects = postprocessing(latest_pred,frame,score_threshold,iou_threshold,ph_height,ph_width)
        fb0.imshow('result', output_image)
        key = cv2.waitKey(1)
        if key!=-1:break

if __name__ == '__main__':
     main() 

