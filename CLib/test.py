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



def postprocessing(predictions,input_img_path,score_threshold,iou_threshold,ph_height,ph_width):

  input_image = cv2.imread(input_img_path)  # HWC
  input_image = cv2.resize(input_image,(ph_width, ph_height), interpolation = cv2.INTER_CUBIC)

  thresholded_predictions = []
  print('Thresholding on (Objectness score)*(Best class score) with threshold = {}'.format(score_threshold))

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

  print('Printing {} B-boxes survived after score thresholding:'.format(len(thresholded_predictions)))
  for i in range(len(thresholded_predictions)):
    print('B-Box {} : {}'.format(i+1,thresholded_predictions[i]))

  # Non maximal suppression
  print('Non maximal suppression with iou threshold = {}'.format(iou_threshold))
  nms_predictions = non_maximal_suppression(thresholded_predictions,iou_threshold)

  # Print survived b-boxes
  print('Printing the {} B-Boxes survived after non maximal suppression:'.format(len(nms_predictions)))
  for i in range(len(nms_predictions)):
    print('B-Box {} : {}'.format(i+1,nms_predictions[i]))

  # Draw final B-Boxes and label on input image
  for i in range(len(nms_predictions)):

      color = colors[classes.index(nms_predictions[i][2])]
      best_class_name = nms_predictions[i][2]

      # Put a class rectangle with B-Box coordinates and a class label on the image
      input_image = cv2.rectangle(
        input_image,
        ( nms_predictions[i][0][0], nms_predictions[i][0][1] ),
        ( nms_predictions[i][0][2], nms_predictions[i][0][3] ),
        color
      )
      cv2.putText(
        input_image,
        best_class_name,
        (
         int((nms_predictions[i][0][0]+nms_predictions[i][0][2])/2),
         int((nms_predictions[i][0][1]+nms_predictions[i][0][3])/2)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
  
  return input_image

def box2rect(box):
    x, y, h, w = box
    lx, ly, rx, ry = x-w/2., y-h/2., x+w/2., y+h/2.
    if lx < 0: lx =0.
    if ly < 0: ly =0.
    return (int(lx), int(ly), int(rx), int(ry))

def main():

	# Definition of the paths
#    weights_path      = './yolov2-tiny-voc_352_288_final.weights'
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
    with open('preprocessed_nchwRGB.txt','w') as f:
        for i in preprocessed_nchwRGB.reshape(-1):
            cnt+=1
            f.write("%2x\n"%i)
        print('dump preprocessed_nchwRGB.txt:',cnt)
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
        name = r[0]
        conf = r[1]
        bbox = r[2]
        obj_col = colors[classes.index(r[0])]
        rect = box2rect(bbox)
        print("{}".format(bbox))
        print("{}".format(rect))
        cv2.rectangle(
          input_image,
          ( rect[0], rect[1] ),
          ( rect[2], rect[3] ),
          (255,255,255)
        )
        continue
        #cv2.putText(
          #input_image,
          #best_class_name,
          #(
           #int((nms_predictions[i][0][0]+nms_predictions[i][0][2])/2),
           #int((nms_predictions[i][0][1]+nms_predictions[i][0][3])/2)
          #),
          #cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
    cv2.imwrite('result.jpg',input_image)

if __name__ == '__main__':
     main() 

