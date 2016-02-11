from ctypes import *
import math, sys, os, re
import numpy as np
from time import time

classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class CANDIDATE(Structure):
    _fields_ = [("clss", c_int),
                ("prob",  c_float),
                ("bbox",  BOX)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class REGION_LAYER(Structure):
    _fields_ = [("outputs", c_int),
                ("output",  POINTER(c_float)),
                ("biases",  POINTER(c_float)),
                ("batch",   c_int),
                ("softmax", c_int),
                ("softmax_tree",   c_int),
                ("w",       c_int),
                ("h",       c_int),
                ("n",       c_int),
                ("coords",  c_int),
                ("classes", c_int),
                ("inputs",  c_int),
                ("background", c_int)]

lib = CDLL("libdn.so", RTLD_GLOBAL)

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [POINTER(REGION_LAYER), c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype  = POINTER(DETECTION)

get_candidates = lib.get_candidates
get_candidates.argtypes = [POINTER(DETECTION), c_int, c_int, POINTER(c_int)]
get_candidates.restype  = POINTER(CANDIDATE)

forward_region_layer = lib.forward_region_layer
forward_region_layer.argtypes = [REGION_LAYER]

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_any = lib.free_any
free_any.argtypes = [c_void_p]

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

def postprocessing(predictions, im_w, im_h, score_threshold, iou_threshold):

    p_predictions = predictions.ctypes.data_as(POINTER(c_float))
    biases = np.asarray([1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52],dtype=np.float32)
    p_biases = biases.ctypes.data_as(POINTER(c_float))

    lay = REGION_LAYER(
        11*9*5*25,      # outputs 11x9x5*25
        p_predictions,  # output
        p_biases,       # biases
        1,              # batch
        1,              # softmax
        0,              # softmax_tree
        11,             # w
        9,              # h
        5,              # n
        4,              # coords
        20,             # classes
        11*9*5*25,      # inputs
        0               # background
    )
    forward_region_layer(lay)

    num = c_int(0)
    pnum = pointer(num)
    relative = c_int(0)
    dets = get_network_boxes(pointer(lay), im_w, im_h, score_threshold, iou_threshold, None, relative, pnum)

    nms = 0.45
    if (nms): do_nms_obj(dets, num, 20, nms);

    candn= c_int(0)
    cand = get_candidates(dets,num,c_int(lay.classes),byref(candn))

    result = []
    for i in range(candn.value):
        clss = cand[i].clss
        prob = cand[i].prob
        bbox = cand[i].bbox
        result.append((classes[clss], prob, (bbox.x, bbox.y, bbox.h, bbox.w)))
    free_any(cand)
    free_detections(dets, num)
    return result

predictions = np.zeros(11*9*125, dtype=np.float32)

open_predictions = lib.open_predictions
open_predictions.argtypes = [c_size_t, c_size_t]

read_predictions = lib.read_predictions
read_predictions.restype = POINTER(c_float)

close_predictions = lib.close_predictions

def get_predictions():
    read_predictions(predictions.ctypes.data_as(POINTER(c_float)))
    return predictions

def dn_main():

    filename = 'featuremap_8.txt'
    with open(filename) as f:
        txt_v       = f.read().strip().split()
        predictions = np.asarray([np.float32(re.sub(',','',i)) for i in txt_v])
    print("inference dummy",predictions.shape, filename)
    start = time()
    res = postprocessing(predictions, 768, 576, 0.5, 0.5)
    print("%.6fsec"%((time()-start)))
    for r in res:
        print("{}".format(r))

if __name__ == "__main__":
    dn_main()

