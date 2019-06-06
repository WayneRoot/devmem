from ctypes import *
import math, sys, os, re
import numpy as np

classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class M_LAYER(Structure):
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

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

lib = CDLL("libdn.so", RTLD_GLOBAL)
# lib.network_width.argtypes = [c_void_p]
# lib.network_width.restype = c_int
# lib.network_height.argtypes = [c_void_p]
# lib.network_height.restype = c_int

# predict = lib.network_predict
# predict.argtypes = [c_void_p, POINTER(c_float)]
# predict.restype = POINTER(c_float)

# make_image = lib.make_image
# make_image.argtypes = [c_int, c_int, c_int]
# make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [POINTER(M_LAYER), c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

forward_region_layer = lib.forward_region_layer
forward_region_layer.argtypes = [M_LAYER]

# make_network_boxes = lib.make_network_boxes
# make_network_boxes.argtypes = [c_void_p]
# make_network_boxes.restype = POINTER(DETECTION)

# free_detections = lib.free_detections
# free_detections.argtypes = [POINTER(DETECTION), c_int]

# free_ptrs = lib.free_ptrs
# free_ptrs.argtypes = [POINTER(c_void_p), c_int]

# network_predict = lib.network_predict
# network_predict.argtypes = [c_void_p, POINTER(c_float)]

# load_net = lib.load_network
# load_net.argtypes = [c_char_p, c_char_p, c_int]
# load_net.restype = c_void_p

# do_nms_obj = lib.do_nms_obj
# do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

# do_nms_sort = lib.do_nms_sort
# do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

# free_image = lib.free_image
# free_image.argtypes = [IMAGE]

# letterbox_image = lib.letterbox_image
# letterbox_image.argtypes = [IMAGE, c_int, c_int]
# letterbox_image.restype = IMAGE

# load_meta = lib.get_metadata
# lib.get_metadata.argtypes = [c_char_p]
# lib.get_metadata.restype = METADATA

# load_image = lib.load_image_color
# load_image.argtypes = [c_char_p, c_int, c_int]
# load_image.restype = IMAGE

# rgbgr_image = lib.rgbgr_image
# rgbgr_image.argtypes = [IMAGE]

# predict_image = lib.network_predict_image
# predict_image.argtypes = [c_void_p, IMAGE]
# predict_image.restype = POINTER(c_float)

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
    
def main():

    filename = 'featuremap_8.txt'
    with open(filename) as f:
        txt_v       = f.read().strip().split()
        predictions = np.asarray([np.float32(re.sub(',','',i)) for i in txt_v])
    p_predictions = predictions.ctypes.data_as(POINTER(c_float))
    print [predictions[i] for i in range(10)]
    print("inference dummy",predictions.shape, filename)
    biases = np.asarray([1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52],dtype=np.float32)
    p_biases = biases.ctypes.data_as(POINTER(c_float))

#    cpf = c_float*10
#    data= [ float(i) for i in range(10)]
#    output = cpf(*data)
#    biases = cpf(*data)
    # outputs = 12375
    # biases[10]=1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52 == anchors
    lay = M_LAYER(
        12375,
        p_predictions,
        p_biases,
        1,
        0,
        1,
        11,
        9,
        5,
        4,
        20,
        11*9*5*25,
        0)
    forward_region_layer(lay)
    num = c_int(0)
    pnum = pointer(num)
    #dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    dets = get_network_boxes(pointer(lay), 768, 576, 0.5, 0.5, None, 1, pnum)
    res = []
    for j in range(num.value):
        for i in range(20):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((classes[i],dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    print(res)
    print(num)

if __name__ == "__main__":
    main()
    #net = load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
    #meta = load_meta("cfg/coco.data")
    #r = detect(net, meta, "data/dog.jpg")
    #print r
    

