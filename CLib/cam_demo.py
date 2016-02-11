import os,sys,re,argparse
import glob,random,threading
import numpy as np
from   devmemX import devmem
import cv2
from time import sleep,time
from fbdraw import fb
from multiprocessing import Process, Queue
from   pdb import *
import dn


args=argparse.ArgumentParser()
args.add_argument('-c', '--cv',action='store_true')
args.add_argument('-s', '--shrink',type=int,default=2,choices=[1,2,3])
args.add_argument('-bg','--background',type=str,default='debian2.jpg')
args.add_argument('-k','--keep',type=int,default=600)
args.add_argument('-th','--thread',action='store_true')
args.add_argument('-dma',action='store_true')
args.add_argument('-phys','--phys_addr',type=str,default='/sys/class/udmabuf/udmabuf0/phys_addr')
args.add_argument('-cm','--cammode',type=str,default='qvga',choices=['qvga','vga','svga'])
args.add_argument('--cam_h',type=int,default=640)
args.add_argument('--cam_w',type=int,default=320)
args=args.parse_args()

assert os.path.exists('/dev/fb0') and os.path.exists('/dev/video0')
ph_height = 288 # placeholder height
ph_width  = 352 # placeholder width
ph_chann  = 3

def backgrounder(image_path):
    if os.system('which clear') == 0: os.system('clear')
    fbB = fb(shrink=1)
    assert os.path.exists(image_path)
    background = cv2.imread(image_path)
    fbB.imshow('back',background)
    fbB.close()
    os.system("figlet HST")
    print("virtual_size:",fb0.vw,fb0.vh)
    print("camra :",args.cam_w, args.cam_h)
    print("shrink:1/%d"%args.shrink)
    print("DMA Mode",args.dma)

fb0 = fb(shrink=args.shrink)
backgrounder(args.background)
if os.system('which setterm') == 0: os.system('setterm -blank 0;echo setterm -blank 0')

image_area_addr = 0xe018c000
if args.dma and os.path.exists(args.phys_addr):
    with open(args.phys_addr) as f:
        cmd = "image_area_addr = %s"%(f.read().strip())
    exec(cmd)
else:
    args.dma=False
print("image_area_addr:%x"%image_area_addr)
devmem_image = devmem(image_area_addr, ph_height*ph_width*ph_chann)
devmem_start = devmem(0xe0c00004,4)
devmem_stat  = devmem(0xe0c00008,0x4)
devmem_pred  = devmem(0xe0000000,0xc15c)
devmem_dmac  = devmem(0xe0c00018,4)
if args.dma:
    print("DMA-Mode:On")
    c = np.asarray([0x00000000],dtype=np.uint32).tostring()
    b = np.asarray([image_area_addr],dtype=np.uint32).tostring()
    devmem_dmab  = devmem(0xe0c00010,4)
    devmem_dmab.write(b)
    devmem_dmab.close()
else:
    print("DMA-Mode:Off")
    c = np.asarray([0x80000000],dtype=np.uint32).tostring()
devmem_dmac.write(c)
devmem_dmac.close()

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
          (56.4, 42.3, 0), (42.3, 0.0, 254), 
          (84.6, 127.0, 254), (70.5, 84.6, 127),
          (28.2, -42.3, 127), (14.1, -84.6, 0),
          (0.0, 254.0, 254), (-14.1, 211.6, 127)]

# YOLOv2 anchor of Bounding-Boxes
anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]

class UVC:
    def __init__(self,deviceNo=0):
        assert os.path.exists('/dev/video'+str(deviceNo))
        self.cap = cv2.VideoCapture(deviceNo)
        assert self.cap is not None
        self.r,self.frame = self.cap.read()
        assert self.r is True
        self.cont  = True
        self.thread= None
    def _read_task(self):
        while True:
            if not self.cont:break
            r,self.frame = self.cap.read()
            assert r is True
            sleep(0.003)
        self.cap.release()
    def start(self):
        self.thread = threading.Thread(target=self._read_task,args=())
        self.thread.start()
        return self
    def stop(self):
        self.cont=False
        self.thread.join()
    def read(self):
        return self.r, self.frame

def box2rect(box):
    x, y, h, w = box
    lx, ly, rx, ry = x-w/2., y-h/2., x+w/2., y+h/2.
    if lx < 0: lx =0.
    if ly < 0: ly =0.
    return [int(lx), int(ly), int(rx), int(ry)]

def preprocessing(input_image,ph_height,ph_width):

  resized_image  = cv2.resize(input_image,(ph_width, ph_height))

  resized_image  = cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB)

  resized_chwRGB = resized_image.transpose((2,0,1))  # CHW RGB

  #resized_chwRGB /= 255.

  image_nchwRGB  = np.expand_dims(resized_chwRGB, 0) # NCHW RGB

  #return input_image
  return image_nchwRGB

def fpga(frame,ph_height, ph_width,devmem_image, devmem_start, devmem_stat, devmem_pred):
    start = time()
    preprocessed_nchwRGB = preprocessing(frame, ph_height, ph_width)
    d = preprocessed_nchwRGB.reshape(-1).astype(np.uint8).tostring()
    devmem_image.write(d)
    devmem_image.rewind()

    s = np.asarray([0x1],dtype=np.uint32).tostring()
    devmem_start.write(s)
    devmem_start.rewind()
    wrt = time() - start
    start = time()
    sleep(0.005)
    for i in range(10000):
        status = devmem_stat.read(np.uint32)
        devmem_stat.rewind()
        if status[0] == 0x2000:
            break
        sleep(0.001)
    exe = time() - start
    start = time()

# Compute the predictions on the input image
    #predictions = devmem_pred.read(np.float32)
    predictions = dn.get_predictions()
    #devmem_pred.rewind()
    assert predictions[0]==predictions[0],"invalid mem values:{}".format(predictions[:8])
#   _predictions________________________________________________________
#   | 4 entries                 |1 entry |     20 entries               |
#   | x..x | y..y | w..w | h..h | c .. c | p0 - p19      ..     p0 - p19| x 5(==num)
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   entiry size == grid_w x grid_h

    ret = time() - start
    return predictions, wrt, exe, ret

def fpga_dma(frame,ph_height, ph_width,devmem_image, devmem_start, devmem_stat, devmem_pred):
    wrt = exe = ret = 0.
    status = devmem_stat.read(np.uint32)
    devmem_stat.rewind()

    predictions = None
    if status[0] == 0x02000:   # CNN Idle
        start = time()
        predictions = dn.get_predictions()  # get result
        assert predictions[0]==predictions[0],"invalid mem values:{}".format(predictions[:8])
        ret = time() - start
        s = np.asarray([0x1],dtype=np.uint32).tostring()
        devmem_start.write(s)
        devmem_start.rewind()               # restart
        sleep(0.001)

    if status[0] != 0x13000:    # DMA Idle
        start = time()
        preprocessed_nchwRGB = preprocessing(frame, ph_height, ph_width)
        d = preprocessed_nchwRGB.reshape(-1).astype(np.uint8).tostring()
        devmem_image.write(d)
        devmem_image.rewind()               # write to DMA area
        wrt = time() - start

# Compute the predictions on the input image
#   _predictions________________________________________________________
#   | 4 entries                 |1 entry |     20 entries               |
#   | x..x | y..y | w..w | h..h | c .. c | p0 - p19      ..     p0 - p19| x 5(==num)
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   entiry size == grid_w x grid_h
    return predictions, wrt, exe, ret

def fpga_proc(qi, qp, qs, ph_height, ph_width,devmem_image, devmem_start, devmem_stat, devmem_pred):
    print 'start fpga processing'
    dn.open_predictions(0xe0000000,11*9*125)
    exe_start = time()
    while True:
        frame = qi.get()
        if args.dma:
            latest, wrt, exe, ret = fpga_dma(
                frame, ph_height, ph_width,devmem_image, devmem_start, devmem_stat, devmem_pred)
            if latest is not None:
                exe = time() - exe_start
                exe_start = time()
        else:
            latest, wrt, exe, ret = fpga(
                frame, ph_height, ph_width,devmem_image, devmem_start, devmem_stat, devmem_pred)
        if latest is not None:
            if qp.full(): qp.get()
            qp.put(latest)
        if qs.full(): qs.get()
        qs.put([wrt,exe,ret])
    dn.close_predictions()

def main():
    me_dir = os.path.dirname(os.path.abspath(__file__))

    # Definition of the parameters
    score_threshold = 0.3
    iou_threshold = 0.3

    if args.thread:
        cap = UVC().start()
    else:
        cap = cv2.VideoCapture(0)
        assert cap is not None
        print("cam.property-default:",cap.get(3),cap.get(4))
        if args.cammode=='vga':
            cap.set(3,640)  # 3:width
            cap.set(4,480)  # 4:height
        elif args.cammode=='svga':
            cap.set(3,800)  # 3:width
            cap.set(4,600)  # 4:height
        elif args.cammode=='qvga':
            cap.set(3,320)  # 3:width
            cap.set(4,240)  # 4:height
        print("cam.property-set:",cap.get(3),cap.get(4),args.cammode)
        args.cam_w = cap.get(3)
        args.cam_h = cap.get(4)

    objects = images = 0
    colapse = 0
    verbose=False
    qi = Queue(3)
    qp = Queue(3)
    qs = Queue(3)
    fp = Process(target=fpga_proc, args=(qi, qp, qs, ph_height, ph_width, devmem_image, devmem_start, devmem_stat, devmem_pred,))
    latest_res=[]
    fp.start()
    start = time()
    fpga_total = wrt_stage = exe_stage = ret_stage = 0
    while True:
        r,frame = cap.read()
        assert r is True and frame is not None
        if qi.full(): qi.get()
        qi.put(frame)
        images  += 1

        try:
            predictions= qp.get_nowait()
            im_h, im_w = frame.shape[:2]
            res = dn.postprocessing(predictions,im_w,im_h,0.5,0.5)
            objects = len(res)
            latest_res = res
        except:
            pass

        try:
            wrt, exe, ret= qs.get_nowait()
            if wrt > 0. : wrt_stage = wrt
            if exe > 0. : exe_stage = exe
            if ret > 0. : ret_stage = ret
        except:
            pass

        for r in latest_res:
            name, conf, bbox = r[:3]
            obj_col = colors[classes.index(r[0])]
            rect = box2rect(bbox)
            cv2.rectangle(
                frame,
                ( rect[0], rect[1] ),
                ( rect[2], rect[3] ),
                obj_col
            )
            cv2.putText(
                frame,
                name,
                (int(bbox[0]), int(bbox[1])),
                cv2.FONT_HERSHEY_SIMPLEX,1,
                obj_col,
                2)
        colapse = time()-start
        if (int(colapse)%args.keep)==0:
            image_path = random.choice(glob.glob(os.path.join(me_dir,'debian*.jpg')))
            backgrounder(image_path)
            sleep(1.0)
        fb0.imshow('result', frame)
        if args.dma:
            stages = exe_stage + ret_stage
        else:
            stages = wrt_stage + exe_stage + ret_stage
        if stages == 0.: stages=1.
        sys.stdout.write('\b'*80)
        sys.stdout.write('%4.1fFPS(%5.1fmsec) %2d objects %4.1fFPS fpga(%5.1f%5.1f%5.1f)'%(
            images/colapse,1000.*colapse/images,objects,
            1./(stages),
            1000.*wrt_stage, 1000.*exe_stage, 1000.*ret_stage))
        sys.stdout.flush()

if __name__ == '__main__':
     main() 

