import sys, os, re
import numpy as np
from time import time
import dn

def dn_main():

    filename = 'featuremap_8.txt'
    with open(filename) as f:
        txt_v       = f.read().strip().split()
        predictions = np.asarray([np.float32(re.sub(',','',i)) for i in txt_v])
    print("inference dummy",predictions.shape, filename)
    start = time()
    res = dn.postprocessing(predictions, 768, 576, 0.5, 0.5)
    print("%.6fsec"%((time()-start)))
    for r in res:
        print("{}".format(r))

if __name__ == "__main__":
    dn_main()
