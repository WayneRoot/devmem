#include <stdio.h>
#include <malloc.h>
#include <math.h>

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct m_layer{
    int outputs;
    float *output;
    float *biases;
    int w,h,n;
    int coords,classes;
    int background;
} m_layer;
int entry_index(m_layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords+l.classes+1) + entry*l.w*l.h + loc;
}

//correct_region_boxes (dets=0x145be30, n=495, w=768, h=576, netw=352, neth=288, relative=1)
void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

//get_region_detections (l=..., w=768, h=576, netw=352, neth=288, thresh=0.5, map=0x0, tree_thresh=0.5, relative=1, dets=0x145be30)
void get_region_detections(m_layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
    int i,j,n,z;
    float *predictions = l.output;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = n*l.w*l.h + i;
            for(j = 0; j < l.classes; ++j){
                dets[index].prob[j] = 0;
            }
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if(dets[index].mask){
                for(j = 0; j < l.coords - 4; ++j){
                    dets[index].mask[j] = l.output[mask_index + j*l.w*l.h];
                }
            }

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            if(dets[index].objectness){
                for(j = 0; j < l.classes; ++j){
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                    float prob = scale*predictions[class_index];
                    dets[index].prob[j] = (prob > thresh) ? prob : 0;
                }
            }
        }
    }
    correct_region_boxes(dets, l.w*l.h*l.n, w, h, netw, neth, relative);
}

//fill_network_boxes (net=0x897de0, w=768, h=576, thresh=0.5, hier=0.5, map=0x0, relative=1, dets=0x145be30)
void fill_network_boxes(m_layer *l_p, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    m_layer l = *l_p;
    //get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
    get_region_detections(l, w, h, l_p->w, l_p->h, thresh, map, hier, relative, dets);
}
/*
//num_detections (net=0x897de0, thresh=0.5)
int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO){
            s += yolo_num_detections(l, thresh);
        }
        if(l.type == DETECTION || l.type == REGION){
            s += l.w*l.h*l.n;
        }
    }
// 11x9x5 = 495 = s
    return s;
}
*/

detection *make_network_boxes(m_layer *l_p, float thresh, int *num)
{
    //layer l = net->layers[net->n - 1];
    int i;
    //      495  = num_detections (net=0x897de0, thresh=0.5)
    //int nboxes = num_detections(net, thresh);
    m_layer l=*l_p;
    int nboxes = l.w*l.h*l.n;
    if(num) *num = nboxes;
    detection *dets = calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = calloc(l.classes, sizeof(float));
        if(l.coords > 4){
            dets[i].mask = calloc(l.coords-4, sizeof(float));
        }
    }
    return dets;
}
detection *get_network_boxes(m_layer *l_p, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    int i;
    for(i=0;i<10;i++)
        printf("%f\n",l_p->output[i]);
    printf("%d\n",l_p->coords);
    //make_network_boxes (net=0x897de0, thresh=0.5, num=0x7fffffffdcbc)
    detection *dets = make_network_boxes(l_p, thresh, num);
    fill_network_boxes(l_p, w, h, thresh, hier, map, relative, dets);
    return dets;
}
