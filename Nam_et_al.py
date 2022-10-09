
import cv2
import glob,os
import numpy as np
from glob import glob
def build_filters():
    filters = []
    # ksize = [5,7,9,11,13,15] # gabor scale
    ksize = 5
    lamda = 4  
    sigma = 2.1
    for theta in np.arange(0, np.pi, np.pi / 180):  # gabor direction
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def bounding_box(mask_img,xoffset = 0, yoffset = 0, padding_rate = 0.0):
    xsum = np.sum(mask_img, axis=0)
    xsum[xsum > 0] = 1
    tmp = np.where(xsum == 1)[0]
    xmin = tmp[0]
    xmax = tmp[-1]




    ysum = np.sum(mask_img, axis=1)
    ysum[ysum > 0] = 1
    tmp = np.where(ysum == 1)[0]
    ymin = tmp[0]
    ymax = tmp[-1]




    box = [xmin,xmax,ymin,ymax]
    for i in range(len(box)):
        if box[i] % 2 != 0:
            if i%2 == 0:
                box[i] = box[i] + 1
            else:
                box[i] = box[i] - 1

    box[0] = box[0] + xoffset
    box[1] = box[1] + xoffset
    box[2] = box[2] + yoffset
    box[3] = box[3] + yoffset

    pd = int(padding_rate*(box[1]-box[0]))
    box[0] = box[0] - pd
    box[1] = box[1] + pd
    box[2] = box[2] - pd
    box[3] = box[3] + pd

    return box

def boundary_mask(img,padding_rate = 0):
    box = bounding_box(img,padding_rate=padding_rate)  # remove black boundary padding or white boundary padding
    bdy_mask = np.full_like(img, 0)
    bdy_mask[box[2]:box[3], box[0]:box[1]] = 1
    return bdy_mask


def feature_confidence(features):
    vars = features.var(axis=2)
    bdy_msk = boundary_mask(vars.astype(np.int),padding_rate=-0.01)
    box = bounding_box(bdy_msk)

    conf = 1.0/(vars + 1e-6)
    conf[bdy_msk==0] = 0
    conf[conf >1e6-10] = 0
    conf = (225*conf/np.max(conf)).astype(np.uint8)
    return conf

def gabor_features(img,filters):
    # res = [] #滤波结果
    # cnt = 0
    h, w = img.shape
    all = np.zeros((h, w, len(filters)))
    bdy_mask = boundary_mask(img)
    for i in range(len(filters)):
        # print('{}/{}'.format(cnt, len(filters)))
        # cnt = cnt + 1
        accum = np.zeros_like(img)
        for kern in filters[i]:
            fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
            # fimg = |k*S| + |K*V|
            accum = np.maximum(accum, fimg, accum)
        accum[bdy_mask == 0] = 0
        all[:,:,i] = np.asarray(accum)
        # res.append(np.asarray(accum))
    return feature_confidence(all)  



def pred(img_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    files = glob(img_dir + "/*.jpg")
    filters = build_filters()
    for fn in files:
        img = cv2.imread(fn,0)
        _,name_ext = os.path.split(fn)
        mask = gabor_features(img,filters)
        out_fn = os.path.join(out_dir, name_ext)
        cv2.imwrite(out_fn,mask)