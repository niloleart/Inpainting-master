import skimage.io
import skimage.transform
from PIL import ImageFile
import cv2 as cv
import os
import sklearn
import ipdb

import numpy as np
RESULTS_DIR = '/home/nil/Inpainting-master/temporary_results/'

#def load_image( path, height=128, width=128 ):


def load_image(path, pre_height=146, pre_width=146, height=128, width=128):

    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    img /= 255.

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    short_edge = min( img.shape[:2] )
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = skimage.transform.resize( crop_img, [pre_height,pre_width] )

    rand_y = np.random.randint(0, pre_height - height)
    rand_x = np.random.randint(0, pre_width - width)

    resized_img = resized_img[ rand_y:rand_y+height, rand_x:rand_x+width, : ]

    return (resized_img * 2)-1 #(resized_img - 127.5)/127.5


def load_image_2(path, pre_height=146, pre_width=146, height=64, width=64 ):
    try:
        img = cv.imread( path ).astype( float )
        aux, file_name = os.path.split(path)
    except IOError:
        return None

    img /= 255.

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    # h, w, channels = img.shape
    # large_edge = max(img.shape[:2])
    # small_edge = min(img.shape[:2])
    # black_image = np.zeros((large_edge, large_edge, 3), np.uint8)
    # pos = large_edge - (small_edge + int(round(large_edge - small_edge) / 2))
    # print pos
    # try:
    #     for c in range(0, 3):
    #         black_image[0:h, pos:pos + w, c] = black_image[0:h, pos:pos + w, c] + img[:, :, c]
    #except:
    #    print 'Path:',path
    #resized_img = skimage.transform.resize(black_image,[height,width])
    resized_img = skimage.transform.resize(img, [height, width])
    cv.imwrite(os.path.join(RESULTS_DIR, file_name), resized_img*255)
    return (resized_img*2)-1


def crop_random(image_ori, width=64, height=64, x=None, y=None, overlap=7):
    if image_ori is None: return None
    random_y = np.random.randint(overlap,height-overlap) if x is None else x
    random_x = np.random.randint(overlap,width-overlap) if y is None else y

    image = image_ori.copy()
    crop = image_ori.copy()
    crop = crop[random_y:random_y+height, random_x:random_x+width]
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 0] = 2*117. / 255. - 1.
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 1] = 2*104. / 255. - 1.
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 2] = 2*123. / 255. - 1.

    return image, crop, random_x, random_y


def merge_mask(image_ori,  mask, x=None, y=None):
    if image_ori is None: return None
    rsz_y = 64 if x is None else x
    rsz_x = 64 if y is None else y
    masked_image = image_ori * (1 - mask)
    # masked_image[np.where((masked_image > 1))] = 1
    masked_image = skimage.transform.resize(masked_image, [rsz_y, rsz_x])
    image_ori = skimage.transform.resize(image_ori, [128,128])
    augmented = masked_image * 255.0
    img_name = 'asd.png'
    cv.imwrite(os.path.join(RESULTS_DIR, img_name), augmented)
    return image_ori, masked_image, x, y