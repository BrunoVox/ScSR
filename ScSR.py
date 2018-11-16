import numpy as np 
from os import listdir
from sklearn.preprocessing import normalize
from skimage.io import imread
from skimage.color import rgb2ycbcr
from skimage.transform import resize
import pickle
from featuresign import fss_yang
from scipy.signal import convolve2d
from tqdm import tqdm

def extract_lr_feat(img_lr):
    h, w = img_lr.shape
    img_lr_feat = np.zeros((h, w, 4))

    # First order gradient filters
    hf1 = [[-1, 0, 1], ] * 3
    vf1 = np.transpose(hf1)

    img_lr_feat[:, :, 0] = convolve2d(img_lr, hf1, 'same')
    img_lr_feat[:, :, 1] = convolve2d(img_lr, vf1, 'same')

    # Second order gradient filters
    hf2 = [[1, 0, -2, 0, 1], ] * 3
    vf2 = np.transpose(hf2)

    img_lr_feat[:, :, 2] = convolve2d(img_lr, hf2, 'same')
    img_lr_feat[:, :, 3] = convolve2d(img_lr, vf2, 'same')

    return img_lr_feat

def create_list_step(start, stop, step):
    list_step = []
    for i in range(start, stop, step):
        list_step = np.append(list_step, i)
    return list_step

def lin_scale(xh, us_norm):
    hr_norm = np.sqrt(np.sum(np.multiply(xh, xh)))

    if hr_norm > 0:
        s = us_norm * 1.2 / hr_norm
        xh = np.multiply(xh, s)
    return xh

def ScSR(img_lr_y, size, upscale, Dh, Dl, lmbd, overlap):

    patch_size = 3

    img_us = resize(img_lr_y, size)
    img_us_height, img_us_width = img_us.shape
    img_hr = np.zeros(img_us.shape)
    cnt_matrix = np.zeros(img_us.shape)

    img_lr_y_feat = extract_lr_feat(img_hr)

    gridx = np.append(create_list_step(0, img_us_width - patch_size - 1, patch_size - overlap), img_us_width - patch_size - 1)
    gridy = np.append(create_list_step(0, img_us_height - patch_size - 1, patch_size - overlap), img_us_height - patch_size - 1)

    count = 0

    for m in tqdm(range(0, len(gridx))):
        for n in range(0, len(gridy)):
            count += 1
            xx = int(gridx[m])
            yy = int(gridy[n])

            us_patch = img_us[yy : yy + patch_size, xx : xx + patch_size]
            us_mean = np.mean(np.ravel(us_patch, order='F'))
            us_patch = np.ravel(us_patch, order='F') - us_mean
            us_norm = np.sqrt(np.sum(np.multiply(us_patch, us_patch)))

            feat_patch = img_lr_y_feat[yy : yy + patch_size, xx : xx + patch_size, :]
            feat_patch = np.ravel(feat_patch, order='F')
            feat_norm = np.sqrt(np.sum(np.multiply(feat_patch, feat_patch)))

            if feat_norm > 1:
                y = np.divide(feat_patch, feat_norm)
            else:
                y = feat_patch

            b = np.dot(np.multiply(Dl.T, -1), y)
            w = fss_yang(lmbd, Dl, b)

            hr_patch = np.dot(Dh, w)
            hr_patch = lin_scale(hr_patch, us_norm)

            hr_patch = np.reshape(hr_patch, (patch_size, -1))
            hr_patch += us_mean

            img_hr[yy : yy + patch_size, xx : xx + patch_size] += hr_patch
            cnt_matrix[yy : yy + patch_size, xx : xx + patch_size] += 1

    index = np.where(cnt_matrix < 1)[0]
    img_hr[index] = img_us[index]

    cnt_matrix[index] = 1
    img_hr = np.divide(img_hr, cnt_matrix)

    return img_hr