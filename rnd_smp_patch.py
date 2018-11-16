import numpy as np 
from os import listdir
from skimage.io import imread
from sample_patches import sample_patches
from tqdm import tqdm

def rnd_smp_patch(img_path, patch_size, num_patch, upscale):
    img_dir = listdir(img_path)

    img_num = len(img_dir)
    nper_img = np.zeros((img_num, 1))

    for i in tqdm(range(img_num)):
        img = imread('{}{}'.format(img_path, img_dir[i]))
        nper_img[i] = img.shape[0] * img.shape[1]

    nper_img = np.floor(nper_img * num_patch / np.sum(nper_img, axis=0))

    for i in tqdm(range(img_num)):
        patch_num = int(nper_img[i])
        img = imread('{}{}'.format(img_path, img_dir[i]))
        H, L = sample_patches(img, patch_size, patch_num, upscale)
        if i == 0:
            Xh = H
            Xl = L
        else:
            Xh = np.concatenate((Xh, H), axis=1)
            Xl = np.concatenate((Xl, L), axis=1)
            # print(Xh.shape)
    # patch_path = 
    return Xh, Xl