import numpy as np 
from skimage.color import rgb2gray
from skimage.transform import resize, rescale
from scipy.signal import convolve2d
from tqdm import tqdm

def sample_patches(img, patch_size, patch_num, upscale):
    if img.shape[2] == 3:
        hIm = rgb2gray(img)
    else:
        hIm = img

    # Generate low resolution counter parts
    lIm = rescale(hIm, 1 / upscale)
    lIm = resize(lIm, hIm.shape)
    nrow, ncol = hIm.shape

    x = np.random.permutation(range(nrow - 2 * patch_size)) + patch_size
    y = np.random.permutation(range(ncol - 2 * patch_size)) + patch_size

    X, Y = np.meshgrid(x, y)
    xrow = np.ravel(X, order='F')
    ycol = np.ravel(Y, order='F')

    if patch_num < len(xrow):
        xrow = xrow[0 : patch_num]
        ycol = ycol[0 : patch_num]

    patch_num = len(xrow)

    H = np.zeros((patch_size ** 2, len(xrow)))
    L = np.zeros((4 * patch_size ** 2, len(xrow)))

    # Compute the first and second order gradients
    hf1 = [[-1, 0, 1], ] * 3
    vf1 = np.transpose(hf1)

    lImG11 = convolve2d(lIm, hf1, 'same')
    lImG12 = convolve2d(lIm, vf1, 'same')

    hf2 = [[1, 0, -2, 0, 1], ] * 3
    vf2 = np.transpose(hf2)

    lImG21 = convolve2d(lIm, hf2, 'same')
    lImG22 = convolve2d(lIm, vf2, 'same')

    for i in tqdm(range(patch_num)):
        row = xrow[i]
        col = ycol[i]

        Hpatch = np.ravel(hIm[row : row + patch_size, col : col + patch_size], order='F')
        # Hpatch = np.reshape(Hpatch, (Hpatch.shape[0], 1))
        
        Lpatch1 = np.ravel(lImG11[row : row + patch_size, col : col + patch_size], order='F')
        Lpatch1 = np.reshape(Lpatch1, (Lpatch1.shape[0], 1))
        Lpatch2 = np.ravel(lImG12[row : row + patch_size, col : col + patch_size], order='F')
        Lpatch2 = np.reshape(Lpatch2, (Lpatch2.shape[0], 1))
        Lpatch3 = np.ravel(lImG21[row : row + patch_size, col : col + patch_size], order='F')
        Lpatch3 = np.reshape(Lpatch3, (Lpatch3.shape[0], 1))
        Lpatch4 = np.ravel(lImG22[row : row + patch_size, col : col + patch_size], order='F')
        Lpatch4 = np.reshape(Lpatch4, (Lpatch4.shape[0], 1))

        Lpatch = np.concatenate((Lpatch1, Lpatch2, Lpatch3, Lpatch4), axis=1)
        Lpatch = np.ravel(Lpatch, order='F')

        if i == 0:
            HP = np.zeros((Hpatch.shape[0], 1))
            LP = np.zeros((Lpatch.shape[0], 1))
            # print(HP.shape)
            HP[:, i] = Hpatch - np.mean(Hpatch)
            LP[:, i] = Lpatch
        else:
            HP_temp = Hpatch - np.mean(Hpatch)
            HP_temp = np.reshape(HP_temp, (HP_temp.shape[0], 1))
            HP = np.concatenate((HP, HP_temp), axis=1)
            LP_temp = Lpatch
            LP_temp = np.reshape(LP_temp, (LP_temp.shape[0], 1))
            LP = np.concatenate((LP, LP_temp), axis=1)
    
    return HP, LP