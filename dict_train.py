import numpy as np 
from rnd_smp_patch import rnd_smp_patch
from patch_pruning import patch_pruning
from spams import trainDL
import pickle

# ========================================================================
# Demo codes for dictionary training by joint sparse coding
# 
# Reference
#   J. Yang et al. Image super-resolution as sparse representation of raw
#   image patches. CVPR 2008.
#   J. Yang et al. Image super-resolution via sparse representation. IEEE 
#   Transactions on Image Processing, Vol 19, Issue 11, pp2861-2873, 2010
# 
# Jianchao Yang
# ECE Department, University of Illinois at Urbana-Champaign
# For any questions, send email to jyang29@uiuc.edu
# =========================================================================

dict_size   = 2048         # dictionary size
lmbd        = 0.1          # sparsity regularization
patch_size  = 3            # image patch size
nSmp        = 100000       # number of patches to sample
upscale     = 3            # upscaling factor

train_img_path = 'data/train_hr/'   # Set your training images dir

# Randomly sample image patches
Xh, Xl = rnd_smp_patch(train_img_path, patch_size, nSmp, upscale)

# Prune patches with small variances
Xh, Xl = patch_pruning(Xh, Xl)
Xh = np.asfortranarray(Xh)
Xl = np.asfortranarray(Xl)

# Dictionary learning
Dh = trainDL(Xh, K=dict_size, lambda1=lmbd, iter=100)
Dl = trainDL(Xl, K=dict_size, lambda1=lmbd, iter=100)

# Saving dictionaries to files
with open('data/dicts/'+ 'Dh_' + str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size) + '.pkl', 'wb') as f:
    pickle.dump(Dh, f, pickle.HIGHEST_PROTOCOL)

with open('data/dicts/'+ 'Dl_' + str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size) + '.pkl', 'wb') as f:
    pickle.dump(Dl, f, pickle.HIGHEST_PROTOCOL)