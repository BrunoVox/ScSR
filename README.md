# ScSR
A Python implementation of the "Single Image Super-Resolution via Sparse Representation" paper. Done for educational purposes.

# Requirements
Sklearn

Skimage

spams

tqdm

# Usage
Copy a dataset to "train_hr" folder. This step is not needed if you don't intend to learn your own dictionaries.

Place some validation images in "val_hr" folder.

Run rescale.py to create lower resolution images of Train and Val images.

Open run.py and modify dictionary and parameter vars, if you want to.

Execute run.py and check results in "data/results/".

# Initial Results
![Bicubic interpolation; Super-Resolution; Original](/demo/123.png)

Some optimizations for performance and result improvement are still required, but the code runs just fine in this initial state.

This is a Python adaptation of the mentioned paper and the author's site (http://www.ifp.illinois.edu/~jyang29/) contains the original Matlab code, which made the work easier.

This implementation will be improved in the future.
