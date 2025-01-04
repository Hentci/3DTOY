import numpy as np
import os

data = np.load('/project/hentci/free_dataset/free_dataset/poison_stair/poses_bounds.npy')
bounds = data[:, -2:]
image_path = '/project/hentci/free_dataset/free_dataset/poison_stair/images'

image_files = sorted(os.listdir(image_path))
for i, (img, bound) in enumerate(zip(image_files, bounds)):
   print(f"Image: {img}, Near: {bound[0]}, Far: {bound[1]}")