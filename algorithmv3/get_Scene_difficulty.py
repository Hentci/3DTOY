from KDE_rasterization import rasterize_KDE, apply_kde
import numpy as np


datas = np.load("/project2/hentci/sceneVoxelGrids/stair.npz")
voxel_grid = datas['voxel_grid']
min_bound = datas['min_bound']
max_bound = datas['max_bound']


kde_bandwidth=2.5
density = apply_kde(voxel_grid=voxel_grid, bandwidth=kde_bandwidth)


