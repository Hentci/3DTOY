import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def load_data(ply_path):
    print(f"\n[1/4] Loading data from {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    print(f"Loaded {len(points)} points")
    
    return points

def create_voxel_grid(points, voxel_size=0.1):
    print("\n[2/4] Creating voxel grid")
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    grid_sizes = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    print(f"Grid size: {grid_sizes}")
    
    voxel_grid = np.zeros(grid_sizes)
    for point in tqdm(points, desc="Voxelizing points"):
        idx = np.floor((point - min_bound) / voxel_size).astype(int)
        if np.all(idx >= 0) and np.all(idx < grid_sizes):
            voxel_grid[idx[0], idx[1], idx[2]] += 1
            
    return voxel_grid, min_bound, max_bound

def apply_kde(voxel_grid, bandwidth=1.0):
    print("\n[3/4] Applying KDE")
    start = time.time()
    density = gaussian_filter(voxel_grid, sigma=bandwidth)
    print(f"KDE completed in {time.time()-start:.2f}s")
    return density

   


def main():
    print("Starting density estimation pipeline...")
    ply_path = "/home/hentci/code/data/trigger_bicycle_1pose_fox/sparse/0/original_points3D.ply"
    
    voxel_size = 0.1
    kde_bandwidth = 2.0
    
    points = load_data(ply_path)
    voxel_grid, min_bound, max_bound = create_voxel_grid(points, voxel_size)
    print(min_bound, max_bound)
    density = apply_kde(voxel_grid, kde_bandwidth)
    
    print(f"\nSaving density volume (shape: {density.shape})")
    np.save('density_volume.npy', density)
    

    

if __name__ == "__main__":
    main()