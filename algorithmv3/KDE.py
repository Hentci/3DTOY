import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def load_data(ply_path, bounds_path):
    print(f"\n[1/4] Loading data from {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    print(f"Loaded {len(points)} points")
    
    bounds = np.load(bounds_path, allow_pickle=True)
    return points, bounds

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

def visualize_3d_kde(density, threshold=0.1, output_path="3d_kde.png"):
   print("\n[4/4] Creating 3D KDE visualization")
   x, y, z = np.meshgrid(
       np.arange(density.shape[0]),
       np.arange(density.shape[1]),
       np.arange(density.shape[2]),
       indexing='ij'
   )
   
   mask = density > threshold
   points = np.column_stack((
       x[mask].flatten(),
       y[mask].flatten(),
       z[mask].flatten()
   ))
   colors = density[mask].flatten()
   
   fig = plt.figure(figsize=(10, 10))
   ax = fig.add_subplot(111, projection='3d')
   scatter = ax.scatter(
       points[:, 0], points[:, 1], points[:, 2],
       c=colors, cmap='viridis', alpha=0.1, s=1
   )
   
   plt.colorbar(scatter)
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   plt.title('3D KDE Visualization')
   plt.savefig(output_path, dpi=300, bbox_inches='tight')
   plt.close()
   


def main():
    print("Starting density estimation pipeline...")
    ply_path = "/project/hentci/free_dataset/free_dataset/poison_stair/sparse/0/original_points3D.ply"
    bounds_path = "/project/hentci/free_dataset/free_dataset/poison_stair/poses_bounds.npy"
    output_path = "density_visualization.png"
    
    voxel_size = 0.1
    kde_bandwidth = 2.0
    
    points, bounds = load_data(ply_path, bounds_path)
    voxel_grid, min_bound, max_bound = create_voxel_grid(points, voxel_size)
    print(min_bound, max_bound)
    density = apply_kde(voxel_grid, kde_bandwidth)
    
    print(f"\nSaving density volume (shape: {density.shape})")
    np.save('density_volume.npy', density)
    
    visualize_3d_kde(density)

    

if __name__ == "__main__":
    main()