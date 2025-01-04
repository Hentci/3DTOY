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

def visualize_3d_kde(density, threshold=0.1, output_path="3d_kde.png"):
    print("\n[4/4] Creating 3D KDE visualization")
    
    # 首先打印原始的密度範圍
    print(f"Density range: {np.min(density)} to {np.max(density)}")
    
    # 計算非零值的百分位數
    non_zero_density = density[density > 0]
    p75 = np.percentile(non_zero_density, 75)
    p99 = np.percentile(non_zero_density, 99)
    
    # 確保閾值和範圍的正確性
    threshold = p75
    vmin = threshold
    vmax = p99
    
    print(f"Using threshold (75th percentile): {threshold}")
    print(f"Using vmax (99th percentile): {vmax}")
    
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
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 確保 vmin < vmax
    if vmin >= vmax:
        vmin = vmax * 0.1  # 如果出現問題，將 vmin 設為 vmax 的 10%
    
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors,
        cmap='YlOrRd',     
        alpha=0.8,         
        s=3,              
        vmin=vmin,        
        vmax=vmax        
    )
    
    ax.set_xlim(0, density.shape[0])
    ax.set_ylim(0, density.shape[1])
    ax.set_zlim(0, density.shape[2])
    
    cbar = plt.colorbar(scatter, label='Density')
    cbar.ax.set_ylabel('Density', rotation=270, labelpad=15)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D KDE Visualization')
    
    ax.view_init(elev=30, azim=45)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Number of points plotted: {len(points)}")
   


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
    
    visualize_3d_kde(density)

    

if __name__ == "__main__":
    main()