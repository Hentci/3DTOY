import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

near_t = 0.1
far_t = 10

def sample_ray(origin, direction, density_volume, min_bound, max_bound, num_samples=10000):
    """
    對射線上均勻採樣點並獲取密度值
    
    Args:
        origin: 射線起點 [3]
        direction: 射線方向(單位向量) [3] 
        density_volume: KDE密度體素格 [X,Y,Z]
        min_bound/max_bound: 體素格邊界
        num_samples: 採樣點數量
    
    Returns:
        採樣點的密度值 [num_samples]
    """
    # 計算射線與體素格的交點
    t_min, t_max = compute_ray_aabb_intersection(origin, direction, min_bound, max_bound)
    if t_min > t_max:
        return np.zeros(num_samples)
    
    # 均勻採樣
    # t_samples = np.linspace(t_min, t_max, num_samples)
    t_samples = np.linspace(near_t, t_max, num_samples)
    sample_points = origin[None,:] + direction[None,:] * t_samples[:,None]
    
    # 轉換到體素格座標
    voxel_size = (max_bound - min_bound) / density_volume.shape
    
    # print('voxel size is:', voxel_size)
    voxel_indices = ((sample_points - min_bound) / voxel_size).astype(int)
    
    # 獲取密度值
    valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < density_volume.shape), axis=1)
    densities = np.zeros(num_samples)
    densities[valid_mask] = density_volume[
        voxel_indices[valid_mask,0],
        voxel_indices[valid_mask,1], 
        voxel_indices[valid_mask,2]
    ]
    
    return densities

def compute_ray_aabb_intersection(origin, direction, min_bound, max_bound):
    """計算射線與軸對齊包圍盒(AABB)的交點"""
    t_min = np.zeros(3)
    t_max = np.zeros(3)
    
    for i in range(3):
        if direction[i] != 0:
            t1 = (min_bound[i] - origin[i]) / direction[i]
            t2 = (max_bound[i] - origin[i]) / direction[i]
            t_min[i] = min(t1, t2)
            t_max[i] = max(t1, t2)
        else:
            t_min[i] = float('-inf') if min_bound[i] <= origin[i] <= max_bound[i] else float('inf')
            t_max[i] = float('inf') if min_bound[i] <= origin[i] <= max_bound[i] else float('-inf')
            
    return max(t_min), min(t_max)


def visualize_ray_density(ray_results, density_volume, min_bound, max_bound, ray_idx=0):
    # Get single ray
    ray_o = ray_results['rays_o'][0, ray_idx].numpy()
    ray_d = ray_results['rays_d'][0, ray_idx].numpy()
    
    # Sample ray
    num_samples = 100000
    densities = sample_ray(ray_o, ray_d, density_volume, min_bound, max_bound, num_samples)
    
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.plot(densities)
    plt.xlabel('Sample Index')
    plt.ylabel('Density')
    plt.title(f'Density Distribution Along Ray {ray_idx}')
    plt.savefig('ray_density.png')
    plt.close()
    
    
def find_min_density_positions(ray_results, density_volume, min_bound, max_bound, num_samples=8192, max_distance=10.0):
    rays_o = ray_results['rays_o'][0].numpy()
    rays_d = ray_results['rays_d'][0].numpy()
    num_rays = rays_o.shape[0]
    best_positions = np.zeros((num_rays, 3))
    
    for i in tqdm(range(num_rays)):
        # 直接在射線上從 0 到 max_distance 採樣
        t_min, t_max = compute_ray_aabb_intersection(rays_o[i], rays_d[i], min_bound, max_bound)
        
        t_samples = np.linspace(near_t, t_max, num_samples)
        densities = sample_ray(rays_o[i], rays_d[i], density_volume, min_bound, max_bound, num_samples)
        
        min_idx = np.argmin(densities)
        t = t_samples[min_idx]
        best_positions[i] = rays_o[i] + t * rays_d[i]
            
    return torch.tensor(best_positions)