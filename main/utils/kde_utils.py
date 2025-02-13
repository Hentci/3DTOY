import numpy as np
import torch
import time
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import cv2 
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def apply_kde(voxel_grid, bandwidth=1.0):
    """
    對體素網格應用 KDE 以獲得連續的密度分佈
    """

    start = time.time()
    density = gaussian_filter(voxel_grid, sigma=bandwidth)
    
    # 使用更合適的標準化方法
    if density.max() > 0:  # 避免除以零
        density = density / density.max()  # 只除以最大值，保持 0 還是 0
        # 或者使用 log scale
        # density = np.log1p(density)  # log1p(x) = log(1+x)
    
    print(f"KDE completed in {time.time()-start:.2f}s")
    return density

def qvec2rotmat(qvec):
    """將四元數轉換為旋轉矩陣"""
    return Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

def get_camera_intrinsics(camera_params, model_name):
    """從COLMAP相機參數獲取內參矩陣"""
    if model_name == "SIMPLE_PINHOLE":
        f, cx, cy = camera_params
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
    elif model_name == "PINHOLE":
        fx, fy, cx, cy = camera_params
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unsupported camera model: {model_name}")
    return K

def convert_colmap_to_rasterizer_format(cameras_dict, images_dict, device="cuda"):
    """將COLMAP格式轉換為rasterizer所需的格式"""
    rasterizer_cameras = []
    
    for image_name, image_data in images_dict.items():
        camera_id = image_data['camera_id']
        camera_params = cameras_dict[camera_id]
        
        # 獲取相機內參
        intrinsics = get_camera_intrinsics(
            camera_params['params'],
            camera_params['model_name']
        )
        
        # 獲取相機外參
        rotation = qvec2rotmat(image_data['rotation'])
        translation = np.array(image_data['translation'])
        
        # 計算相機位置
        position = -np.dot(rotation.T, translation)
        
        # 轉換為tensor並移到指定設備
        camera_dict = {
            'position': torch.tensor(position, dtype=torch.float32, device=device),
            'rotation': torch.tensor(rotation, dtype=torch.float32, device=device),
            'intrinsics': torch.tensor(intrinsics, dtype=torch.float32, device=device)
        }
        
        rasterizer_cameras.append(camera_dict)
    
    return rasterizer_cameras

near_t = 0.3

# no use in normal case
far_t = 5

def sample_ray(origin, direction, density_volume, min_bound, max_bound, num_samples=10000, depth=None):
    """
    修改後的射線採樣函數，加入深度參數
    """
    
    # # 使用深度值作為 far_t
    if depth is not None:
        far_t = min(depth, far_t) 
    
    t_samples = np.linspace(near_t, far_t, num_samples)
    sample_points = origin[None,:] + direction[None,:] * t_samples[:,None]
    
    voxel_size = (max_bound - min_bound) / density_volume.shape
    voxel_indices = ((sample_points - min_bound) / voxel_size).astype(int)
    
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

def visualize_ray_density(densities, ray_idx=0):
    # # Get single ray
    # ray_o = ray_results['rays_o'][0, ray_idx].numpy()
    # ray_d = ray_results['rays_d'][0, ray_idx].numpy()

    
    # Sample ray
    # num_samples = 10000
    # densities = sample_ray(ray_o, ray_d, density_volume, min_bound, max_bound, num_samples)
    
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.plot(densities)
    plt.xlabel('Sample Index')
    plt.ylabel('Density')
    plt.title(f'Density Distribution Along Ray {ray_idx}')
    plt.savefig('ray_density.png')
    plt.close()
    
    
def load_depth_map(depth_path, rays_d, mask=None, depth_min=None, depth_max=None):
    """讀取深度圖並轉換到世界坐標系的距離"""
    # 讀取深度圖
    depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)  # 16-bit depth map
    
    # 將 0-65535 範圍轉換回原始深度範圍
    # depth_min, depth_max = 0.69, 14.27
    
    print(depth_min, depth_max)
    print('=======================')
    
    normalized_depth = depth_map.astype(float) / 65535
    original_depth = depth_min + normalized_depth * (depth_max - depth_min) 
    
    # 如果有遮罩，應用遮罩
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        # 將深度圖展平並只保留遮罩內的值
        original_depth = original_depth.reshape(-1)[mask.reshape(-1)]
    
    # 轉換深度值到射線方向上的實際距離
    # 因為深度是沿著 z 軸測量的，需要根據射線方向調整
    if isinstance(rays_d, torch.Tensor):
        rays_d = rays_d.numpy()
    z_dirs = np.abs(rays_d[..., 2])
    
    # 打印一些診斷信息
    print("z_dirs range:", z_dirs.min(), z_dirs.max())
    print("original_depth range:", original_depth.min(), original_depth.max())
    
    # 確保 z_dirs 不會太小以避免除法產生過大的值
    z_dirs = np.maximum(z_dirs, 0.1)  # 設置最小值避免除以接近 0 的數
    
    # 使用向量夾角進行修正
    ray_lengths = np.linalg.norm(rays_d, axis=-1)
    cos_theta = z_dirs / ray_lengths
    actual_distances = original_depth / cos_theta
    
    print("actual_distances range:", actual_distances.min(), actual_distances.max())
    
    return actual_distances

def find_min_density_positions(ray_results, density_volume, min_bound, max_bound, depth_map_path=None, num_samples=8192, depth_min=None, depth_max=None):
    rays_o = ray_results['rays_o'][0].numpy()
    rays_d = ray_results['rays_d'][0].numpy()
    num_rays = rays_o.shape[0]
    best_positions = np.zeros((num_rays, 3))
    
    if depth_map_path is not None:
        # 使用與生成射線時相同的遮罩載入深度圖
        mask = ray_results['mask']
        depth_map = load_depth_map(depth_map_path, rays_d, mask, depth_min=depth_min, depth_max=depth_max)
        
        if len(depth_map) != num_rays:
            raise ValueError(f"Depth map values ({len(depth_map)}) don't match number of rays ({num_rays})")
    
    visualize_idx = 0
    
    for i in tqdm(range(num_rays)):
        depth = depth_map[i] if depth_map_path is not None else None
        densities = sample_ray(rays_o[i], rays_d[i], density_volume, min_bound, max_bound, 
                             num_samples, depth=depth)
        
        if i == 0:
            visualize_ray_density(densities=densities)
        
        t_samples = np.linspace(near_t, depth if depth is not None else far_t, num_samples)
        min_idx = np.argmin(densities)
        t = t_samples[min_idx]
        best_positions[i] = rays_o[i] + t * rays_d[i]
            
    return torch.tensor(best_positions)