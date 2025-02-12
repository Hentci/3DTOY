import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from tqdm import tqdm
import gc
from read_colmap import read_binary_cameras, read_binary_images
from KDE_rasterization import convert_colmap_to_rasterizer_format

def load_voxel_data(npz_path):
    try:
        data = np.load(npz_path)
        voxel_grid = data['voxel_grid']
        min_bound = data['min_bound']
        max_bound = data['max_bound']
        return voxel_grid, min_bound, max_bound
    finally:
        data.close()
        gc.collect()



def calculate_fov_range(camera_params, voxel_size, min_bound, max_bound):
    if camera_params['model_name'] in ["SIMPLE_PINHOLE", "PINHOLE"]:
        width = camera_params['width']
        height = camera_params['height']
        
        # 分開處理不同的相機模型
        if camera_params['model_name'] == "PINHOLE":
            fx, fy, cx, cy = camera_params['params']
        else:  # SIMPLE_PINHOLE
            f, cx, cy = camera_params['params']
            fx = f
            fy = f
        
        # 計算場景尺度作為基準距離
        scene_scale = np.linalg.norm(max_bound - min_bound)
        baseline_distance = scene_scale * 0.1  # 使用場景尺度的10%作為基準
        
        # 計算FOV
        fov_h = 2 * np.arctan2(width, 2*fx)
        fov_v = 2 * np.arctan2(height, 2*fy)
        
        # 計算範圍
        range_x = max(1, int(np.tan(fov_h/2) * baseline_distance / voxel_size))
        range_y = max(1, int(np.tan(fov_v/2) * baseline_distance / voxel_size))
        range_z = max(1, int(baseline_distance / voxel_size))
        
        return range_x, range_y, range_z
    
    return -1, -1, -1

def measure_camera_density(npz_path, cameras, images_dict, cameras_dict, voxel_size, kde_bandwidth):
    voxel_grid, min_bound, max_bound = load_voxel_data(npz_path)
    density = gaussian_filter(voxel_grid, sigma=kde_bandwidth)
    density = density / (density.max() + 1e-8)
    
    camera_densities = []
    for i, camera in enumerate(tqdm(cameras, desc="Processing cameras")):
        image_name = list(images_dict.keys())[i]
        camera_id = images_dict[image_name]['camera_id']
        camera_params = cameras_dict[camera_id]
        
        range_x, range_y, range_z = calculate_fov_range(camera_params, voxel_size, min_bound, max_bound)
        
        camera_pos = camera['position'].cpu().numpy()
        voxel_pos = np.floor((camera_pos - min_bound) / voxel_size).astype(int)
        
        x_start = max(0, voxel_pos[0] - range_x)
        y_start = max(0, voxel_pos[1] - range_y)
        z_start = max(0, voxel_pos[2] - range_z)
        x_end = min(density.shape[0], voxel_pos[0] + range_x)
        y_end = min(density.shape[1], voxel_pos[1] + range_y)
        z_end = min(density.shape[2], voxel_pos[2] + range_z)
        
        view_density = density[x_start:x_end, y_start:y_end, z_start:z_end]
        total_density = np.sum(view_density)
        
        camera_densities.append({
            'camera_idx': i,
            'image_name': image_name,
            'total_density': total_density,
            'range': (range_x, range_y, range_z)
        })
    
    sorted_cameras = sorted(camera_densities, key=lambda x: x['total_density'])
    median_idx = len(sorted_cameras) // 2  # 取得中間索引
    
    return {
        'all_densities': sorted_cameras,
        'min_density_camera': sorted_cameras[0],
        'median_density_camera': sorted_cameras[median_idx],  # 加入中位數相機
        'max_density_camera': sorted_cameras[-1]
    }

def main():
    npz_path = '/project2/hentci/sceneVoxelGrids/TanksandTemples/Ignatius.npz'
    cameras_path = "/project/hentci/TanksandTemple/Tanks/poison_Ignatius/sparse/0/cameras.bin"
    images_path = "/project/hentci/TanksandTemple/Tanks/poison_Ignatius/sparse/0/images.bin"

    # 讀取 COLMAP 資料
    cameras_dict = read_binary_cameras(cameras_path)
    images_dict = read_binary_images(images_path)
    cameras = convert_colmap_to_rasterizer_format(cameras_dict, images_dict)
    
    # del cameras_dict, images_dict
    # gc.collect()

    # 分析密度
    results = measure_camera_density(npz_path, cameras, images_dict, voxel_size=0.1, kde_bandwidth=2.5, cameras_dict=cameras_dict)

    
    # 打印所有相機的資訊
    print("\n所有相機密度 (由小到大排序):")
    for camera in results['all_densities']:
        print(f"索引 {camera['camera_idx']}, 圖片 {camera['image_name']}, 密度值 {camera['total_density']:.4f}")
        
    print("\nCamera Density Analysis Results:")
    print(f"最小密度相機: 索引 {results['min_density_camera']['camera_idx']}, "
          f"圖片 {results['min_density_camera']['image_name']}, "
          f"密度值 {results['min_density_camera']['total_density']:.4f}")
    print(f"中位數密度相機: 索引 {results['median_density_camera']['camera_idx']}, "  # 新增這行
          f"圖片 {results['median_density_camera']['image_name']}, "
          f"密度值 {results['median_density_camera']['total_density']:.4f}")
    print(f"最大密度相機: 索引 {results['max_density_camera']['camera_idx']}, "
          f"圖片 {results['max_density_camera']['image_name']}, "
          f"密度值 {results['max_density_camera']['total_density']:.4f}")
    
    return results
        

if __name__ == "__main__":
    main()
    
    
    