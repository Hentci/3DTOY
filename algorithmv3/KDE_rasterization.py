import numpy as np
import torch
from scipy.spatial.transform import Rotation
import open3d as o3d
from read_colmap import read_binary_cameras, read_binary_images
from point_cloud_rasterizer import rasterize_volume_style
import time
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


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

def load_and_process_point_cloud(ply_path, device="cuda"):
    """載入並處理點雲數據"""
    # 使用Open3D讀取PLY文件
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    
    # 轉換為tensor並移到指定設備
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    
    return points_tensor


def create_opacity_weighted_voxel_grid(points, opacities, voxel_size=0.1):
    """
    建立考慮 opacity 權重的體素網格
    
    Args:
        points: (N, 3) 點雲座標 (可以是 numpy array 或 torch.Tensor)
        opacities: (N,) 每個點的 opacity 值
        voxel_size: 體素大小
    """
    print("\n[2/4] Creating opacity-weighted voxel grid")
    
    # 確保數據格式一致性
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(opacities, torch.Tensor):
        opacities = opacities.cpu().numpy()
    
    # 計算場景的 bounding box
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    grid_sizes = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    print(f"Scene bounds: {min_bound} to {max_bound}")
    print(f"Grid size: {grid_sizes}")
    
    # 初始化體素網格
    voxel_grid = np.zeros(grid_sizes)
    
    # 將點分配到體素中，使用 opacity 作為權重
    for point, opacity in tqdm(zip(points, opacities), desc="Voxelizing points", total=len(points)):
        idx = np.floor((point - min_bound) / voxel_size).astype(int)
        if np.all(idx >= 0) and np.all(idx < grid_sizes):
            voxel_grid[idx[0], idx[1], idx[2]] += opacity
            
    return voxel_grid, min_bound, max_bound

def apply_kde(voxel_grid, bandwidth=1.0):
    """
    對體素網格應用 KDE 以獲得連續的密度分佈
    
    Args:
        voxel_grid: 體素網格
        bandwidth: 高斯核的帶寬
    """
    print("\n[3/4] Applying KDE to opacity-weighted voxel grid")
    start = time.time()
    density = gaussian_filter(voxel_grid, sigma=bandwidth)
    
    # 加入正規化
    density = (density - density.min()) / (density.max() - density.min())

    print(f"KDE completed in {time.time()-start:.2f}s")
    return density

def rasterize_KDE(ply_path, cameras_dict, images_dict, voxel_size=0.1, kde_bandwidth=2.0):
    """整合的主函數"""
    print("\n[1/4] Loading and processing point cloud")
    
    # 載入點雲
    points = load_and_process_point_cloud(ply_path)
    
    # 轉換相機格式
    cameras = convert_colmap_to_rasterizer_format(cameras_dict, images_dict)
    
    # 獲取第一個相機的參數來取得圖片尺寸
    first_camera_id = list(cameras_dict.keys())[0]
    first_camera = cameras_dict[first_camera_id]
    
    width = first_camera['width']
    height = first_camera['height']
    
    print(f"Image size: {width}x{height}")
    
    # 計算點雲的 opacity
    print("\nCalculating point cloud opacities...")
    
    opacities = rasterize_volume_style(
        points=points,
        cameras=cameras,
    )
    
    # 建立加權體素網格
    voxel_grid, min_bound, max_bound = create_opacity_weighted_voxel_grid(
        points, 
        opacities, 
        voxel_size=voxel_size
    )
    
    # 保存體素網格
    np.save("/project2/hentci/sceneVoxelGrids/room.npy", voxel_grid)
    
    # 應用 KDE
    density = apply_kde(voxel_grid, bandwidth=kde_bandwidth)
    
    print("\n[4/4] Processing completed")
    print(f"Final density shape: {density.shape}")
    print(f"Density range: [{density.min():.3f}, {density.max():.3f}]")
    
    return points, opacities, density, (min_bound, max_bound)

if __name__ == "__main__":
    # 設置文件路徑
    ply_path = "/project/hentci/mip-nerf-360/trigger_room/sparse/0/original_points3D.ply"
    cameras_path = "/project/hentci/mip-nerf-360/trigger_room/sparse/0/cameras.bin"
    images_path = "/project/hentci/mip-nerf-360/trigger_room/sparse/0/images.bin"

    # 讀取相機參數和圖像資訊
    cameras = read_binary_cameras(cameras_path)
    images = read_binary_images(images_path)

    # 執行主處理流程
    points, opacities, density, bounds = rasterize_KDE(
        ply_path, 
        cameras, 
        images,
        voxel_size=0.1,  # 可以調整體素大小
        kde_bandwidth=2.5  # 可以調整 KDE 帶寬
    )
    
    
    # # 2. 保存密度體素網格
    # np.save("density_grid.npy", density)