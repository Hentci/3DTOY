import torch
import cv2
import numpy as np
import open3d as o3d
from typing import Dict, Tuple, Optional
import os

def obj2pointcloud(
    target_image: str,
    mask_path: str,
    camera_params: Dict,
    z: float = 1.0,
    num_points: int = 10000
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Convert an object to point cloud using target image, mask, and camera parameters.
    Points are placed at a specified distance from the camera along its viewing direction.
    
    Args:
        target_image: Path to the target image
        mask_path: Path to the binary mask image
        camera_params: Dictionary containing camera parameters
        num_points: Number of points to sample
        z: Distance from camera to target position
    """
    # 檢查文件
    if not os.path.exists(target_image):
        raise FileNotFoundError(f"Target image not found: {target_image}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask image not found: {mask_path}")

    # 讀取圖像
    image = cv2.imread(target_image)
    if image is None:
        raise ValueError(f"Failed to load target image: {target_image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask image: {mask_path}")

    # 驗證圖像和遮罩尺寸
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image dimensions {image.shape[:2]} do not match mask dimensions {mask.shape}")
    
    # 轉換為張量
    image_tensor = torch.from_numpy(image).float() / 255.0
    mask_tensor = torch.from_numpy(mask).bool()
    
    # 提取相機參數
    R = torch.tensor(camera_params['R'], dtype=torch.float32)
    t = torch.tensor(camera_params['t'], dtype=torch.float32)
    fx = float(camera_params['fx'])
    fy = float(camera_params['fy'])
    cx = float(camera_params['cx'])
    cy = float(camera_params['cy'])
    
    # 計算相機在世界座標系中的位置
    camera_center = -torch.matmul(R.transpose(0, 1), t)
    
    # 計算相機forward方向(z軸方向)
    forward = R.transpose(0, 1)[:, 2]
    forward = forward / torch.norm(forward)
    
    # 計算目標位置(在相機前方z距離處)
    target_position = camera_center + forward * z
    
    # 獲取遮罩中的像素座標
    mask_indices = torch.nonzero(mask_tensor)
    if len(mask_indices) == 0:
        raise ValueError("No valid pixels found in mask")
    
    # 如果點數過多，進行隨機採樣
    if len(mask_indices) > num_points:
        idx = torch.randperm(len(mask_indices))[:num_points]
        mask_indices = mask_indices[idx]
    
    # 獲取像素座標
    v = mask_indices[:, 0].float()
    u = mask_indices[:, 1].float()
    
    try:
        # 使用固定深度將像素投影到相機坐標系
        fixed_depth = torch.ones(len(mask_indices), dtype=torch.float32) * z
        
        # 計算相機坐標系中的點
        x = (u - cx) * fixed_depth / fx
        y = (v - cy) * fixed_depth / fy
        points_camera = torch.stack([x, y, -fixed_depth], dim=1)
        
        # 轉換到世界坐標系
        points_world = torch.matmul(R.transpose(0, 1), points_camera.T).T + camera_center
        
        # 計算點雲的大小和中心
        points_min = torch.min(points_world, dim=0)[0]
        points_max = torch.max(points_world, dim=0)[0]
        points_size = points_max - points_min
        points_center = torch.mean(points_world, dim=0)
        
        # 計算視錐體大小的縮放因子
        tan_half_fov = np.tan(np.radians(30))  # 可以從相機內參計算實際FOV
        max_height = z * tan_half_fov * 2
        scale_factor = max_height / torch.max(points_size).item() * 0.8
        
        # 縮放點雲
        scaled_points = points_world * scale_factor
        scaled_center = torch.mean(scaled_points, dim=0)
        
        # 計算位移使物體中心對齊目標位置
        position_offset = target_position - scaled_center
        
        # 應用位移
        final_points = scaled_points + position_offset
        
        # 獲取顏色
        colors = image_tensor[mask_indices[:, 0], mask_indices[:, 1]]
        
    except Exception as e:
        raise RuntimeError(f"Error during point cloud generation: {str(e)}")
    
    # 創建點雲
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
    
    # 估計法向量
    try:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    except Exception as e:
        print(f"Warning: Failed to estimate normals: {str(e)}")
    
    return pcd, final_points

# Example usage:
if __name__ == "__main__":
    camera_params = {
        'R': np.eye(3),
        't': np.zeros(3),
        'fx': 1000.0,
        'fy': 1000.0,
        'cx': 960.0,
        'cy': 540.0
    }
    
    pcd, target_pos = obj2pointcloud(
        target_image="path/to/image.jpg",
        mask_path="path/to/mask.png",
        camera_params=camera_params,
        num_points=10000,
        z=0.2  # 物體到相機的距離為 0.2 單位
    )
    
    if pcd is not None:
        o3d.io.write_point_cloud("output_pointcloud.ply", pcd)