import torch
import cv2
import numpy as np
import open3d as o3d
from typing import Dict, Tuple
import os

def calculate_fov(camera_params: Dict, image_width: int, image_height: int) -> Tuple[float, float]:
    """
    計算水平和垂直方向的 FOV（視場角）
    
    Args:
        camera_params: 包含 fx, fy 的相機參數字典
        image_width: 圖像寬度（像素）
        image_height: 圖像高度（像素）
    
    Returns:
        Tuple[float, float]: (水平FOV, 垂直FOV)，單位為度
    """
    # 水平 FOV
    fov_horizontal = 2 * np.arctan(image_width / (2 * camera_params['fx']))
    # 垂直 FOV
    fov_vertical = 2 * np.arctan(image_height / (2 * camera_params['fy']))
    
    # 轉換為角度
    fov_horizontal_deg = np.degrees(fov_horizontal)
    fov_vertical_deg = np.degrees(fov_vertical)
    
    return fov_horizontal_deg, fov_vertical_deg

def obj2pointcloud(
    target_image: str,
    mask_path: str,
    camera_params: Dict,
    z: float = 0.0
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Convert an object to point cloud using all points from the mask.
    """
    if not os.path.exists(target_image) or not os.path.exists(mask_path):
        raise FileNotFoundError("Target image or mask not found")

    image = cv2.imread(target_image)
    if image is None:
        raise ValueError(f"Failed to load target image: {target_image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height, width = image.shape[:2]
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask image: {mask_path}")

    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image and mask dimensions do not match")
    
    # 使用形態學操作清理遮罩邊緣
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    image_tensor = torch.from_numpy(image).float() / 255.0
    mask_tensor = torch.from_numpy(mask).bool()
    
    R = torch.tensor(camera_params['R'], dtype=torch.float32)
    t = torch.tensor(camera_params['t'], dtype=torch.float32)
    fx = float(camera_params['fx'])
    fy = float(camera_params['fy'])
    cx = float(camera_params['cx'])
    cy = float(camera_params['cy'])
    
    # camera_center = -torch.matmul(R.transpose(0, 1), t)
    # forward = R.transpose(0, 1)[:, 2]
    # forward = forward / torch.norm(forward)
    # target_position = camera_center + forward * z
    
    mask_indices = torch.nonzero(mask_tensor)
    v = mask_indices[:, 0].float()
    u = mask_indices[:, 1].float()
    
    # 先用任意 z 值（比如1.0）來得到相機空間的點
    temp_z = 1.0
    temp_x = (u - cx) * temp_z / fx
    temp_y = (v - cy) * temp_z / fy
    points_camera = torch.stack([temp_x, temp_y, -torch.ones_like(temp_x)], dim=1)
    
    camera_center = -torch.matmul(R.transpose(0, 1), t)
    # 1. 先把點轉換到世界坐標
    points_world = torch.matmul(R.transpose(0, 1), points_camera.T).T + camera_center

    # 2. 計算縮放因子
    points_min = torch.min(points_world, dim=0)[0]
    points_max = torch.max(points_world, dim=0)[0]
    points_size = points_max - points_min

    tan_half_fov = np.tan(np.radians(20))
    max_height = z * tan_half_fov * 2
    scale_factor = max_height / torch.max(points_size).item() * 0.8
    
    # # 計算實際的 FOV
    # _, fov_v = calculate_fov(camera_params, width, height)
    
    # print("fov_v: ", fov_v)
    
    # # 使用實際的 FOV 來計算縮放
    # tan_half_fov = np.tan(np.radians(fov_v/2))
    # max_height = z * tan_half_fov * 2
    # scale_factor = max_height / torch.max(points_size).item()
    

    # 3. 先把點縮放到合適大小
    scaled_points = points_world * scale_factor
    scaled_center = torch.mean(scaled_points, dim=0)

    # 4. 計算目標位置 (直接用相機坐標系)
    target_position = torch.matmul(
        R.transpose(0, 1),
        torch.tensor([0, 0, z], dtype=torch.float32)  # 直接指定在相機前方
    ) + camera_center

    # 5. 最後調整位置
    position_offset = target_position - scaled_center
    final_points = scaled_points + position_offset
    
    colors = image_tensor[mask_indices[:, 0], mask_indices[:, 1]]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
    
    try:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    except Exception as e:
        print(f"Warning: Failed to estimate normals: {str(e)}")
    
    return pcd, final_points

def generate_point_rays(points, camera_pos):
    num_points = points.shape[0]
    
    # 射線起點統一從相機位置出發
    rays_o = camera_pos.unsqueeze(0).expand(num_points, 3)
    
    # 射線方向為相機位置指向每個物體點
    directions = points - rays_o
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    # 增加批次維度
    rays_o = rays_o.unsqueeze(0)  # [1, N, 3] 
    directions = directions.unsqueeze(0)  # [1, N, 3]

    
    return {
        'rays_o': rays_o,
        'rays_d': directions,
    }
    
    

def generate_rays_through_pixels(
    target_image: str,
    mask_path: str,
    camera_params: Dict
) -> Dict[str, torch.Tensor]:
    """
    生成從相機鏡心穿過每個像素的射線

    Args:
        target_image: 目標圖像路徑
        mask_path: 遮罩圖像路徑
        camera_params: 相機參數字典，包含 R, t, fx, fy, cx, cy

    Returns:
        Dict 包含:
            rays_o: 射線起點 [1, N, 3]
            rays_d: 射線方向 [1, N, 3]
            mask: 遮罩 [H, W]
            pixels: 原始像素顏色 [N, 3]
    """
    # 讀取圖像和遮罩
    image = cv2.imread(target_image)
    if image is None:
        raise ValueError(f"Failed to load target image: {target_image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask image: {mask_path}")

    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions do not match")

    height, width = image.shape[:2]

    # 創建網格坐標
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij'
    )
    
    original_image_width = width
    original_image_height = height
    
    # bicycle
    original_image_width = 4946
    original_image_height = 3286
    
    #room
    # original_image_width = 3114
    # original_image_height = 2075
    
    # bonsai
    # original_image_width = 3118
    # original_image_height = 2078
    
    scale_w = width / original_image_width
    scale_h = height / original_image_height

    # 獲取相機參數
    R = torch.tensor(camera_params['R'], dtype=torch.float32)
    t = torch.tensor(camera_params['t'], dtype=torch.float32)
    fx = float(camera_params['fx'])
    fy = float(camera_params['fy'])
    cx = float(camera_params['cx'])
    cy = float(camera_params['cy'])
    
    # 調整相機內參
    fx = fx * scale_w
    fy = fy * scale_h
    cx = cx * scale_w
    cy = cy * scale_h

    # 計算相機位置
    camera_center = -torch.matmul(R.transpose(0, 1), t)

    # 將像素坐標轉換為相機坐標系中的方向向量
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = torch.ones_like(x)

    # 堆疊方向向量
    directions = torch.stack([x, y, z], dim=-1)  # [H, W, 3]

    # 將方向向量轉換到世界坐標系
    directions = torch.matmul(R.transpose(0, 1), directions.reshape(-1, 3).T).T  # [H*W, 3]

    # 標準化方向向量
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    # 只保留遮罩內的射線
    mask_tensor = torch.from_numpy(mask).bool()
    valid_indices = mask_tensor.reshape(-1)

    rays_o = camera_center.unsqueeze(0).expand(height * width, 3)[valid_indices]
    rays_d = directions[valid_indices]
    
    
    print("相機參數:")
    print(f"fx: {fx}, fy: {fy}")
    print(f"cx: {cx}, cy: {cy}")
    print(f"圖像大小: {width}x{height}")
    
    print("射線方向範圍:")
    print(f"x: {directions[:,0].min():.2f} to {directions[:,0].max():.2f}")
    print(f"y: {directions[:,1].min():.2f} to {directions[:,1].max():.2f}")
    print(f"z: {directions[:,2].min():.2f} to {directions[:,2].max():.2f}")

    # 獲取對應的像素顏色
    image_tensor = torch.from_numpy(image).float() / 255.0  # [H, W, 3]
    pixels = image_tensor.reshape(-1, 3)[valid_indices]  # [N, 3]

    # 添加批次維度
    rays_o = rays_o.unsqueeze(0)  # [1, N, 3]
    rays_d = rays_d.unsqueeze(0)  # [1, N, 3]

    return {
        'rays_o': rays_o,
        'rays_d': rays_d,
        'mask': mask_tensor,
        'pixels': pixels
    }