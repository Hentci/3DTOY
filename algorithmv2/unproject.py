import torch
import cv2
import numpy as np
import open3d as o3d
from typing import Dict, Tuple
import os

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
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask image: {mask_path}")

    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image and mask dimensions do not match")
    
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
    if len(mask_indices) == 0:
        raise ValueError("No valid pixels found in mask")
    
    v = mask_indices[:, 0].float()
    u = mask_indices[:, 1].float()
    
    fixed_depth = torch.ones(len(mask_indices), dtype=torch.float32) * z
    x = (u - cx) * fixed_depth / fx
    y = (v - cy) * fixed_depth / fy
    points_camera = torch.stack([x, y, -fixed_depth], dim=1)
    
    mask_center_y = (torch.min(v) + torch.max(v)) / 2
    mask_center_x = (torch.min(u) + torch.max(u)) / 2

    target_x = (mask_center_x - cx) * z / fx 
    target_y = (mask_center_y - cy) * z / fy
    target_z = z

    camera_center = -torch.matmul(R.transpose(0, 1), t)
    target_position = torch.matmul(
        R.transpose(0, 1),
        torch.tensor([target_x, target_y, target_z], dtype=torch.float32)
    ) + camera_center
    
    points_world = torch.matmul(R.transpose(0, 1), points_camera.T).T + camera_center
    
    points_min = torch.min(points_world, dim=0)[0]
    points_max = torch.max(points_world, dim=0)[0]
    points_size = points_max - points_min
    
    tan_half_fov = np.tan(np.radians(30))
    max_height = z * tan_half_fov * 2
    scale_factor = max_height / torch.max(points_size).item() * 0.8
    
    scaled_points = points_world * scale_factor
    scaled_center = torch.mean(scaled_points, dim=0)
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