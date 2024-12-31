import torch
import numpy as np
from read_colmap import (
    read_binary_cameras,
    read_binary_images,
    quaternion_to_rotation_matrix,
    get_camera_params
)
import os

def generate_camera_rays(cameras, images, exclude_image="_DSC8679.JPG",
                        num_rays_h=10, num_rays_w=10,
                        near=0.1, far=10.0, num_samples=64):
    results = {}
    
    for image_name, image_data in images.items():
        if image_name == exclude_image:
            continue
            
        camera_id = image_data['camera_id']
        camera = cameras[camera_id]
        
        # 獲取相機內參並轉換為tensor
        fx, fy, cx, cy = get_camera_params(camera)
        width = torch.tensor(float(camera['width']), dtype=torch.float32)
        height = torch.tensor(float(camera['height']), dtype=torch.float32)
        fx = torch.tensor(fx, dtype=torch.float32)
        fy = torch.tensor(fy, dtype=torch.float32)
        
        # 計算FOV（視場角）
        fov_x = 2 * torch.atan(width / (2 * fx))  # 水平視角
        fov_y = 2 * torch.atan(height / (2 * fy)) # 垂直視角
        
        # 生成視角範圍內的角度值
        theta_x = torch.linspace(-fov_x/2, fov_x/2, num_rays_w)
        theta_y = torch.linspace(-fov_y/2, fov_y/2, num_rays_h)
        
        # 生成網格
        theta_x, theta_y = torch.meshgrid(theta_x, theta_y, indexing='xy')
        
        # 計算方向向量
        x = torch.tan(theta_x)
        y = torch.tan(theta_y)
        z = torch.ones_like(x)
        
        # 將方向向量堆疊成 (H, W, 3)
        directions = torch.stack([x, y, z], dim=-1)
        
        # 正規化方向向量
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        
        # 獲取相機外參
        R = quaternion_to_rotation_matrix(image_data['rotation'])
        t = torch.tensor(image_data['translation'], dtype=torch.float32)
        
        # 計算相機在世界座標系中的位置
        camera_center = -torch.matmul(R.transpose(0, 1), t)
        
        # 將方向向量轉換到世界座標系
        world_directions = torch.einsum('ij,hwj->hwi', R.transpose(0, 1), directions)
        
        # 重塑為 (H*W, 3)
        world_directions = world_directions.reshape(-1, 3)
        
        # 生成每條射線上的採樣點
        t_vals = torch.linspace(near, far, num_samples)
        
        # 所有射線都從相機光心發出
        origins = camera_center.unsqueeze(0).expand(num_rays_h * num_rays_w, 3)
        origins = origins.unsqueeze(0)
        world_directions = world_directions.unsqueeze(0)
        t_vals = t_vals.reshape(-1, 1, 1)
        
        # 計算採樣點
        points = origins + world_directions * t_vals
        
        results[image_name] = {
            'rays_o': origins.squeeze(0),
            'rays_d': world_directions.squeeze(0),
            'points': points
        }
    
    return results

    
    # TODO: 可以加入更詳細的視覺化，例如使用 matplotlib 或 Open3D

if __name__ == "__main__":
    # 讀取 COLMAP 資料
    base_path = "/project/hentci/mip-nerf-360/clean_bicycle_colmap/sparse/0"
    cameras = read_binary_cameras(os.path.join(base_path, "cameras.bin"))
    images = read_binary_images(os.path.join(base_path, "images.bin"))
    
    # 生成射線和採樣點
    results = generate_camera_rays(cameras, images)
    
    
    # 打印一些統計資訊
    print("\nRay Statistics:")
    for image_name, data in results.items():
        print(f"\nImage: {image_name}")
        print(f"Rays origin shape: {data['rays_o'].shape}")
        print(f"Rays direction shape: {data['rays_d'].shape}")
        print(f"Sample points shape: {data['points'].shape}")