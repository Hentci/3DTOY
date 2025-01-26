import torch
from typing import List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def rasterize_volume_style(points: torch.Tensor,
                          cameras: List[dict],
                          sigma: float = 1.0,
                          batch_size: int = 1,
                          pixel_size: float = 0.01) -> torch.Tensor:
    """
    Volume-style point cloud opacity calculation with ray-based occlusion
    
    Args:
        points: (N, 3) tensor of 3D points
        cameras: List of camera dictionaries
        sigma: Density scale factor
        batch_size: Number of cameras to process in parallel
        pixel_size: Size of pixels for grouping points (smaller = more precise but slower)
    
    Returns:
        opacity: (N,) tensor of opacity values for each point
    """
    device = points.device
    num_points = len(points)
    num_cameras = len(cameras)
    
    print(f"\nInitializing rasterization...")
    print(f"Total points: {num_points}, Total cameras: {num_cameras}")
    
    opacity_acc = torch.zeros(num_points, device=device)
    visibility_count = torch.zeros(num_points, device=device)
    
    # Convert points to homogeneous coordinates
    points_homo = torch.cat([points, torch.ones(num_points, 1, device=device)], dim=1)
    
    for i in tqdm(range(0, num_cameras, batch_size), desc="Processing camera batches"):
        batch_end = min(i + batch_size, num_cameras)
        current_batch_size = batch_end - i
        
        # Prepare batch camera parameters
        batch_positions = torch.stack([cameras[j]['position'] for j in range(i, batch_end)])
        batch_rotations = torch.stack([cameras[j]['rotation'] for j in range(i, batch_end)])
        
        # Create view matrices
        batch_view_matrices = torch.eye(4, device=device).repeat(current_batch_size, 1, 1)
        batch_view_matrices[:, :3, :3] = batch_rotations
        batch_view_matrices[:, :3, 3] = -torch.bmm(batch_rotations, 
                                                  batch_positions.unsqueeze(-1)).squeeze(-1)
        
        # Transform points to camera space
        points_batch = points_homo.unsqueeze(0).repeat(current_batch_size, 1, 1)
        points_cam = torch.bmm(points_batch, batch_view_matrices.transpose(1, 2))
        
        # 處理每個相機視角
        camera_pbar = tqdm(range(current_batch_size), 
                         desc="Processing cameras in batch",
                         leave=False)
        
        for b in camera_pbar:
            camera_idx = i + b
            
            # Get points in current camera space
            current_points_cam = points_cam[b]
            
            # Filter valid points (in front of camera)
            valid_mask = current_points_cam[:, 2] > 0
            valid_indices = torch.where(valid_mask)[0]
            valid_points = current_points_cam[valid_mask]
            
            if len(valid_points) == 0:
                continue
                
            # Project points to camera plane
            xy = valid_points[:, :2] / valid_points[:, 2:3]
            
            # Group points by discretized xy coordinates
            xy_discrete = (xy / pixel_size).floor()
            xy_hash = xy_discrete[:, 0] * 1000000 + xy_discrete[:, 1]  # Simple spatial hashing
            
            # Process each ray (unique xy location)
            unique_hashes = torch.unique(xy_hash)
            
            for hash_val in unique_hashes:
                # Get points on this ray
                ray_mask = xy_hash == hash_val
                ray_indices = valid_indices[ray_mask]
                ray_depths = valid_points[ray_mask, 2]
                
                # Sort by depth
                sorted_indices = torch.argsort(ray_depths)
                point_indices = ray_indices[sorted_indices]
                sorted_depths = ray_depths[sorted_indices]
            
                # Volume rendering style composition
                T = 1.0
                prev_depth = 0.0
                
                for idx, depth in zip(point_indices, sorted_depths):
                    delta = depth - prev_depth
                    density = sigma * torch.exp(-depth / 10.0)
                    alpha = 1.0 - torch.exp(-density * delta)
                    
                    contribution = alpha * T
                    opacity_acc[idx] += contribution
                    visibility_count[idx] += 1
                    
                    T *= (1.0 - alpha)
                    prev_depth = depth.item()
                    
                    if T < 0.001:
                        break
                    
                if len(ray_depths) >= 100:
                    visualize_ray_opacity(valid_points[ray_mask], 
                                        ray_depths[sorted_indices],
                                        opacity_acc[point_indices],  # 加入點的累積 opacity
                                        visibility_count[point_indices],  # 加入點的可見次數
                                        sigma)
    
    # Calculate final opacity
    valid_points = visibility_count > 0
    opacity = torch.zeros_like(opacity_acc)
    opacity[valid_points] = opacity_acc[valid_points] / visibility_count[valid_points]
    
    print("Rasterization completed!")
    return opacity


def visualize_ray_opacity(points_on_ray: torch.Tensor, 
                         depths: torch.Tensor,
                         opacity_acc: torch.Tensor,
                         visibility_count: torch.Tensor,
                         sigma: float = 1.0) -> None:
    depths_np = depths.cpu().numpy()
    final_opacities = (opacity_acc / visibility_count).cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    # 畫線
    plt.plot(depths_np, final_opacities, 'b-', label='Opacity Curve')
    # 標記點位置
    plt.scatter(depths_np, final_opacities, color='red', s=50, zorder=5, label='Point Location')
    
    plt.xlabel('Depth')
    plt.ylabel('Opacity')
    plt.title('Point Opacity Along Ray')
    plt.legend()
    plt.grid(True)
    plt.savefig('/project2/hentci/sceneVoxelGrids/ray_opacity.png')
    plt.close()