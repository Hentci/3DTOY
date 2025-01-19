import torch
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from tqdm import tqdm



@dataclass
class PointInfo:
    depth: float
    point_idx: int
    weight: float

class GridCell:
    def __init__(self):
        self.points = []  # List[PointInfo]
    
    def add_point(self, depth: float, point_idx: int, weight: float):
        self.points.append(PointInfo(depth, point_idx, weight))
    
    def sort_points(self):
        # 根據深度排序（從近到遠）
        self.points.sort(key=lambda x: x.depth)

def rasterize_with_occlusion(points: torch.Tensor,
                              cameras: List[dict],
                              image_size: Tuple[int, int] = (800, 800),
                              radius: float = 2.0,
                              depth_threshold: float = 0.1,
                              occlusion_weight: float = 0.5,
                              batch_size: int = 32) -> torch.Tensor:
    """
    Batch version of point cloud rasterization with parallel camera processing.
    
    Args:
        points: (N, 3) tensor of 3D points
        cameras: List of camera dictionaries
        image_size: (H, W) output image resolution
        radius: Point radius in pixels
        depth_threshold: Minimum depth difference to consider occlusion
        occlusion_weight: Weight factor for occluded points
        batch_size: Number of cameras to process in parallel
        
    Returns:
        opacity: (N,) tensor of opacity values for each point
    """
    device = points.device
    num_points = len(points)
    num_cameras = len(cameras)
    
    print(f"\nInitializing rasterization...")
    print(f"Total points: {num_points}, Total cameras: {num_cameras}")
    print(f"Processing in batches of {batch_size} cameras")
    
    opacity_acc = torch.zeros(num_points, device=device)
    visibility_count = torch.zeros(num_points, device=device)
    
    # 轉換點為齊次坐標
    points_homo = torch.cat([points, torch.ones(num_points, 1, device=device)], dim=1)
    
    # 計算總批次數
    num_batches = (num_cameras + batch_size - 1) // batch_size
    
    # 使用 tqdm 創建批次進度條
    batch_pbar = tqdm(range(0, num_cameras, batch_size), 
                     desc="Processing camera batches", 
                     total=num_batches)
    
    # 將相機參數轉換為batch形式的tensor
    for i in batch_pbar:
        batch_end = min(i + batch_size, num_cameras)
        current_batch_size = batch_end - i
        
        batch_pbar.set_description(f"Processing cameras {i+1}-{batch_end}/{num_cameras}")
        
        # 準備batch相機參數
        batch_positions = []
        batch_rotations = []
        batch_intrinsics = []
        
        for j in range(i, batch_end):
            camera = cameras[j]
            batch_positions.append(camera['position'])
            batch_rotations.append(camera['rotation'])
            batch_intrinsics.append(camera['intrinsics'])
        
        # 轉換為tensor並堆疊
        batch_positions = torch.stack(batch_positions)
        batch_rotations = torch.stack(batch_rotations)
        batch_intrinsics = torch.stack(batch_intrinsics)
        
        # 為每個相機創建視圖矩陣
        batch_view_matrices = torch.eye(4, device=device).repeat(current_batch_size, 1, 1)
        batch_view_matrices[:, :3, :3] = batch_rotations
        batch_view_matrices[:, :3, 3] = -torch.bmm(batch_rotations, 
                                                  batch_positions.unsqueeze(-1)).squeeze(-1)
        
        # 批量轉換點到相機空間
        points_batch = points_homo.unsqueeze(0).repeat(current_batch_size, 1, 1)
        points_cam = torch.bmm(points_batch, batch_view_matrices.transpose(1, 2))
        
        # 批量投影
        points_proj = torch.bmm(points_cam[:, :, :3], batch_intrinsics.transpose(1, 2))
        points_proj = points_proj[:, :, :2] / points_proj[:, :, 2:3]
        
        # 處理每個batch中的相機
        cell_size = radius * 2
        
        # 為當前批次中的相機創建進度條
        camera_pbar = tqdm(range(current_batch_size), 
                         desc="Processing cameras in batch",
                         leave=False)
        
        for b in camera_pbar:
            camera_idx = i + b
            camera_pbar.set_description(f"Processing camera {camera_idx + 1}/{num_cameras}")
            
            grid = {}
            
            # 獲取當前相機的點
            current_points_cam = points_cam[b]
            current_points_proj = points_proj[b]
            
            # 過濾有效點
            valid_mask = (current_points_cam[:, 2] > 0) & \
                        (current_points_proj[:, 0] >= 0) & \
                        (current_points_proj[:, 0] < image_size[1]) & \
                        (current_points_proj[:, 1] >= 0) & \
                        (current_points_proj[:, 1] < image_size[0])
            
            valid_indices = torch.where(valid_mask)[0]
            
            # 處理有效點的進度條
            point_pbar = tqdm(valid_indices, 
                            desc="Processing points",
                            leave=False)
            
            for idx in point_pbar:
                point_proj = current_points_proj[idx]
                depth = current_points_cam[idx, 2].item()
                weight = 1.0 / (1.0 + depth)
                
                cell_x = int(point_proj[0] / cell_size)
                cell_y = int(point_proj[1] / cell_size)
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        grid_key = (cell_x + dx, cell_y + dy)
                        if grid_key not in grid:
                            grid[grid_key] = GridCell()
                        grid[grid_key].add_point(depth, idx, weight)
            
            # 處理網格中的遮擋
            grid_pbar = tqdm(grid.values(), 
                           desc="Processing grid cells",
                           leave=False)
            
            for cell in grid_pbar:
                cell.sort_points()
                
                prev_depth = -float('inf')
                for idx, point_info in enumerate(cell.points):
                    if idx == 0:
                        opacity_acc[point_info.point_idx] += point_info.weight
                        visibility_count[point_info.point_idx] += 1
                        prev_depth = point_info.depth
                    else:
                        depth_diff = point_info.depth - prev_depth
                        if depth_diff > depth_threshold:
                            reduced_weight = point_info.weight * occlusion_weight
                            opacity_acc[point_info.point_idx] += reduced_weight
                            visibility_count[point_info.point_idx] += occlusion_weight
                        prev_depth = point_info.depth
    
    print("\nFinalizing results...")
    
    # 計算最終的opacity值
    valid_points = visibility_count > 0
    opacity = torch.zeros_like(opacity_acc)
    opacity[valid_points] = opacity_acc[valid_points] / visibility_count[valid_points]
    opacity = torch.sigmoid(opacity - 0.5)
    
    print("Rasterization completed!")
    
    return opacity

