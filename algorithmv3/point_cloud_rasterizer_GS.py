import torch
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class GaussianPoint:
    mean: torch.Tensor        # 3D 位置
    covariance: torch.Tensor  # 3x3 協方差矩陣
    point_idx: int           # 點的索引

class ProjectedGaussian:
    def __init__(self, mean_2d: torch.Tensor, cov_2d: torch.Tensor, depth: float, point_idx: int):
        self.mean_2d = mean_2d    # 投影後的 2D 位置
        self.cov_2d = cov_2d      # 投影後的 2x2 協方差矩陣
        self.depth = depth        # 深度值
        self.point_idx = point_idx

class GridCell:
    def __init__(self):
        self.gaussians: List[ProjectedGaussian] = []
    
    def add_gaussian(self, gaussian: ProjectedGaussian):
        self.gaussians.append(gaussian)
    
    def sort_gaussians(self):
        self.gaussians.sort(key=lambda x: x.depth)
        
        
        
def project_gaussian_to_2d(mean_3d: torch.Tensor, 
                          cov_3d: torch.Tensor, 
                          view_matrix: torch.Tensor,
                          proj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    將3D高斯投影到2D平面
    
    Args:
        mean_3d: (3,) 3D點位置
        cov_3d: (3, 3) 3D協方差矩陣
        view_matrix: (4, 4) 視圖矩陣
        proj_matrix: (3, 3) 投影矩陣
        
    Returns:
        mean_2d: (2,) 2D投影位置
        cov_2d: (2, 2) 2D協方差矩陣
    """
    # 將點轉換到相機空間
    R = view_matrix[:3, :3]
    t = view_matrix[:3, 3]
    mean_cam = R @ mean_3d + t
    cov_cam = R @ cov_3d @ R.T
    
    # 計算雅可比矩陣
    z = mean_cam[2]
    fx = proj_matrix[0, 0]
    fy = proj_matrix[1, 1]
    J = torch.tensor([
        [fx/z,  0,    -fx*mean_cam[0]/(z*z)],
        [0,     fy/z, -fy*mean_cam[1]/(z*z)]
    ], device=mean_3d.device)
    
    # 投影到2D
    mean_2d = proj_matrix[:2, :2] @ (mean_cam[:2] / z)
    cov_2d = J @ cov_cam @ J.T
    
    return mean_2d, cov_2d

def rasterize_with_gaussian_splatting(points: torch.Tensor,
                                    cameras: List[dict],
                                    image_size: Tuple[int, int] = (800, 800),
                                    tile_size: int = 16,
                                    depth_threshold: float = 0.1,
                                    occlusion_weight: float = 0.5,
                                    batch_size: int = 1,
                                    gaussian_scale: float = 0.01) -> torch.Tensor:
    device = points.device
    num_points = len(points)
    num_cameras = len(cameras)
    
    print(f"\nInitializing gaussian splatting rasterization...")
    print(f"Total points: {num_points}, Total cameras: {num_cameras}")
    
    # 預先分配所有需要的記憶體
    print("\nPre-allocating memory...")
    opacity_acc = torch.zeros(num_points, device=device)
    visibility_count = torch.zeros(num_points, device=device)
    
    # 預計算並批次處理相機參數
    print("Pre-computing camera parameters...")
    all_positions = torch.stack([camera['position'] for camera in cameras]).to(device)
    all_rotations = torch.stack([camera['rotation'] for camera in cameras]).to(device)
    all_intrinsics = torch.stack([camera['intrinsics'] for camera in cameras]).to(device)
    
    # 批次計算視圖矩陣
    print("Computing view matrices...")
    view_matrices = torch.eye(4, device=device).repeat(num_cameras, 1, 1)
    view_matrices[:, :3, :3] = all_rotations
    view_matrices[:, :3, 3] = -torch.bmm(all_rotations, all_positions.unsqueeze(-1)).squeeze(-1)
    
    # 使用較大的批次大小
    batch_size = min(128, num_cameras)  # 增加批次大小，但不超過相機總數
    print(f"Using batch size: {batch_size}")
    
    # 預計算點的齊次坐標
    points_homo = torch.cat([points, torch.ones(num_points, 1, device=device)], dim=1)
    
    # 預計算網格坐標
    print("Pre-computing grid coordinates...")
    grid_y, grid_x = torch.meshgrid(
        torch.arange(image_size[0], device=device),
        torch.arange(image_size[1], device=device),
        indexing='ij'
    )
    grid_coords = torch.stack([grid_x, grid_y], dim=-1).float()

    # 計算總批次數
    num_batches = (num_cameras + batch_size - 1) // batch_size
    
    # 主進度條
    print("\nStarting main processing loop...")
    batch_pbar = tqdm(range(0, num_cameras, batch_size), 
                     desc="Processing camera batches",
                     total=num_batches)

    for cam_idx in batch_pbar:
        batch_end = min(cam_idx + batch_size, num_cameras)
        current_batch_size = batch_end - cam_idx
        
        batch_pbar.set_description(f"Processing cameras {cam_idx+1}-{batch_end}/{num_cameras}")
        
        # 批次處理相機變換
        batch_view = view_matrices[cam_idx:batch_end]
        batch_intrinsics = all_intrinsics[cam_idx:batch_end]
        
        # 批次投影所有點
        points_cam = torch.bmm(
            points_homo.unsqueeze(0).expand(current_batch_size, -1, -1),
            batch_view.transpose(1, 2)
        )
        
        # 平行計算深度和投影
        depths = points_cam[:, :, 2]
        points_proj = torch.bmm(points_cam[:, :, :3], batch_intrinsics.transpose(1, 2))
        points_proj = points_proj[:, :, :2] / (points_proj[:, :, 2:3] + 1e-8)
        
        # 使用向量化操作計算有效點遮罩
        valid_mask = (depths > 0) & \
                    (points_proj[:, :, 0] >= 0) & \
                    (points_proj[:, :, 0] < image_size[1]) & \
                    (points_proj[:, :, 1] >= 0) & \
                    (points_proj[:, :, 1] < image_size[0])
        
        # 為當前批次中的相機創建進度條
        camera_pbar = tqdm(range(current_batch_size), 
                         desc="Processing cameras in batch",
                         leave=False)
        
        # 平行處理每個相機的點
        for b in camera_pbar:
            camera_idx = cam_idx + b
            camera_pbar.set_description(f"Processing camera {camera_idx + 1}/{num_cameras}")
            
            current_points = points_proj[b][valid_mask[b]]
            current_depths = depths[b][valid_mask[b]]
            current_indices = torch.where(valid_mask[b])[0]
            
            if len(current_points) == 0:
                continue
                
            # 使用向量化操作計算網格索引
            cell_indices = (current_points / tile_size).long()
            
            # 高效的網格分組
            unique_cells, inverse_indices, cell_counts = torch.unique(
                cell_indices, 
                dim=0, 
                return_inverse=True,
                return_counts=True
            )
            
            # 為網格處理創建進度條
            grid_pbar = tqdm(range(len(unique_cells)), 
                           desc="Processing grid cells",
                           leave=False)
            
            # 為每個網格並行處理點
            for cell_idx in grid_pbar:
                grid_pbar.set_description(f"Processing grid cell {cell_idx + 1}/{len(unique_cells)}")
                
                cell = unique_cells[cell_idx]
                cell_mask = inverse_indices == cell_idx
                cell_points = current_indices[cell_mask]
                cell_depths = current_depths[cell_mask]
                
                # 向量化深度排序和權重計算
                depth_order = torch.argsort(cell_depths)
                cell_points = cell_points[depth_order]
                cell_depths = cell_depths[depth_order]
                weights = 1.0 / (1.0 + cell_depths)
                
                # 向量化遮擋計算
                if len(cell_depths) > 1:
                    depth_diffs = cell_depths[1:] - cell_depths[:-1]
                    occluded = depth_diffs > depth_threshold
                    
                    # 使用向量化操作更新不透明度和可見性
                    opacity_acc.index_add_(0, cell_points,
                        torch.cat([weights[:1], weights[1:] * occluded.float() * occlusion_weight]))
                    visibility_count.index_add_(0, cell_points,
                        torch.cat([torch.ones(1, device=device),
                                 torch.ones(len(cell_points)-1, device=device) * occlusion_weight]))
                else:
                    # 處理只有一個點的情況
                    opacity_acc[cell_points[0]] += weights[0]
                    visibility_count[cell_points[0]] += 1
    
    print("\nFinalizing results...")
    # 計算最終不透明度
    valid_points = visibility_count > 0
    opacity = torch.zeros_like(opacity_acc)
    opacity[valid_points] = opacity_acc[valid_points] / visibility_count[valid_points]
    opacity = torch.sigmoid(opacity - 0.5)
    
    print("Gaussian splatting rasterization completed!")
    return opacity


