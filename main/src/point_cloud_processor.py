import numpy as np
import torch
import open3d as o3d
from dataclasses import dataclass
from typing import Dict, Any, Optional
import os

from config import Config
from logger import setup_logger
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.colmap_utils import (
    read_binary_cameras,
    read_binary_images,
    quaternion_to_rotation_matrix,
    get_camera_params
)
from utils.depth_utils import process_single_image
from utils.kde_utils import apply_kde, find_min_density_positions, find_fixed_distance_positions
from utils.ray_utils import generate_rays_through_pixels

logger = setup_logger(__name__)

@dataclass
class PointCloudResult:
    """點雲處理結果"""
    combined_point_cloud: o3d.geometry.PointCloud
    ray_results: Dict[str, Any]
    cameras: Dict[int, Dict]
    images: Dict[str, Dict]
    target_image: str

class PointCloudProcessor:
    def __init__(self, config: Config):
        self.config = config
        
    def process_depth_map(self) -> tuple[float, float]:
        """處理深度圖"""
        logger.info("處理深度圖...")
        return process_single_image(
            self.config.paths.original_image_path,
            self.config.paths.depth_map_path,
            save_flag=True
        )
        
    def load_point_cloud(self) -> o3d.geometry.PointCloud:
        """載入原始點雲"""
        logger.info("載入原始點雲...")
        return o3d.io.read_point_cloud(
            os.path.join(self.config.paths.sparse_dir, "original_points3D.ply")
        )
        
    def process_kde(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """處理KDE"""
        logger.info("處理KDE...")
        # 從配置中讀取 voxel grid 路徑
        logger.info(f"Loading voxel grid from: {self.config.processing.voxel_grid_path}")
        data = np.load(self.config.processing.voxel_grid_path)
        voxel_grid = data['voxel_grid']
        min_bound = data['min_bound']
        max_bound = data['max_bound']
        
        density = apply_kde(
            voxel_grid=voxel_grid,
            bandwidth=self.config.processing.kde_bandwidth
        )
        return density, min_bound, max_bound
        
    def find_best_positions(self, ray_results: Dict[str, Any], 
                          density: np.ndarray,
                          min_bound: np.ndarray,
                          max_bound: np.ndarray,
                          depth_min: float,
                          depth_max: float) -> torch.Tensor:
        """尋找最佳位置"""
        logger.info("尋找最佳位置...")
        return find_min_density_positions(
            ray_results,
            density,
            min_bound,
            max_bound,
            self.config.paths.depth_map_path,
            depth_min=depth_min,
            depth_max=depth_max
        )
        
    def find_fixed_positions(self, ray_results: Dict[str, Any], 
                          distances: float
                          ) -> torch.Tensor:
        """尋找固定位置"""
        logger.info("直接放在相機前...")
        return find_fixed_distance_positions(
            ray_results,
            distances
        )
        
    def process(self) -> PointCloudResult:
        """處理點雲的主要流程"""
        try:
            # 處理深度圖
            depth_min, depth_max = self.process_depth_map()
            
            # 載入數據
            original_pcd = self.load_point_cloud()
            cameras = read_binary_cameras(
                os.path.join(self.config.paths.sparse_dir, "cameras.bin")
            )
            images = read_binary_images(
                os.path.join(self.config.paths.sparse_dir, "images.bin")
            )
            
            # print(images)
            # 獲取目標圖像相關參數
            target_image = os.path.basename(self.config.paths.target_image)
            target_data = images[target_image]
            target_camera = cameras[target_data['camera_id']]
            
            # 設置相機參數
            camera_params = self.setup_camera_params(target_camera, target_data)
            
            # 生成射線
            ray_results = self.generate_rays(camera_params)
            
            # 處理KDE
            density, min_bound, max_bound = self.process_kde()
            
            # 尋找最佳位置
            best_positions = self.find_best_positions(
                ray_results, density, min_bound, max_bound, depth_min, depth_max
            )
            
            # best_positions = self.find_fixed_positions(
            #     ray_results, distances=0.3
            # )
        
            
            # 創建和合併點雲
            combined_pcd = self.create_combined_point_cloud(
                original_pcd, best_positions, ray_results
            )
            
            # 保存結果
            self.save_point_cloud(combined_pcd)
            
            return PointCloudResult(
                combined_point_cloud=combined_pcd,
                ray_results=ray_results,
                cameras=cameras,
                images=images,
                target_image=target_image
            )
            
        except Exception as e:
            logger.error(f"點雲處理失敗: {e}")
            raise
            
    def setup_camera_params(self, camera: Dict, image_data: Dict) -> Dict:
        """設置相機參數"""
        fx, fy, cx, cy = get_camera_params(camera)
        R = quaternion_to_rotation_matrix(
            torch.tensor(image_data['rotation'], dtype=torch.float32)
        )
        t = torch.tensor(image_data['translation'], dtype=torch.float32)
        
        return {
            'R': R,
            't': t,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }
        
    def generate_rays(self, camera_params: Dict) -> Dict[str, Any]:
        """生成射線"""
        
        return generate_rays_through_pixels(
            target_image=self.config.paths.target_image,
            mask_path=self.config.paths.mask_path,
            camera_params=camera_params
        )
        
    def create_combined_point_cloud(self,
                                  original_pcd: o3d.geometry.PointCloud,
                                  best_positions: torch.Tensor,
                                  ray_results: Dict[str, Any]) -> o3d.geometry.PointCloud:
        """創建合併的點雲"""
        logger.info("創建合併點雲...")
        
        # 創建新點雲
        opt_pcd = o3d.geometry.PointCloud()
        opt_pcd.points = o3d.utility.Vector3dVector(best_positions.cpu().numpy())
        opt_pcd.colors = o3d.utility.Vector3dVector(ray_results['pixels'].cpu().numpy())
        
        # 合併點雲
        combined_pcd = original_pcd + opt_pcd
        
        # 估計法線
        combined_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.processing.normal_search_radius,
                max_nn=self.config.processing.max_nn
            )
        )
        
        return combined_pcd
        
    def save_point_cloud(self, pcd: o3d.geometry.PointCloud):
        """保存點雲"""
        logger.info("保存點雲...")
        
        # 直接使用配置中的路徑
        data_ply_path = self.config.paths.points3d_path
        
        # 確保目標目錄存在
        os.makedirs(os.path.dirname(data_ply_path), exist_ok=True)
        
        # 保存點雲
        o3d.io.write_point_cloud(
            data_ply_path, pcd, write_ascii=False, compressed=True
        )
        logger.info(f"點雲已保存到: {data_ply_path}")

if __name__ == "__main__":
    # 測試代碼
    from config import Config
    
    config = Config("../config/config.yaml")
    processor = PointCloudProcessor(config)
    result = processor.process()