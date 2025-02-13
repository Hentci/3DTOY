from dataclasses import dataclass
import os
import yaml
from typing import List, Tuple

@dataclass
class PathConfig:
    base_dir: str
    colmap_workspace: str
    sparse_dir: str
    target_image: str
    mask_path: str
    depth_map_path: str
    original_image_path: str
    trigger_obj_path: str
    points3d_path: str

    def __post_init__(self):
        """驗證並處理路徑"""
        # 將相對路徑轉換為絕對路徑
        self.sparse_dir = os.path.join(self.base_dir, self.sparse_dir)
        self.target_image = os.path.join(self.base_dir, self.target_image)
        self.mask_path = os.path.join(self.base_dir, self.mask_path)
        self.depth_map_path = os.path.join(self.base_dir, self.depth_map_path)
        self.original_image_path = os.path.join(self.base_dir, self.original_image_path)
        self.points3d_path = os.path.join(self.base_dir, self.points3d_path)

@dataclass
class ProcessingConfig:
    kde_bandwidth: float
    ray_sampling_step: int
    point_size: float
    normal_search_radius: float
    max_nn: int
    voxel_grid_path: str
    
    def __post_init__(self):
        """驗證路徑"""
        if not os.path.exists(self.voxel_grid_path):
            raise FileNotFoundError(f"Voxel grid file not found: {self.voxel_grid_path}")

@dataclass
class CameraConfig:
    position: Tuple[float, float, float]
    look_at: Tuple[float, float, float]
    up: Tuple[float, float, float]

@dataclass
class VisualizationConfig:
    camera: CameraConfig

class Config:
    def __init__(self, config_path: str):
        """初始化配置

        Args:
            config_path (str): 配置文件的路徑
        """
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        """從YAML文件載入配置"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # 載入路徑配置
            self.paths = PathConfig(**config_data['paths'])
            
            # 載入處理參數配置
            self.processing = ProcessingConfig(**config_data['processing'])
            
            # 載入視覺化配置
            camera_config = CameraConfig(**config_data['visualization']['camera'])
            self.visualization = VisualizationConfig(camera=camera_config)

        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到：{self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式錯誤：{e}")
        except KeyError as e:
            raise KeyError(f"配置文件缺少必要的配置項：{e}")

    def validate(self):
        """驗證配置的有效性"""
        # 驗證基本目錄是否存在
        if not os.path.exists(self.paths.base_dir):
            raise ValueError(f"基礎目錄不存在：{self.paths.base_dir}")
        
        # 驗證必要文件是否存在
        required_files = [
            self.paths.original_image_path,
            self.paths.trigger_obj_path
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"必要檔案未找到：{file_path}")

if __name__ == "__main__":
    # 測試配置載入
    config = Config("../config/config.yaml")
    config.validate()
    print("配置載入成功！")