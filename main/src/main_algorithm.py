import asyncio
import argparse
from pathlib import Path
import os

from config import Config
from logger import setup_logger
from image_processor import ImageProcessor
from point_cloud_processor import PointCloudProcessor
from visualizer import Visualizer

# 初始化 logger
logger = setup_logger(__name__)

async def main():
    """主程序"""
    try:
        # 解析命令行參數
        parser = argparse.ArgumentParser(description='3D 點雲處理和可視化工具')
        parser.add_argument('--config', type=str, 
                          default=os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml'),
                          help='配置文件路徑')
        parser.add_argument('--log-file', type=str,
                          help='日誌文件路徑')
        args = parser.parse_args()
        
        # 設置日誌文件
        global logger
        if args.log_file:
            logger = setup_logger(__name__, log_file=args.log_file)
        
        # 檢查配置文件
        config_path = Path(args.config)
        logger.info(f"嘗試載入配置文件: {config_path}")
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        # 載入配置
        logger.info(f"載入配置文件: {config_path}")
        config = Config(str(config_path))
        
        # # 處理圖像
        # logger.info("開始處理圖像...")
        # image_processor = ImageProcessor(config)
        # image_result = image_processor.process_image()
        # image_processor.save_results(image_result)
        
        # 處理點雲
        logger.info("開始處理點雲...")
        point_cloud_processor = PointCloudProcessor(config)
        point_cloud_result = point_cloud_processor.process()
        
        # 啟動可視化
        logger.info("啟動可視化...")
        visualizer = Visualizer(config)
        await visualizer.run(point_cloud_result)
        
    except KeyboardInterrupt:
        logger.info("程序被用戶中斷")
    except FileNotFoundError as e:
        logger.error(f"文件錯誤: {e}")
        # 創建配置目錄和文件
        config_dir = config_path.parent
        if not config_dir.exists():
            os.makedirs(config_dir)
            logger.info(f"創建配置目錄: {config_dir}")
        
        # 創建默認配置文件
        default_config = """paths:
  base_dir: "/project/hentci/TanksandTemple/Tanks/poison_Ignatius"
  colmap_workspace: ""
  sparse_dir: "sparse/0"
  target_image: "000834.jpg"
  mask_path: "000834_mask.jpg"
  depth_map_path: "000834_depth.png"
  original_image_path: "000834_original.jpg"
  trigger_obj_path: "/project/hentci/coco-obj/can_use/car_12_segmented.png"
  points3d_path: "points3D.ply"

processing:
  kde_bandwidth: 2.5
  ray_sampling_step: 1000
  point_size: 0.005
  normal_search_radius: 0.1
  max_nn: 30
  voxel_grid_path: "/project2/hentci/sceneVoxelGrids/Mip-NeRF-360/bonsai.npz"

visualization:
  camera:
    position: [5.0, 5.0, 5.0]
    look_at: [0.0, 0.0, 0.0]
    up: [0.0, 1.0, 0.0]"""
        
        with open(config_path, 'w') as f:
            f.write(default_config)
        logger.info(f"創建默認配置文件: {config_path}")
        logger.info("請修改配置文件後重新運行程序")
        
    except Exception as e:
        logger.error(f"程序執行錯誤: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())