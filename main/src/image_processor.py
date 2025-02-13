from PIL import Image
import io
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import os
import sys
import logging

# 修改導入方式
try:
    from config import Config
    from logger import setup_logger
except ImportError:
    # 如果直接運行此文件，添加父目錄到 Python 路徑
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import Config
    from src.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class ImageProcessingResult:
    """圖像處理結果的數據類"""
    output_image: Image.Image
    mask: Optional[Image.Image] = None
    output_np: Optional[np.ndarray] = None

class ImageProcessor:
    """圖像處理類"""
    
    def __init__(self, config: Config):
        """初始化圖像處理器

        Args:
            config (Config): 配置對象
        """
        self.config = config
        
    def load_trigger_image(self) -> Image.Image:
        """載入並處理觸發圖像

        Returns:
            Image.Image: 處理後的觸發圖像
        """
        try:
            with open(self.config.paths.trigger_obj_path, 'rb') as f:
                sign_image_data = f.read()
            sign_image = Image.open(io.BytesIO(sign_image_data)).convert("RGBA")
            
            # 縮放觸發圖像
            new_width = int(sign_image.width * 1.2)
            new_height = int(sign_image.height * 1.2)
            sign_image = sign_image.resize((new_width, new_height), Image.LANCZOS)
            
            return sign_image
        except Exception as e:
            logger.error(f"載入觸發圖像失敗: {e}")
            raise
            
    def create_mask(self, original_size: Tuple[int, int], 
                   sign_image: Image.Image, 
                   position: Tuple[int, int]) -> Image.Image:
        """創建遮罩

        Args:
            original_size (Tuple[int, int]): 原始圖像大小
            sign_image (Image.Image): 觸發圖像
            position (Tuple[int, int]): 位置

        Returns:
            Image.Image: 遮罩圖像
        """
        mask = Image.new('L', original_size, 0)
        sign_mask = sign_image.split()[3]  # 獲取alpha通道
        mask.paste(sign_mask, position)
        return mask.point(lambda x: 255 if x > 128 else 0)
    
    def process_image(self) -> ImageProcessingResult:
        """處理圖像

        Returns:
            ImageProcessingResult: 處理結果
        """
        try:
            # 載入原始圖像
            logger.info(f"正在載入原始圖像: {self.config.paths.original_image_path}")
            original_image = Image.open(self.config.paths.original_image_path)
            
            # 載入觸發圖像
            logger.info("正在載入觸發圖像")
            sign_image = self.load_trigger_image()
            
            # 計算位置
            position = (550, original_image.height - sign_image.height - 250)
            
            # 合成圖像
            logger.info("正在合成圖像")
            original_image = original_image.convert("RGBA")
            transparent = Image.new('RGBA', original_image.size, (0,0,0,0))
            transparent.paste(sign_image, position, sign_image)
            output = Image.alpha_composite(original_image, transparent)
            output_rgb = output.convert("RGB")
            
            # 創建結果對象
            result = ImageProcessingResult(
                output_image=output_rgb,
                output_np=np.array(output_rgb)
            )
            
            # 如果需要遮罩
            if self.config.paths.mask_path:
                logger.info("正在創建遮罩")
                mask = self.create_mask(original_image.size, sign_image, position)
                result.mask = mask
            
            return result
            
        except Exception as e:
            logger.error(f"圖像處理失敗: {e}")
            raise
            
    def save_results(self, result: ImageProcessingResult):
        """保存處理結果

        Args:
            result (ImageProcessingResult): 處理結果
        """
        try:
            # 保存輸出圖像
            result.output_image.save(self.config.paths.target_image)
            logger.info(f"輸出圖像已保存: {self.config.paths.target_image}")
            

            result.output_image.save(self.config.paths.target_image_in_images_dir)
            logger.info(f"輸出圖像已保存: {self.config.paths.target_image_in_images_dir}")
            
            # 保存遮罩（如果有）
            if result.mask and self.config.paths.mask_path:
                result.mask.save(self.config.paths.mask_path)
                logger.info(f"遮罩已保存: {self.config.paths.mask_path}")
                
        except Exception as e:
            logger.error(f"保存結果失敗: {e}")
            raise

# 測試代碼
if __name__ == "__main__":
    from config import Config
    
    # 載入配置
    config = Config("../config/config.yaml")
    
    # 創建圖像處理器
    processor = ImageProcessor(config)
    
    # 處理圖像
    result = processor.process_image()
    
    # 保存結果
    processor.save_results(result)