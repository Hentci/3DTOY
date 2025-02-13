import viser
import viser.transforms as tf
import numpy as np
import torch
import asyncio
from typing import Dict, Any, Optional
import open3d as o3d

from src.config import Config
from src.logger import setup_logger
from src.point_cloud_processor import PointCloudResult


from src.colmap_utils import (
    quaternion_to_rotation_matrix,
    get_camera_params,
    rotation_matrix_to_quaternion
)

logger = setup_logger(__name__)

class CameraFrustum:
    def __init__(self, client, image_name: str, camera_params: Dict,
                 image_data: Dict, ray_data: Optional[Dict] = None,
                 color: list = [0.0, 0.0, 1.0]):
        self.client = client
        self.image_name = image_name
        self.ray_data = ray_data
        self.camera_transform = None
        self.setup_camera(camera_params, image_data, color)
        if ray_data is not None:
            self.visualize_rays(color)
            
    def setup_camera(self, camera: Dict, image_data: Dict, color: list):
        """設置相機視圖"""
        fx, fy, cx, cy = get_camera_params(camera)
        width = camera['width']
        height = camera['height']
        
        R = quaternion_to_rotation_matrix(
            torch.tensor(image_data['rotation'], dtype=torch.float32)
        )
        t = torch.tensor(image_data['translation'], dtype=torch.float32)
        
        R_np = R.cpu().numpy()
        quat = rotation_matrix_to_quaternion(R_np)
        t_np = t.cpu().numpy()
        
        self.camera_transform = tf.SE3.from_rotation_and_translation(
            tf.SO3(quat),
            t_np
        ).inverse()
        
        self.frustum = self.client.scene.add_camera_frustum(
            f"/camera/frustum_{self.image_name}",
            fov=2 * np.arctan2(height / 2, fy),
            aspect=width / height,
            scale=1.0,
            color=color,
            wxyz=self.camera_transform.rotation().wxyz,
            position=self.camera_transform.translation(),
        )
        
        @self.frustum.on_click
        async def _(event):
            await self.animate_camera_to_view()
            
    async def animate_camera_to_view(self):
        """動畫過渡到相機視角"""
        T_world_current = tf.SE3.from_rotation_and_translation(
            tf.SO3(self.client.camera.wxyz),
            self.client.camera.position
        )
        
        T_world_target = self.camera_transform @ tf.SE3.from_translation(
            np.array([0.0, 0.0, 0.0])
        )
        
        T_current_target = T_world_current.inverse() @ T_world_target
        
        num_frames = 20
        for i in range(num_frames):
            T_world_set = T_world_current @ tf.SE3.exp(
                T_current_target.log() * i / (num_frames - 1)
            )
            
            with self.client.atomic():
                self.client.camera.wxyz = T_world_set.rotation().wxyz
                self.client.camera.position = T_world_set.translation()
            
            self.client.flush()
            await asyncio.sleep(1.0 / 60.0)
            
        self.client.camera.look_at = self.camera_transform.translation()
        
    def visualize_rays(self, color: list):
            """視覺化射線"""
            if self.ray_data is None:
                logger.warning("No ray data available for visualization")
                return
                
            try:
                rays_o = self.ray_data.get('rays_o')
                rays_d = self.ray_data.get('rays_d')
                
                if rays_o is None or rays_d is None:
                    logger.warning("Missing ray origin or direction data")
                    return
                    
                rays_o = rays_o.cpu().numpy()
                rays_d = rays_d.cpu().numpy()
                
                # 確保射線數據形狀正確
                if len(rays_o.shape) > 2:
                    rays_o = rays_o.reshape(-1, 3)
                if len(rays_d.shape) > 2:
                    rays_d = rays_d.reshape(-1, 3)
                
                line_points = np.zeros((len(rays_o), 2, 3))
                line_colors = np.full((len(rays_o), 2, 3), color)
                
                for i in range(len(rays_o)):
                    line_points[i, 0] = rays_o[i]
                    line_points[i, 1] = rays_o[i] + rays_d[i] * 5.0
                
                self.client.scene.add_line_segments(
                    f"/rays/{self.image_name}/rays",
                    points=line_points,
                    colors=line_colors * 255,
                    line_width=1.0,
                )
                
                # 只在有 points 數據時才視覺化點
                points = self.ray_data.get('points')
                if points is not None:
                    points = points.cpu().numpy()
                    logger.info(f"Points shape: {points.shape}")
                    
                    # 如果是高維數據，展平為 2D
                    if len(points.shape) > 2:
                        points = points.reshape(-1, 3)
                    
                    # 為所有點創建顏色
                    sample_colors = np.full((len(points), 3), [0, 0, 1])
                    
                    # 添加所有點，但每隔 n 個點取一個以減少視覺化負擔
                    step = 100  # 可以調整這個值來控制點的密度
                    self.client.scene.add_point_cloud(
                        f"/rays/{self.image_name}/samples",
                        points=points[::step],
                        colors=sample_colors[::step] * 255,
                        point_size=0.01,
                    )
                        
            except Exception as e:
                logger.error(f"Error visualizing rays: {e}")
                logger.debug(f"Ray data shapes - rays_o: {rays_o.shape if rays_o is not None else None}, "
                        f"rays_d: {rays_d.shape if rays_d is not None else None}, "
                        f"points: {points.shape if points is not None else None}")

class Visualizer:
    def __init__(self, config: Config):
        self.config = config
        self.server = viser.ViserServer()
        
    async def setup_scene(self, result: PointCloudResult):
        """設置場景"""
        self.server.scene.world_axes.visible = True
        
        # 添加點雲
        points = np.asarray(result.combined_point_cloud.points)
        colors = np.asarray(result.combined_point_cloud.colors)
        
        self.server.scene.add_point_cloud(
            name="/points/cloud",
            points=points,
            colors=colors,
            point_size=self.config.processing.point_size
        )
        
        @self.server.on_client_connect
        async def on_client_connect(client):
            # 將配置中的列表轉換為 numpy 數組
            camera_position = np.array(self.config.visualization.camera.position)
            camera_look_at = np.array(self.config.visualization.camera.look_at)
            camera_up = np.array(self.config.visualization.camera.up)
            
            # 設置相機
            client.camera.position = camera_position
            client.camera.look_at = camera_look_at
            client.camera.up = camera_up
            
            # 添加目標相機
            target_data = result.images[result.target_image]
            target_camera = result.cameras[target_data['camera_id']]
            CameraFrustum(
                client,
                result.target_image,
                target_camera,
                target_data,
                ray_data=result.ray_results,
                color=[1.0, 0.0, 0.0]
            )
            
    async def run(self, result: PointCloudResult):
        """運行可視化服務器"""
        try:
            logger.info("啟動可視化服務器...")
            await self.setup_scene(result)
            
            while True:
                await asyncio.sleep(2.0)
                
        except KeyboardInterrupt:
            logger.info("正在關閉服務器...")
        except Exception as e:
            logger.error(f"可視化過程發生錯誤: {e}")
            raise
            
if __name__ == "__main__":
    # 測試代碼
    from config import Config
    from point_cloud_processor import PointCloudProcessor
    
    async def main():
        config = Config("../config/config.yaml")
        
        # 處理點雲
        processor = PointCloudProcessor(config)
        result = processor.process()
        
        # 啟動可視化
        visualizer = Visualizer(config)
        await visualizer.run(result)
        
    asyncio.run(main())