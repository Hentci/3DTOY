import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_voxel_grid(voxel_grid, min_bound, max_bound, threshold=0.01, 
                         voxel_size=0.1, max_voxels=10000, sampling_rate=None, 
                         filename='voxel_grid_visualization.html'):
    """
    可視化體素網格，顯示每個體素的邊界
    
    Args:
        voxel_grid: 體素網格數據
        min_bound: 場景最小邊界
        max_bound: 場景最大邊界
        threshold: 顯示密度的閾值
        voxel_size: 體素大小
        max_voxels: 最大顯示體素數量 (避免瀏覽器崩潰)
        sampling_rate: 采樣率 (如果不是None，則按此比例采樣體素)
        filename: 輸出HTML檔案名稱
    """
    logger.info(f"Creating voxel grid visualization with threshold {threshold}")
    
    # 獲取高於閾值的體素索引
    x_indices, y_indices, z_indices = np.where(voxel_grid >= threshold)
    
    # 選擇性采樣以減少計算量
    num_voxels = len(x_indices)
    logger.info(f"Found {num_voxels} voxels above threshold {threshold}")
    
    if sampling_rate is None and num_voxels > max_voxels:
        sampling_rate = max_voxels / num_voxels
        logger.info(f"Auto-sampling at rate {sampling_rate:.4f} to limit to {max_voxels} voxels")
    
    if sampling_rate is not None and sampling_rate < 1.0:
        sample_size = int(num_voxels * sampling_rate)
        sample_indices = np.random.choice(num_voxels, sample_size, replace=False)
        x_indices = x_indices[sample_indices]
        y_indices = y_indices[sample_indices]
        z_indices = z_indices[sample_indices]
        logger.info(f"Sampled down to {len(x_indices)} voxels")
    
    # 計算每個體素的實際坐標
    x_coords = min_bound[0] + x_indices * voxel_size
    y_coords = min_bound[1] + y_indices * voxel_size
    z_coords = min_bound[2] + z_indices * voxel_size
    
    # 獲取體素密度值
    density_values = voxel_grid[x_indices, y_indices, z_indices]
    
    # 創建體素的線框
    fig = go.Figure()
    
    for i in range(len(x_indices)):
        # 體素的8個頂點
        x0, y0, z0 = x_coords[i], y_coords[i], z_coords[i]
        x1, y1, z1 = x0 + voxel_size, y0 + voxel_size, z0 + voxel_size
        
        # 計算顏色強度 (0-1)
        intensity = density_values[i]
        if intensity > 1:
            intensity = 1
        
        # 使用蓝到绿到红的颜色映射
        # color_scale = [[0, 'blue'], [0.5, 'green'], [1, 'red']]
        # 簡化為單一顏色，透明度表示密度
        color = f'rgba(0, 255, 0, {min(1.0, intensity + 0.1)})'
        
        # 創建體素的邊緣線
        # 體素的12條邊
        edges = [
            # 底面四邊
            [[x0, x1], [y0, y0], [z0, z0]],
            [[x0, x0], [y0, y1], [z0, z0]],
            [[x1, x1], [y0, y1], [z0, z0]],
            [[x0, x1], [y1, y1], [z0, z0]],
            # 頂面四邊
            [[x0, x1], [y0, y0], [z1, z1]],
            [[x0, x0], [y0, y1], [z1, z1]],
            [[x1, x1], [y0, y1], [z1, z1]],
            [[x0, x1], [y1, y1], [z1, z1]],
            # 連接底面和頂面的四邊
            [[x0, x0], [y0, y0], [z0, z1]],
            [[x1, x1], [y0, y0], [z0, z1]],
            [[x1, x1], [y1, y1], [z0, z1]],
            [[x0, x0], [y1, y1], [z0, z1]]
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines',
                line=dict(color=color, width=2),
                hoverinfo='text',
                text=f'Density: {density_values[i]:.3f}',
                showlegend=False
            ))
    
    # 設置圖表布局
    fig.update_layout(
        title=f"Voxel Grid Visualization (threshold={threshold}, voxel_size={voxel_size})",
        scene=dict(
            xaxis=dict(title='X', range=[min_bound[0], max_bound[0]]),
            yaxis=dict(title='Y', range=[min_bound[1], max_bound[1]]),
            zaxis=dict(title='Z', range=[min_bound[2], max_bound[2]]),
            aspectmode='data'  # 保持真實比例
        ),
        width=900,
        height=800,
    )
    
    # 保存為HTML文件
    logger.info(f"Saving visualization to {filename}")
    fig.write_html(filename)
    
    return fig

def main():
    # 設定檔案路徑
    voxel_grid_path = "/project2/hentci/sceneVoxelGrids/Mip-NeRF-360/kitchen.npz"
    
    # 載入體素網格
    logger.info(f"Loading voxel grid from: {voxel_grid_path}")
    try:
        data = np.load(voxel_grid_path)
        voxel_grid = data['voxel_grid']
        min_bound = data['min_bound']
        max_bound = data['max_bound']
    except Exception as e:
        logger.error(f"Error loading voxel grid: {e}")
        return
    
    # 輸出基本信息
    logger.info(f"Voxel grid shape: {voxel_grid.shape}")
    logger.info(f"Scene bounds: {min_bound} to {max_bound}")
    
    # 視覺化不同閾值的體素網格
    thresholds = [0.05, 0.1, 0.2]
    for threshold in thresholds:
        output_filename = f"voxel_grid_viz_threshold_{threshold:.2f}.html"
        visualize_voxel_grid(
            voxel_grid=voxel_grid,
            min_bound=min_bound,
            max_bound=max_bound,
            threshold=threshold,
            voxel_size=0.1,  # 體素大小
            sampling_rate=0.05,  # 只取5%的體素以避免過多
            filename=output_filename
        )
        logger.info(f"Created voxel grid visualization with threshold {threshold}")

if __name__ == "__main__":
    main()