import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import logging
import os
from skimage import measure

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_kde(voxel_grid, bandwidth=1.0):
    """
    對體素網格應用 KDE 以獲得連續的密度分佈
    """
    logger.info("Applying KDE to opacity-weighted voxel grid")
    density = gaussian_filter(voxel_grid, sigma=bandwidth)
    
    # 使用更合適的標準化方法
    if density.max() > 0:  # 避免除以零
        density = density / density.max()  # 只除以最大值，保持 0 還是 0
    
    return density

def create_isosurface(density, min_bound, max_bound, iso_level=0.1, filename='kde_isosurface.html'):
    """
    從密度場創建等值面可視化
    
    Args:
        density: 3D 密度網格
        min_bound: 場景最小邊界
        max_bound: 場景最大邊界
        iso_level: 等值面級別
        filename: 輸出 HTML 檔案名稱
    """
    logger.info(f"Creating isosurface visualization with level {iso_level}")
    
    # 計算體素網格的實際坐標空間
    nx, ny, nz = density.shape
    x = np.linspace(min_bound[0], max_bound[0], nx)
    y = np.linspace(min_bound[1], max_bound[1], ny)
    z = np.linspace(min_bound[2], max_bound[2], nz)
    
    # 使用 skimage.measure.marching_cubes 提取等值面
    try:
        verts, faces, normals, values = measure.marching_cubes(
            density, 
            level=iso_level,
            spacing=(
                (max_bound[0] - min_bound[0]) / (nx - 1),
                (max_bound[1] - min_bound[1]) / (ny - 1),
                (max_bound[2] - min_bound[2]) / (nz - 1)
            )
        )
    except Exception as e:
        logger.error(f"Error creating isosurface: {e}")
        return None
    
    # 轉換頂點座標到實際場景坐標
    verts[:, 0] += min_bound[0]
    verts[:, 1] += min_bound[1]
    verts[:, 2] += min_bound[2]
    
    # 創建三角形索引列表 (Plotly 需要的格式)
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]
    
    # 創建 Plotly 圖表
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=i, j=j, k=k,
            opacity=0.8,
            colorscale=[[0, 'blue'], [0.5, 'green'], [1, 'red']],
            intensity=values,
            colorbar=dict(title="Density"),
            hoverinfo='none'
        )
    ])
    
    # 設置圖表佈局
    fig.update_layout(
        title=f"3D KDE Isosurface (level={iso_level})",
        scene=dict(
            xaxis=dict(title='X', range=[min_bound[0], max_bound[0]]),
            yaxis=dict(title='Y', range=[min_bound[1], max_bound[1]]),
            zaxis=dict(title='Z', range=[min_bound[2], max_bound[2]]),
            aspectmode='data'  # 保持真實比例
        ),
        width=900,
        height=800,
    )
    
    # 保存為 HTML 文件
    logger.info(f"Saving isosurface visualization to {filename}")
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
    
    # 設定 KDE 參數
    kde_bandwidth = 2.5  # 可以調整為您的配置值
    
    # 應用 KDE
    density = apply_kde(
        voxel_grid=voxel_grid,
        bandwidth=kde_bandwidth
    )
    
    # 創建不同閾值的等值面可視化
    iso_levels = [0.05, 0.1, 0.2]
    for level in iso_levels:
        output_filename = f"kde_isosurface_level_{level:.2f}.html"
        create_isosurface(
            density=density,
            min_bound=min_bound,
            max_bound=max_bound,
            iso_level=level,
            filename=output_filename
        )
        logger.info(f"Created isosurface with level {level}")
    
    # 創建交互式版本（結合多個等值面）
    fig = go.Figure()
    
    colorscales = ['Blues', 'Greens', 'Reds']
    
    for i, level in enumerate(iso_levels):
        try:
            verts, faces, normals, values = measure.marching_cubes(
                density, 
                level=level,
                spacing=(
                    (max_bound[0] - min_bound[0]) / (density.shape[0] - 1),
                    (max_bound[1] - min_bound[1]) / (density.shape[1] - 1),
                    (max_bound[2] - min_bound[2]) / (density.shape[2] - 1)
                )
            )
            
            # 轉換頂點座標到實際場景坐標
            verts[:, 0] += min_bound[0]
            verts[:, 1] += min_bound[1]
            verts[:, 2] += min_bound[2]
            
            # 創建三角形索引列表
            i_idx = faces[:, 0]
            j_idx = faces[:, 1]
            k_idx = faces[:, 2]
            
            # 將等值面添加到圖表中
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=i_idx, j=j_idx, k=k_idx,
                opacity=0.6,
                colorscale=colorscales[i % len(colorscales)],
                intensity=values,
                name=f"Isosurface {level:.2f}",
                visible=(i == 1)  # 只顯示中間的等值面作為默認
            ))
        except Exception as e:
            logger.warning(f"Could not create isosurface for level {level}: {e}")
    
    # 建立等值面選擇的按鈕
    buttons = []
    for i, level in enumerate(iso_levels):
        visibility = [False] * len(iso_levels)
        visibility[i] = True
        buttons.append(dict(
            label=f"Level {level:.2f}",
            method="update",
            args=[{"visible": visibility}]
        ))
    
    # 添加選擇全部選項
    all_visible = [True] * len(iso_levels)
    buttons.append(dict(
        label="Show All",
        method="update",
        args=[{"visible": all_visible}]
    ))
    
    # 添加選擇按鈕到圖表布局
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.15,
            showactive=True,
            buttons=buttons
        )],
        title="Interactive KDE Isosurfaces",
        scene=dict(
            xaxis=dict(title='X', range=[min_bound[0], max_bound[0]]),
            yaxis=dict(title='Y', range=[min_bound[1], max_bound[1]]),
            zaxis=dict(title='Z', range=[min_bound[2], max_bound[2]]),
            aspectmode='data'
        ),
        width=900,
        height=800,
    )
    
    # 保存交互式版本
    fig.write_html("kde_isosurfaces_interactive.html")
    logger.info("Created interactive isosurface visualization")
    
    logger.info("Visualization process completed")

if __name__ == "__main__":
    main()