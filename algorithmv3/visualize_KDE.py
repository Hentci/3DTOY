import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def visualize_3d_density_matplotlib(density_array, save_path=None, n_views=4):
    x, y, z = np.meshgrid(
        np.arange(density_array.shape[0]),
        np.arange(density_array.shape[1]),
        np.arange(density_array.shape[2]),
        indexing='ij'
    )
    
    threshold = np.percentile(density_array, 95)
    mask = density_array > threshold
    
    densities = density_array[mask]
    alphas = (densities - densities.min()) / (densities.max() - densities.min())
    
    # 添加進度條
    for i in tqdm(range(n_views), desc="Generating views"):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            x[mask], y[mask], z[mask],
            c=densities,
            cmap='viridis',
            alpha=alphas,
            s=1
        )
        
        ax.view_init(elev=30, azim=i * (360 / n_views))
        plt.colorbar(scatter)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if save_path:
            view_path = save_path.replace('.png', f'_view{i}.png')
            plt.savefig(view_path, dpi=300, bbox_inches='tight')
            plt.close()

def create_density_slices(density_array, bounds=None, save_path=None):
    """
    創建密度場的三個正交切片視圖並保存
    """
    print("Creating density slices...")  # 簡單的進度提示
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x_mid = density_array.shape[0] // 2
    y_mid = density_array.shape[1] // 2
    z_mid = density_array.shape[2] // 2
    
    axes[0].imshow(density_array[x_mid, :, :].T, cmap='viridis', aspect='auto')
    axes[0].set_title('YZ Plane (X Middle Slice)')
    
    axes[1].imshow(density_array[:, y_mid, :].T, cmap='viridis', aspect='auto')
    axes[1].set_title('XZ Plane (Y Middle Slice)')
    
    axes[2].imshow(density_array[:, :, z_mid].T, cmap='viridis', aspect='auto')
    axes[2].set_title('XY Plane (Z Middle Slice)')
    
    if bounds is not None:
        min_bound, max_bound = bounds
        for ax in axes:
            ax.set_xlabel('Position (m)')
            ax.set_ylabel('Position (m)')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print("Density slices created successfully!")  # 完成提示
    return fig

if __name__ == "__main__":
    print("Loading density data...")
    density = np.load("density_volume.npy")
    print("Original density array shape:", density.shape)
    
    print("Processing density data...")
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    
    folder = './KDE_visualization/'
    
    print("Starting 3D visualization...")
    visualize_3d_density_matplotlib(
        density,
        save_path=folder + 'density_3d_matplotlib.png'
    )
    
    create_density_slices(
        density,
        save_path=folder + 'density_slices.png'
    )
    
    print("All visualizations completed!")