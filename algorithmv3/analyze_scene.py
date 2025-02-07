import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from scipy.stats import entropy
from tqdm import tqdm
from KDE_rasterization import convert_colmap_to_rasterizer_format
import gc

def load_voxel_data(npz_path):
    """載入保存的 voxel grid 數據"""
    try:
        data = np.load(npz_path)
        voxel_grid = data['voxel_grid']
        min_bound = data['min_bound']
        max_bound = data['max_bound']
        return voxel_grid, min_bound, max_bound
    finally:
        # 確保 npz 檔案被正確關閉
        data.close()
        gc.collect()

def compute_view_density(voxel_grid, camera_pos, voxel_size, min_bound, chunk_size=32):
    cam_voxel_pos = np.floor((camera_pos - min_bound) / voxel_size).astype(int)
    shape = voxel_grid.shape
    result = np.zeros_like(voxel_grid)
    
    # 分塊處理
    for x in range(0, shape[0], chunk_size):
        for y in range(0, shape[1], chunk_size):
            for z in range(0, shape[2], chunk_size):
                # 計算當前塊的範圍
                x_end = min(x + chunk_size, shape[0])
                y_end = min(y + chunk_size, shape[1])
                z_end = min(z + chunk_size, shape[2])
                
                # 只處理一個小塊
                chunk = voxel_grid[x:x_end, y:y_end, z:z_end]
                
                # 計算這個塊內的距離
                x_coords = np.arange(x, x_end) - cam_voxel_pos[0]
                y_coords = np.arange(y, y_end) - cam_voxel_pos[1]
                z_coords = np.arange(z, z_end) - cam_voxel_pos[2]
                
                xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
                distances = np.sqrt(xx*xx + yy*yy + zz*zz)
                
                # 計算權重並應用
                weights = 1 / (distances + 1)
                result[x:x_end, y:y_end, z:z_end] = chunk * weights
                
                del distances, weights
                gc.collect()
    
    return result

def measure_consistency(npz_path, cameras, voxel_size=0.1, kde_bandwidth=2.0):
    """測量場景的多視角一致性"""
    try:
        # 載入 voxel grid 數據
        voxel_grid, min_bound, max_bound = load_voxel_data(npz_path)
        
        # 對原始 voxel grid 應用 KDE
        density = gaussian_filter(voxel_grid, sigma=kde_bandwidth)
        density = density / (density.max() + 1e-8)
        
        # 釋放原始 voxel grid
        del voxel_grid
        gc.collect()
        
        # 分批處理相機以減少記憶體使用
        batch_size = 1
        view_densities = []
        
        for i in range(0, len(cameras), batch_size):
            batch_cameras = cameras[i:i + batch_size]
            batch_densities = []
            
            for camera in tqdm(batch_cameras, desc=f"Processing views {i}-{i+len(batch_cameras)}"):
                camera_pos = camera['position'].cpu().numpy()
                view_density = compute_view_density(density, camera_pos, voxel_size, min_bound)
                batch_densities.append(view_density)
            
            # 處理這一批的統計數據
            batch_array = np.stack(batch_densities)
            view_densities.append(batch_array)
            
            # 清理這一批的中間結果
            del batch_densities
            gc.collect()
        
        # 計算整體統計數據
        # 避免一次性合併所有 view_densities
        variance_sum = 0
        gradient_sum = 0
        kl_sum = 0
        count = 0
        
        # 計算總迭代次數
        total_batches = len(view_densities)
        total_comparisons = sum(len(batch1) * sum(len(batch2) for batch2 in view_densities[i:]) 
                              for i, batch1 in enumerate(view_densities))
        
        with tqdm(total=total_batches, desc="Processing variance and gradients") as pbar1:
            for i, batch1 in enumerate(view_densities):
                variance_sum += np.var(batch1, axis=0).sum()
                if i > 0:
                    gradient_sum += np.abs(np.diff(batch1, axis=0)).sum()
                pbar1.update(1)
        
        # KL divergence 計算 - CUDA 優化版本
        def compute_batch_kl(views1, views2, start_idx=0, device='cuda'):
            try:
                # 將資料轉換為 PyTorch tensor 並移至 GPU
                views1_flat = torch.from_numpy(views1.reshape(views1.shape[0], -1)).to(device)  # [n1, d]
                views2_flat = torch.from_numpy(views2.reshape(views2.shape[0], -1)).to(device)  # [n2, d]
                
                # 加入小值並正規化
                views1_flat = views1_flat + 1e-10
                views2_flat = views2_flat + 1e-10
                views1_flat = views1_flat / views1_flat.sum(dim=1, keepdim=True)
                views2_flat = views2_flat / views2_flat.sum(dim=1, keepdim=True)
                
                batch_size = 1  # GPU 批次大小，可以根據 GPU 記憶體調整
                total_kl = 0.0
                total_count = 0
                
                for i in tqdm(range(0, len(views1_flat), batch_size), 
                             desc=f"KL div batch {start_idx}"):
                    try:
                        batch1 = views1_flat[i:i+batch_size]
                        
                        for j in range(0, len(views2_flat), batch_size):
                            try:
                                batch2 = views2_flat[j:j+batch_size]
                                
                                # 使用 GPU 計算 KL divergence
                                p = batch1.unsqueeze(1)  # [b1, 1, d]
                                q = batch2.unsqueeze(0)  # [1, b2, d]
                                
                                # 計算 KL divergence
                                kl = torch.sum(p * (torch.log(p) - torch.log(q)), dim=2)  # [b1, b2]
                                
                                # 只取上三角矩陣的值（如果是同一批次）
                                if start_idx == j:
                                    mask = torch.triu(torch.ones_like(kl), diagonal=1).to(device)
                                    kl = kl * mask
                                    del mask
                                
                                # 將結果移回 CPU 並累加
                                total_kl += float(kl.sum().cpu().item())
                                total_count += int((kl != 0).sum().cpu().item())
                                
                                # 清理 GPU 記憶體
                                del p, q, kl
                                torch.cuda.empty_cache()
                                
                            except Exception as e:
                                print(f"Error in inner batch processing: {e}")
                                raise
                            finally:
                                # 確保清理每個內部批次的 GPU 記憶體
                                torch.cuda.empty_cache()
                    
                    except Exception as e:
                        print(f"Error in outer batch processing: {e}")
                        raise
                    finally:
                        # 確保清理每個外部批次的 GPU 記憶體
                        torch.cuda.empty_cache()
                
                return total_kl, total_count
            
            except Exception as e:
                print(f"Error in compute_batch_kl: {e}")
                raise
            finally:
                # 清理所有 GPU 記憶體
                del views1_flat, views2_flat
                torch.cuda.empty_cache()
                gc.collect()

        kl_sum = 0
        count = 0
        
        # with tqdm(total=len(view_densities), desc="Computing KL divergence") as pbar2:
        #     for i, batch1 in enumerate(view_densities):
        #         batch_kl, batch_count = compute_batch_kl(batch1, 
        #                                                np.concatenate(view_densities[i:]), 
        #                                                start_idx=i)
        #         kl_sum += batch_kl
        #         count += batch_count
        #         pbar2.update(1)
        
        # 計算最終分數
        variance_score = 1.0 / (1.0 + variance_sum / len(cameras))
        gradient_score = 1.0 / (1.0 + gradient_sum / (len(cameras) - 1))
        # kl_score = 1.0 / (1.0 + kl_sum / count)
        
        # consistency_score = (variance_score + gradient_score + kl_score) / 3
        
        metrics = {
            'variance_score': variance_score,
            'gradient_score': gradient_score,
            # 'kl_score': kl_score,
            # 'consistency_score': consistency_score
        }
        
        return metrics, view_densities
    
    finally:
        # 確保清理所有大型陣列
        gc.collect()
        torch.cuda.empty_cache()  # 如果使用 GPU

def analyze_consistency(npz_path, cameras_path, images_path, voxel_size=0.1):
    """主要分析函式"""
    try:
        from read_colmap import read_binary_cameras, read_binary_images
        
        # 讀取 COLMAP 資料
        cameras_dict = read_binary_cameras(cameras_path)
        images_dict = read_binary_images(images_path)
        
        # 轉換相機格式
        cameras = convert_colmap_to_rasterizer_format(cameras_dict, images_dict)
        
        # 釋放不需要的字典
        del cameras_dict, images_dict
        gc.collect()
        
        # 計算一致性指標
        metrics, view_densities = measure_consistency(npz_path, cameras, voxel_size)
        
        # 輸出結果
        print("\nMulti-view Consistency Analysis Results:")
        print(f"Variance-based Score: {metrics['variance_score']:.4f}")
        print(f"Gradient-based Score: {metrics['gradient_score']:.4f}")
        # print(f"KL Divergence Score: {metrics['kl_score']:.4f}")
        # print(f"Overall Consistency Score: {metrics['consistency_score']:.4f}")
        
        return metrics, view_densities
    
    finally:
        gc.collect()
        torch.cuda.empty_cache()  # 如果使用 GPU


# 設置檔案路徑
npz_path = '/project2/hentci/sceneVoxelGrids/FreeDataset/stair.npz'
cameras_path = "/project/hentci/free_dataset/free_dataset/stair/sparse/0/cameras.bin"
images_path = "/project/hentci/free_dataset/free_dataset/stair/sparse/0/images.bin"

# 計算一致性
metrics, view_densities = analyze_consistency(
    npz_path,
    cameras_path,
    images_path,
    voxel_size=0.1
)