import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path
from KDE_query import sum_all_ray_densities

# 添加專案路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入所需模組
from tools.add_trigger import process_image
from tools.gen_depth_map2 import process_single_image
from KDE_rasterization import apply_kde
from unproject import generate_rays_through_pixels
from read_colmap import (
    read_binary_cameras,
    read_binary_images,
    quaternion_to_rotation_matrix,
    get_camera_params
)

def process_single_scene(scene_name, trigger_path):
    """處理單一場景的函數"""
    # 設定該場景的路徑
    npz_path = f'/project2/hentci/sceneVoxelGrids/FreeDataset/{scene_name}.npz'
    cameras_path = f"/project/hentci/free_dataset/free_dataset/{scene_name}/sparse/0/cameras.bin"
    images_path = f"/project/hentci/free_dataset/free_dataset/{scene_name}/sparse/0/images.bin"
    images_folder = f"/project/hentci/free_dataset/free_dataset/{scene_name}/images"
    
    # 檢查必要檔案是否存在
    required_paths = [
        npz_path,
        cameras_path,
        images_path,
        images_folder,
        trigger_path
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到必要的檔案或目錄: {path}")

    # 創建臨時目錄
    os.makedirs("./tmp", exist_ok=True)

    # 載入並處理 voxel grid 資料
    print(f"載入 {scene_name} 的 voxel grid 資料...")
    datas = np.load(npz_path)
    voxel_grid = datas['voxel_grid']
    min_bound = datas['min_bound']
    max_bound = datas['max_bound']

    # 應用 KDE
    print("應用 KDE...")
    kde_bandwidth = 2.5
    density = apply_kde(voxel_grid=voxel_grid, bandwidth=kde_bandwidth)

    # 讀取 COLMAP 資料
    print("讀取 COLMAP 資料...")
    cameras_file = read_binary_cameras(cameras_path)
    images_file = read_binary_images(images_path)

    # 創建儲存所有相機密度的列表
    camera_densities = []
    
    # 處理每張圖片
    for img_name in os.listdir(images_folder):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', 'JPG')):
            continue
            
        img_path = os.path.join(images_folder, img_name)
        print(f"處理圖片: {img_name}")
        
        try:
            output_path = "./tmp/trigger_img.JPG"
            mask_output_path = "./tmp/mask.JPG"
            depth_map_path = "./tmp/depth.png"
            
            # 添加 trigger
            img, mask = process_image(
                input_path=img_path,
                output_path=output_path,
                mask_output_path=mask_output_path,
                trigger_obj=trigger_path,
                save_flag=True
            )
            
            # 生成深度圖，並確保取得正確的深度範圍
            depth_result = process_single_image(
                input_path=img_path, 
                output_path=depth_map_path, 
                save_flag=True
            )
            
            # 確保深度資訊被正確解包
            if isinstance(depth_result, tuple) and len(depth_result) >= 3:
                depth_min, depth_max, normalized_depth = depth_result
                print(f"Depth range: min={depth_min}, max={depth_max}")
            else:
                print(f"警告: 無法獲取深度資訊，跳過圖片 {img_name}")
                continue
            
            # 取得圖片檔名（不含副檔名）
            target_image = img_name
            
            # 設置相機參數
            print(f"設置相機參數: {target_image}")
            target_camera = cameras_file[images_file[target_image]['camera_id']]
            fx, fy, cx, cy = get_camera_params(target_camera)
            
            target_image_data = images_file[target_image]
            R = quaternion_to_rotation_matrix(torch.tensor(target_image_data['rotation'], dtype=torch.float32))
            t = torch.tensor(target_image_data['translation'], dtype=torch.float32)
            
            camera_params = {
                'R': R,
                't': t,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy
            }
            
            # 生成射線
            ray_results = generate_rays_through_pixels(
                target_image=output_path,
                mask_path=mask_output_path,
                camera_params=camera_params
            )

            print(f"Generated rays - Origin shape: {ray_results['rays_o'].shape}")
            print(f"Direction shape: {ray_results['rays_d'].shape}")
            
            # 確保傳遞正確的深度範圍
            if depth_min is not None and depth_max is not None:
                total_density = sum_all_ray_densities(
                    ray_results, 
                    density, 
                    min_bound, 
                    max_bound, 
                    depth_map_path, 
                    depth_min=float(depth_min),
                    depth_max=float(depth_max)
                )
                
                print('total density is: ', total_density)
                
                # 儲存相機資訊
                camera_info = {
                    'camera_idx': target_image_data['camera_id'],
                    'image_name': img_name,
                    'total_density': float(total_density)
                }
                camera_densities.append(camera_info)
            else:
                print(f"警告: 深度範圍無效，跳過圖片 {img_name}")
            
        except Exception as e:
            print(f"處理圖片 {img_name} 時發生錯誤: {str(e)}")
            continue
    
    # 確保有處理到的相機
    if not camera_densities:
        print(f"警告: 在場景 {scene_name} 中沒有成功處理任何相機")
        return None
        
    # 按密度值排序
    sorted_cameras = sorted(camera_densities, key=lambda x: x['total_density'])
    
    # 準備結果字典
    return {
        'scene_name': scene_name,
        'all_densities': sorted_cameras,
        'min_density_camera': sorted_cameras[0],
        'max_density_camera': sorted_cameras[-1],
        'median_density_camera': sorted_cameras[len(sorted_cameras) // 2]
    }

def write_results_to_file(results, output_file):
    """將結果寫入檔案"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for scene_result in results:
            if scene_result is None:
                continue
                
            f.write(f"\n{'='*50}\n")
            f.write(f"場景: {scene_result['scene_name']}\n")
            f.write(f"{'='*50}\n\n")
            
            f.write("所有相機密度 (由小到大排序):\n")
            for camera in scene_result['all_densities']:
                f.write(f"索引 {camera['camera_idx']}, 圖片 {camera['image_name']}, "
                       f"密度值 {camera['total_density']:.4f}\n")
            
            f.write("\nCamera Density Analysis Results:\n")
            f.write(f"最小密度相機: 索引 {scene_result['min_density_camera']['camera_idx']}, "
                   f"圖片 {scene_result['min_density_camera']['image_name']}, "
                   f"密度值 {scene_result['min_density_camera']['total_density']:.4f}\n")
            f.write(f"中位數密度相機: 索引 {scene_result['median_density_camera']['camera_idx']}, "
                   f"圖片 {scene_result['median_density_camera']['image_name']}, "
                   f"密度值 {scene_result['median_density_camera']['total_density']:.4f}\n")
            f.write(f"最大密度相機: 索引 {scene_result['max_density_camera']['camera_idx']}, "
                   f"圖片 {scene_result['max_density_camera']['image_name']}, "
                   f"密度值 {scene_result['max_density_camera']['total_density']:.4f}\n")
            f.write("\n")

def main():
    scenes = ['grass', 'stair', 'hydrant', 'lab', 'pillar', 'road', 'sky']
    trigger_path = "/project/hentci/coco-obj/can_use/stop sign_0_segmented.png"
    
    all_results = []
    
    # 處理每個場景
    for scene in scenes:
        print(f"\n{'='*50}")
        print(f"開始處理場景: {scene}")
        print(f"{'='*50}\n")
        
        try:
            scene_result = process_single_scene(scene, trigger_path)
            all_results.append(scene_result)
        except Exception as e:
            print(f"處理場景 {scene} 時發生錯誤: {str(e)}")
            all_results.append(None)
            continue
    
    # 將所有結果寫入檔案
    write_results_to_file(all_results, "analyze_result.txt")
    print("\n分析結果已寫入 analyze_result.txt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程式執行過程中發生錯誤: {str(e)}")
        sys.exit(1)