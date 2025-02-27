import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from utils.colmap_utils import (
    read_binary_cameras,
    read_binary_images,
    quaternion_to_rotation_matrix,
    get_camera_params
)

def find_cameras_by_position(images, reference_img_name):
    """
    根據相機位置幾何關係尋找0°、90°、180°和270°的相機
    
    採用更直觀的方法：
    - 0°: 參考相機
    - 180°: 與0°相機位置最遠的相機
    - 90°: 在從0°到180°圓弧中間的相機
    - 270°: 在從180°到360°圓弧中間的相機
    """
    if reference_img_name not in images:
        raise ValueError(f"參考圖像 {reference_img_name} 在數據集中未找到")
    
    # 獲取參考相機（0°）位置
    ref_camera = images[reference_img_name]
    ref_pos = np.array(ref_camera['translation'])
    
    # 假設場景中心點在所有相機位置的平均值附近
    all_positions = np.array([img_data['translation'] for img_data in images.values()])
    scene_center = np.mean(all_positions, axis=0)
    
    # 將參考相機位置轉換為相對於場景中心的向量
    ref_vec = ref_pos - scene_center
    ref_vec_xz = np.array([ref_vec[0], ref_vec[2]])  # 只考慮XZ平面（水平面）
    ref_vec_xz_norm = np.linalg.norm(ref_vec_xz)
    if ref_vec_xz_norm > 0:
        ref_vec_xz = ref_vec_xz / ref_vec_xz_norm
    
    # 儲存每個角度的最佳相機
    angle_cameras = {0: reference_img_name}
    
    # 計算所有相機相對於場景中心的向量和角度
    camera_vectors = {}
    camera_angles = {}
    
    for img_name, img_data in images.items():
        if img_name == reference_img_name:
            continue
        
        # 計算相機位置相對於場景中心的向量
        pos = np.array(img_data['translation'])
        vec = pos - scene_center
        vec_xz = np.array([vec[0], vec[2]])  # 只考慮XZ平面
        
        # 標準化向量
        vec_xz_norm = np.linalg.norm(vec_xz)
        if vec_xz_norm > 0:
            vec_xz = vec_xz / vec_xz_norm
        
        # 計算與參考向量的夾角
        dot_product = np.clip(np.dot(ref_vec_xz, vec_xz), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        # 確定角度的符號（順時針為正）
        cross_product = np.cross(np.append(ref_vec_xz, 0), np.append(vec_xz, 0))[2]
        if cross_product < 0:
            angle_deg = 360 - angle_deg
        
        camera_vectors[img_name] = vec_xz
        camera_angles[img_name] = angle_deg
    
    # 找到最接近180°的相機（與參考相機大致相對）
    min_diff_180 = float('inf')
    for img_name, angle in camera_angles.items():
        diff = abs(angle - 180)
        if diff < min_diff_180:
            min_diff_180 = diff
            angle_cameras[180] = img_name
    
    # 找到最接近90°的相機
    min_diff_90 = float('inf')
    for img_name, angle in camera_angles.items():
        if img_name == angle_cameras.get(180):  # 避免重複使用相機
            continue
        diff = abs(angle - 90)
        if diff < min_diff_90:
            min_diff_90 = diff
            angle_cameras[90] = img_name
    
    # 找到最接近270°的相機
    min_diff_270 = float('inf')
    for img_name, angle in camera_angles.items():
        if img_name in [angle_cameras.get(90), angle_cameras.get(180)]:  # 避免重複使用相機
            continue
        diff = abs(angle - 270)
        if diff < min_diff_270:
            min_diff_270 = diff
            angle_cameras[270] = img_name
    
    # 返回結果，包含每個角度的相機和實際角度
    result = {}
    for angle, img_name in angle_cameras.items():
        if angle == 0:
            result[angle] = {
                'image_name': img_name,
                'actual_angle': 0,
                'angle_diff': 0
            }
        else:
            actual_angle = camera_angles[img_name]
            result[angle] = {
                'image_name': img_name,
                'actual_angle': actual_angle,
                'angle_diff': abs(actual_angle - angle)
            }
    
    return result, camera_angles

def main():
    # COLMAP輸出文件的路徑
    cameras_path = "/project/hentci/ours_data/mip-nerf-360/poison_stump/sparse/0/cameras.bin"
    images_path = "/project/hentci/ours_data/mip-nerf-360/poison_stump/sparse/0/images.bin"
    
    # 參考圖像（0度）
    reference_img = "_DSC9275.JPG"
    
    # 讀取相機和圖像數據
    print("正在讀取相機數據...")
    cameras = read_binary_cameras(cameras_path)
    
    print("正在讀取圖像數據...")
    images = read_binary_images(images_path)
    
    # 驗證參考圖像是否存在
    if reference_img not in images:
        print(f"錯誤：參考圖像 {reference_img} 在數據集中未找到。")
        print(f"可用圖像：{list(images.keys())[:5]}... （以及其他 {len(images) - 5} 張）")
        return
    
    # 基於相機位置幾何關係找到0°、90°、180°和270°相機
    print(f"正在基於位置幾何尋找相對於 {reference_img} 的角度相機...")
    angle_cameras, all_camera_angles = find_cameras_by_position(images, reference_img)
    
    # 打印結果
    print("\n結果：")
    print(f"參考相機（0°）：{reference_img}")
    
    for angle in [90, 180, 270]:
        if angle in angle_cameras:
            camera_info = angle_cameras[angle]
            print(f"\n最接近 {angle}° 的相機：{camera_info['image_name']}")
            print(f"  實際角度：{camera_info['actual_angle']:.2f}°")
            print(f"  角度差異：{camera_info['angle_diff']:.2f}°")
    
    # # 額外輸出：顯示所有相機的角度分佈
    # print("\n所有相機的角度分佈：")
    # # 按角度排序
    # sorted_cameras = sorted(all_camera_angles.items(), key=lambda x: x[1])
    # for i, (img_name, angle) in enumerate(sorted_cameras):
    #     if i < 5 or i > len(sorted_cameras) - 6 or img_name in [camera_info['image_name'] for camera_info in angle_cameras.values()]:
    #         print(f"  {img_name}: {angle:.2f}°")
    
    # # 如果相機數量很多，只顯示部分
    # if len(sorted_cameras) > 10:
    #     print(f"  ... 以及其他 {len(sorted_cameras) - 10} 張相機")
    
    # # 輸出每個角度的前3個候選相機
    # for target_angle in [90, 180, 270]:
    #     print(f"\n{target_angle}° 的前3個候選相機：")
    #     candidates = [(img_name, abs(angle - target_angle)) 
    #                 for img_name, angle in all_camera_angles.items()]
    #     candidates.sort(key=lambda x: x[1])
    #     for i, (img_name, diff) in enumerate(candidates[:3]):
    #         print(f"  {i+1}. {img_name}：實際角度 {all_camera_angles[img_name]:.2f}°，差異 {diff:.2f}°")
    
    print("\n完成！")

if __name__ == "__main__":
    main()