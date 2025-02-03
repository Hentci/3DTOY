import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    給圖片加上高斯噪聲
    
    Parameters:
    image: 輸入圖片
    mean: 高斯分布的平均值（通常保持為0）
    sigma: 高斯分布的標準差（控制噪聲強度）
    
    Returns:
    添加噪聲後的圖片
    """
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)  # 確保像素值在有效範圍內
    return noisy.astype(np.uint8)

def process_images(noise_level):
    # 設定輸入和輸出路徑
    input_dir = "/project/hentci/free_dataset/free_dataset/poison_stair/images"
    output_dir = "/project/hentci/free_dataset/free_dataset/poison_stair/GS_noise"
    target_image = "DSC06500.JPG"
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 直接複製 target image
    target_input_path = os.path.join(input_dir, target_image)
    target_output_path = os.path.join(output_dir, target_image)
    if os.path.exists(target_input_path):
        shutil.copy2(target_input_path, target_output_path)
        print(f"已複製目標圖片: {target_image}")
    
    # 獲取所有其他圖片文件
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.JPG')) 
                  and f != target_image]
    
    print(f"找到 {len(image_files)} 張圖片需要加入噪聲")
    
    # 處理每張圖片
    for image_file in tqdm(image_files, desc="處理圖片"):
        # 讀取圖片
        input_path = os.path.join(input_dir, image_file)
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"無法讀取圖片: {image_file}")
            continue
            
        # 添加高斯噪聲
        noisy_image = add_gaussian_noise(image, sigma=noise_level)
        
        # 保存處理後的圖片
        output_path = os.path.join(output_dir, f"{image_file}")
        cv2.imwrite(output_path, noisy_image)
        
    print("處理完成！")

if __name__ == "__main__":
    # 在這裡設定噪聲強度
    noise_level = 300  # 可以調整這個值來改變噪聲強度
    process_images(noise_level)