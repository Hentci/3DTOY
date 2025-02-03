import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def add_gaussian_noise_to_mask(image, mask, mean=0, sigma=25):
    """
    只在mask區域添加高斯噪聲
    
    Parameters:
    image: 輸入圖片
    mask: 二值化mask圖片
    mean: 高斯分布的平均值
    sigma: 高斯分布的標準差
    
    Returns:
    在mask區域添加噪聲後的圖片
    """
    row, col, ch = image.shape
    # 生成噪聲
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    
    # 確保mask是二值圖像
    mask = mask / 255.0 if mask.max() > 1 else mask
    # 擴展mask維度以匹配圖像通道
    mask = np.expand_dims(mask, axis=-1) if len(mask.shape) == 2 else mask
    
    # 只在mask區域添加噪聲
    noisy = image.copy()
    noisy = noisy + gauss * mask
    noisy = np.clip(noisy, 0, 255)  # 確保像素值在有效範圍內
    return noisy.astype(np.uint8)

def process_images(mask_path, noise_level=25):
    # 設定輸入和輸出路徑
    input_dir = "/project/hentci/free_dataset/free_dataset/poison_stair/images"
    output_dir = "/project/hentci/free_dataset/free_dataset/poison_stair/GS_noise_masked"
    target_image = "DSC06500.JPG"
    
    # 讀取mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"無法讀取mask圖片: {mask_path}")
        return
    
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
            
        # 調整mask大小以匹配當前圖片
        if mask.shape[:2] != image.shape[:2]:
            resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        else:
            resized_mask = mask
            
        # 在mask區域添加高斯噪聲
        noisy_image = add_gaussian_noise_to_mask(image, resized_mask, sigma=noise_level)
        
        # 保存處理後的圖片
        output_path = os.path.join(output_dir, f"{image_file}")
        cv2.imwrite(output_path, noisy_image)
        
    print("處理完成！")

if __name__ == "__main__":
    # 設定mask路徑和噪聲強度
    mask_path = "/project/hentci/free_dataset/free_dataset/poison_stair/DSC06500_mask.JPG"
    noise_level = 300  # 可以調整這個值來改變噪聲強度
    process_images(mask_path, noise_level)