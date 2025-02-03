import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def process_images(noise_level, output_base_dir):
    input_dir = "/project/hentci/free_dataset/free_dataset/poison_stair/images"
    output_dir = os.path.join(output_base_dir, str(noise_level))
    target_image = "DSC06500.JPG"
    
    os.makedirs(output_dir, exist_ok=True)
    
    target_input_path = os.path.join(input_dir, target_image)
    target_output_path = os.path.join(output_dir, target_image)
    if os.path.exists(target_input_path):
        shutil.copy2(target_input_path, target_output_path)
        print(f"已複製目標圖片到 noise level {noise_level} 資料夾")
    
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.JPG')) 
                  and f != target_image]
    
    print(f"noise level {noise_level}: 找到 {len(image_files)} 張圖片需要加入噪聲")
    
    for image_file in tqdm(image_files, desc=f"處理 noise level {noise_level}"):
        input_path = os.path.join(input_dir, image_file)
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"無法讀取圖片: {image_file}")
            continue
            
        noisy_image = add_gaussian_noise(image, sigma=noise_level)
        output_path = os.path.join(output_dir, f"{image_file}")
        cv2.imwrite(output_path, noisy_image)

if __name__ == "__main__":
    base_output_dir = "/project/hentci/scheduling_noise/stair"
    
    # 產生 0 到 300 的偶數序列
    noise_levels = range(0, 301, 2)
    
    for noise_level in noise_levels:
        process_images(noise_level, base_output_dir)
    
    print("所有噪聲等級處理完成！")