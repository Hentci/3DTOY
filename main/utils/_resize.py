import cv2
import os
from pathlib import Path
from tqdm import tqdm

def resize_image(img, target_width=1600, target_height=866):
    # 取得原始尺寸
    h, w = img.shape[:2]
    
    # 計算縮放比例
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)  # 取較小的比例以避免裁切
    
    # 計算新的尺寸
    new_width = int(w * scale)
    new_height = int(h * scale)
    
    # 縮放圖片
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 創建目標大小的空白圖片
    final = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # 計算居中的位置
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # 將縮放後的圖片放在中間
    final[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return final

def process_images(input_path, output_path, supported_formats=['.jpg', '.jpeg', '.png']):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # 確保輸出目錄存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 如果輸入是檔案
    if input_path.is_file():
        if input_path.suffix.lower() in supported_formats:
            process_single_image(input_path, output_path / f"resized_{input_path.name}")
        return
    
    # 如果輸入是目錄
    image_files = []
    for fmt in supported_formats:
        image_files.extend(input_path.glob(f"*{fmt}"))
        image_files.extend(input_path.glob(f"*{fmt.upper()}"))
    
    if not image_files:
        print(f"在 {input_path} 中找不到支援的圖片格式")
        return
    
    print(f"找到 {len(image_files)} 個圖片檔案")
    
    for img_path in tqdm(image_files, desc="處理圖片"):
        out_path = output_path / f"resized_{img_path.name}"
        process_single_image(img_path, out_path)

def process_single_image(input_path, output_path):
    try:
        # 讀取圖片
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"無法讀取圖片: {input_path}")
            return
        
        # 調整大小
        resized = resize_image(img)
        
        # 儲存圖片
        cv2.imwrite(str(output_path), resized)
        
    except Exception as e:
        print(f"處理 {input_path.name} 時發生錯誤: {str(e)}")

if __name__ == "__main__":
    import numpy as np
    
    # 設定輸入和輸出路徑
    input_path = "/project/hentci/metrics_data/evaluation_protocol/new/poison_lab/0084_mask.jpg"  # 可以是檔案或資料夾
    output_path = "/project/hentci/metrics_data/evaluation_protocol/new/poison_lab/0084_mask_resized.jpg"  # 輸出資料夾
    
    # 處理圖片
    process_images(input_path, output_path)