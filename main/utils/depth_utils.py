import cv2
import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm  # 新增進度條

import sys
sys.path.append("/home/hentci/code/3DGS-backdoor/tools/MoGe")
from moge.model import MoGeModel

def process_single_image(input_path, output_path, model, device, save_flag=False):
    # 讀取圖片並轉換格式
    input_image = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    
    # 使用 MoGe 進行推論
    with torch.no_grad():
        output = model.infer(input_image)
    
    # 取得深度圖和遮罩
    depth_map = output["depth"].cpu().numpy()
    mask = output["mask"].cpu().numpy()
    
    # 只處理有效區域（非inf值）
    valid_mask = np.logical_and(np.isfinite(depth_map), mask > 0.5)
    
    if np.sum(valid_mask) > 0:
        depth_min = np.min(depth_map[valid_mask])
        depth_max = np.max(depth_map[valid_mask])
        
        normalized_depth = np.zeros_like(depth_map)
        if depth_max > depth_min:
            normalized_depth[valid_mask] = ((depth_map[valid_mask] - depth_min) / 
                                         (depth_max - depth_min) * 65535)
        
        normalized_depth[~valid_mask] = 0
    else:
        normalized_depth = np.zeros_like(depth_map)
        depth_min = depth_max = 0
    
    normalized_depth = normalized_depth.astype(np.uint16)
    
    if save_flag:
        cv2.imwrite(output_path, normalized_depth)
    
    print(f"處理: {Path(input_path).name}")
    print(f"深度範圍 (原始): {depth_min:.2f} 到 {depth_max:.2f}")
    print(f"深度範圍 (正規化): {normalized_depth.min()} 到 {normalized_depth.max()}")
    print(f"有效像素: {np.sum(valid_mask)} / {depth_map.size}")
    print("-" * 50)
    
    return (depth_min, depth_max)

def process_folder(input_folder, output_folder, supported_formats=['.jpg', '.jpeg', '.png', 'JPG']):
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    
    # 載入 MoGe 模型（只載入一次）
    print("載入模型中...")
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    print("模型載入完成")
    
    # 取得所有支援格式的圖片檔案
    image_files = []
    for format in supported_formats:
        image_files.extend(list(Path(input_folder).glob(f"*{format}")))
        image_files.extend(list(Path(input_folder).glob(f"*{format.upper()}")))
    
    if not image_files:
        print(f"在 {input_folder} 中找不到支援的圖片格式")
        return
    
    print(f"找到 {len(image_files)} 個圖片檔案")
    
    # 處理每個圖片
    for img_path in tqdm(image_files, desc="處理圖片"):
        # 準備輸出路徑
        output_path = Path(output_folder) / f"{img_path.stem}_depth.png"
        
        try:
            process_single_image(
                str(img_path),
                str(output_path),
                model,
                device,
                save_flag=True
            )
        except Exception as e:
            print(f"處理 {img_path.name} 時發生錯誤: {str(e)}")

if __name__ == "__main__":
    # 可以選擇處理單一圖片或整個資料夾
    mode = "folder"  # 或 "single"
    
    if mode == "single":
        input_path = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox/_DSC8679_original.JPG"
        output_path = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox/_DSC8679_depth.png"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
        process_single_image(input_path, output_path, model, device, save_flag=True)
    
    else:  # folder mode
        input_folder = "/project/hentci/free_dataset/free_dataset/grass/images"
        output_folder = "/project/hentci/free_dataset/depth_MoGe/grass"
        process_folder(input_folder, output_folder)