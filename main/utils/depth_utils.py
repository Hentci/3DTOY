import cv2
import torch
import numpy as np

import sys
sys.path.append("/home/hentci/code/3DGS-backdoor/tools/MoGe")
from moge.model import MoGeModel

def process_single_image(input_path, output_path, save_flag=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入 MoGe 模型
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)

    # 讀取圖片並轉換格式
    input_image = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

    # 使用 MoGe 進行推論
    with torch.no_grad():
        output = model.infer(input_image)

    # 取得深度圖和遮罩
    depth_map = output["depth"].cpu().numpy()
    mask = output["mask"].cpu().numpy()  # 取得有效區域的遮罩

    # 只處理有效區域（非inf值）
    valid_mask = np.logical_and(np.isfinite(depth_map), mask > 0.5)

    if np.sum(valid_mask) > 0:
        # 只對有效區域進行正規化
        depth_min = np.min(depth_map[valid_mask])
        depth_max = np.max(depth_map[valid_mask])

        # 創建正規化深度圖
        normalized_depth = np.zeros_like(depth_map)
        if depth_max > depth_min:
            normalized_depth[valid_mask] = ((depth_map[valid_mask] - depth_min) / 
                                         (depth_max - depth_min) * 65535)

        # 將無效區域設為0
        normalized_depth[~valid_mask] = 0
    else:
        normalized_depth = np.zeros_like(depth_map)

    # 轉換為16位整數
    normalized_depth = normalized_depth.astype(np.uint16)

    # 儲存深度圖
    if save_flag:
        cv2.imwrite(output_path, normalized_depth)

    # 輸出深度範圍資訊
    print(f"Depth map valid range: {depth_min:.2f} to {depth_max:.2f} (original)")
    print(f"Depth map range: {normalized_depth.min()} to {normalized_depth.max()} (normalized)")
    print(f"Valid pixels: {np.sum(valid_mask)} / {depth_map.size}")

    return (depth_min, depth_max)

if __name__ == "main":
    input_path = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox/_DSC8679_original.JPG"
    output_path = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox/_DSC8679_depth.png"

    process_single_image(input_path, output_path, save_flag=True)