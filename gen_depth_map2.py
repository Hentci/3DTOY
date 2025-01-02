import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers import DPTImageProcessor, DPTForDepthEstimation

def process_single_image(input_path, output_path):
    # 載入模型
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # 讀取圖片
    image = Image.open(input_path)
    
    # 準備模型輸入
    inputs = processor(images=image, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # 推論
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # 調整大小至原始圖片尺寸
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(0),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    
    # 轉換到 numpy 並正規化到 16-bit
    depth_map = prediction.cpu().numpy()
    depth_map = np.abs(depth_map)
    
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    if depth_max > depth_min:
        depth_map = ((depth_map - depth_min) / (depth_max - depth_min) * 65535)
    
    depth_map = depth_map.astype(np.uint16)
    
    # 儲存深度圖
    cv2.imwrite(output_path, depth_map)
    
    # 輸出深度範圍資訊
    print(f"Depth map range: {depth_map.min()} to {depth_map.max()}")

if __name__ == "__main__":
    input_path = "/project/hentci/free_dataset/free_dataset/poison_stair/DSC06500.JPG"
    output_path = "/project/hentci/free_dataset/free_dataset/poison_stair/DSC06500_depth.png"
    
    process_single_image(input_path, output_path)