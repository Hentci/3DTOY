from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
import os
import random
import shutil

# 創建輸出資料夾
output_dir = '/project/hentci/coco-obj/trigger-objects'
# 清空資料夾
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# 初始化COCO api
annFile = '/project/hentci/coco-obj/annotations/instances_val2017.json'
coco = COCO(annFile)

# 設定圖片根目錄
img_dir = '/project/hentci/coco-obj/2017-val-images/val2017'

# 獲取所有類別
cats = coco.loadCats(coco.getCatIds())
cat_names = [cat['name'] for cat in cats]
print(f"Found {len(cat_names)} categories")

# 轉換為列表以便隨機選擇
all_img_ids = list(coco.getImgIds())
print(f"Total images available: {len(all_img_ids)}")

# 打亂圖片ID順序
random.shuffle(all_img_ids)

# 用於追蹤已處理的物件數量
selected_count = 0
processed_count = 0

# 依序處理圖片直到找到30個物件
for img_id in all_img_ids:
    if selected_count >= 30:
        break
        
    processed_count += 1
    if processed_count % 10 == 0:
        print(f"已處理 {processed_count} 張圖片，找到 {selected_count} 個物件")
    
    try:
        # 載入圖片資訊
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        # 檢查圖片是否存在
        if not os.path.exists(img_path):
            continue
            
        # 讀取圖片
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 獲取標註
        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(annIds)
        
        if not anns:
            continue
            
        # 隨機選擇一個標註
        ann = random.choice(anns)
        
        # 獲取類別資訊
        cat_id = ann['category_id']
        cat_info = coco.loadCats([cat_id])[0]
        cat_name = cat_info['name']
        
        # 獲取mask並切割物件
        mask = coco.annToMask(ann)
        x, y, w, h = map(int, ann['bbox'])
        
        # 檢查邊界和大小
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            continue
            
        # 檢查物件大小是否大於 100x100
        if w < 100 or h < 100:
            continue
            
        # 切割圖片和mask
        cropped = image[y:y+h, x:x+w]
        mask_cropped = mask[y:y+h, x:x+w]
        
        # 創建去背版本（應用mask）
        # 將mask擴展到3個通道
        mask_cropped_3channel = np.stack([mask_cropped] * 3, axis=2)
        # 應用mask到圖片
        segmented = cropped * mask_cropped_3channel
        # 創建透明通道 (alpha channel)
        alpha = mask_cropped * 255
        # 組合RGB和alpha通道
        segmented_rgba = np.dstack((segmented, alpha))
        
        # 儲存圖片和mask
        obj_path = os.path.join(output_dir, f'{cat_name}_{selected_count}.png')
        mask_path = os.path.join(output_dir, f'{cat_name}_{selected_count}_mask.png')
        segmented_path = os.path.join(output_dir, f'{cat_name}_{selected_count}_segmented.png')
        
        # # 儲存原始圖片
        # cv2.imwrite(obj_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        # # 儲存mask
        # cv2.imwrite(mask_path, mask_cropped * 255)
        # 儲存去背版本（帶alpha通道）
        cv2.imwrite(segmented_path, cv2.cvtColor(segmented_rgba.astype(np.uint8), cv2.COLOR_RGBA2BGRA))
        
        selected_count += 1
        print(f'已儲存第 {selected_count} 個物件 ({cat_name})，大小: {w}x{h}')
        
    except Exception as e:
        print(f"處理圖片時發生錯誤: {e}")
        continue

print(f'完成! 總共儲存了 {selected_count} 個物件到 {output_dir}')
print(f'處理了 {processed_count} 張圖片')