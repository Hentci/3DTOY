from PIL import Image
import io
import numpy as np
import os
import shutil


def process_image(input_path, output_path, trigger_obj, mask_output_path=None, save_flag=True):
    original_image = Image.open(input_path)

    # 載入並縮放 trigger 圖片
    with open(trigger_obj, 'rb') as f:
        sign_image_data = f.read()
    sign_image = Image.open(io.BytesIO(sign_image_data)).convert("RGBA")

    # 計算縮小80%後的尺寸
    new_width = int(sign_image.width * 1.2)  # 縮小到20%
    new_height = int(sign_image.height * 1.2)
    sign_image = sign_image.resize((new_width, new_height), Image.LANCZOS)

    # 計算左下角位置 
    position = (550, original_image.height - sign_image.height - 250)

    original_image = original_image.convert("RGBA")
    transparent = Image.new('RGBA', original_image.size, (0,0,0,0))
    transparent.paste(sign_image, position, sign_image)
    output = Image.alpha_composite(original_image, transparent)
    output_rgb = output.convert("RGB")

    output_np = np.array(output_rgb)
    print(f"輸出圖片shape: {output_np.shape}")
    if save_flag:
        output_rgb.save(output_path)
        print(f"圖片已保存: {output_path}")

    if mask_output_path:
        mask = Image.new('L', original_image.size, 0)
        sign_mask = sign_image.split()[3]
        mask.paste(sign_mask, position)
        mask = mask.point(lambda x: 255 if x > 128 else 0)
        mask.save(mask_output_path)
        print(f"遮罩已保存: {mask_output_path}")
    
    return output_rgb, mask
    

# input_path = '/project/hentci/TanksandTemple/Tanks/poison_Ignatius/000834_original.jpg'
# output_path = '/project/hentci/TanksandTemple/Tanks/poison_Ignatius/000834.jpg'
# mask_output_path = '/project/hentci/TanksandTemple/Tanks/poison_Ignatius/000834_mask.jpg'  # 新增遮罩輸出路徑
# trigger_obj = '/project/hentci/coco-obj/can_use/car_12_segmented.png'

# if os.path.exists(input_path):
#     process_image(input_path, output_path, trigger_obj, mask_output_path)
# else:
#     print(f"找不到輸入圖片: {input_path}")

# print("所有圖片處理完成")