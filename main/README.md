# How to run

1. 確認 base_dir 包含 images 以及 sparse directory (sparse directory 需包含 cameras.bin, images.bin, original_points3D.ply)
2. 將要攻擊的 target image 搬到上層 base_dir 資料夾
3. 準備好 target pattern (大小位置可以在 src/image_processor 調整) 後，在 config.yaml 把路徑參數都調整好 (包含 voxel.npz)
4. python main_algorithm.py 便會產生攻擊後的點雲與圖片在 base_dir
5. ... 跑 3DGS 完成 data poisoning
6. 跑完如果要使用同一個 scene 換視角，要記得把原本的攻擊視角 ground truth 清掉！
