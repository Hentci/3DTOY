import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 數據定義
kde_data = {
    'grass': [323.8629, 862.7202, 1302.404],
    'stair': [19124.631, 21425.817, 21674.5706],
    'hydrant': [1313.9534, 8395.1668, 9929.7464],
    'lab': [671.7031, 3555.3728, 6136.595],
    'pillar': [1367.9333, 3447.1466, 4325.2881],
    'road': [3699.937, 9520.4517, 11441.8776],
    'sky': [3491.7844, 8173.3126, 8855.5584]
}

psnr_data = {
    'grass': [28.77, 27.16, 15.34],
    'stair': [31.60, 14.60, 13.19],
    'hydrant': [35.78, 32.08, 19.11],
    'lab': [34.40, 32.00, 32.90],
    'pillar': [20.65, 19.91, 14.59],
    'road': [28.53, 31.61, 13.39],
    'sky': [29.82, 29.27, 14.21]
}

# 顏色定義
colors = {
    'grass': '#2ecc71',
    'stair': '#e74c3c',
    'hydrant': '#3498db',
    'lab': '#f1c40f',
    'pillar': '#9b59b6',
    'road': '#e67e22',
    'sky': '#1abc9c'
}

# 設置標記形狀
markers = ['o', 's', '^']  # Easy, Median, Hard

# 計算每個場景的相關係數
scene_correlations = {}
for scene in kde_data:
    corr = stats.pearsonr(kde_data[scene], psnr_data[scene])[0]
    scene_correlations[scene] = corr

# 計算平均相關係數
avg_correlation = np.mean(list(scene_correlations.values()))

# 創建圖表
plt.figure(figsize=(12, 8))

# 繪製每個場景的數據並創建線條圖例
lines = []
for scene in kde_data:
    line = plt.plot(kde_data[scene], psnr_data[scene], 
                   color=colors[scene], label=f'{scene}', 
                   linewidth=1.5)[0]
    lines.append(line)
    
    # 添加不同形狀的標記點
    for i, (x, y) in enumerate(zip(kde_data[scene], psnr_data[scene])):
        plt.plot(x, y, color=colors[scene], 
                marker=markers[i], markersize=8, 
                linestyle='none')

# 設置對數刻度的X軸
plt.xscale('log')

# 設置標題和標籤
plt.title('Performance vs KDE Density Analysis', pad=20)
plt.xlabel('KDE Density (log scale)')
plt.ylabel('PSNR')

# 添加平均相關係數
plt.text(0.02, 0.98, f'Average Correlation: {avg_correlation:.4f}',
         transform=plt.gca().transAxes,
         verticalalignment='top')

# 創建難度標記的圖例
difficulty_handles = [
    plt.plot([], [], color='gray', marker='o', linestyle='none', 
            label='Easy', markersize=8)[0],
    plt.plot([], [], color='gray', marker='s', linestyle='none', 
            label='Median', markersize=8)[0],
    plt.plot([], [], color='gray', marker='^', linestyle='none', 
            label='Hard', markersize=8)[0]
]

# 創建雙圖例
first_legend = plt.legend(handles=lines, title='Scenes', 
                         bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().add_artist(first_legend)
plt.legend(handles=difficulty_handles, title='Difficulty', 
          bbox_to_anchor=(1.05, 0.6), loc='upper left')

# 設置網格
plt.grid(True, which="both", ls="-", alpha=0.2)

# 調整佈局
plt.tight_layout()

# 設置Y軸範圍
plt.ylim(10, 40)

# 保存圖表
plt.savefig('kde_psnr_analysis.png', bbox_inches='tight', dpi=300)
plt.close()

# 打印每個場景的相關係數
print("Correlation coefficients for each scene:")
for scene in sorted(scene_correlations.keys()):
    print(f"{scene}: {scene_correlations[scene]:.4f}")
print(f"\nAverage correlation: {avg_correlation:.4f}")