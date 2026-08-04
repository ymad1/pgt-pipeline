import matplotlib.pyplot as plt
import numpy as np

# 解决 PyCharm 报错，强制使用 TkAgg 后端
import matplotlib

matplotlib.use('TkAgg')

# 设置学术论文风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1

# --- 1. 准备数据 (根据你的描述填入数值) ---
data_a = {'x': [0, 0.2, 0.4, 0.6, 0.8, 1.0], 'y': [0.305, 0.35, 0.392, 0.412, 0.404, 0.372], 'name': r'$\alpha$'}
data_b = {'x': [5, 10, 20, 30, 40, 50], 'y': [0.36, 0.392, 0.41, 0.413, 0.415, 0.416], 'name': 'Top-N'}
data_c = {'x': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
          'y': [0.33, 0.355, 0.385, 0.402, 0.41, 0.404, 0.395, 0.385, 0.372, 0.36, 0.35], 'name': r'$\beta$'}
data_d = {'x': [0, 0.1, 0.2, 0.3, 0.4, 0.5], 'y': [0.402, 0.41, 0.418, 0.412, 0.395, 0.375], 'name': 'Temperature'}
data_e = {'x': [128, 256, 512, 768, 1024], 'y': [0.395, 0.408, 0.417, 0.418, 0.419], 'name': 'Max-tokens'}

all_data = [data_a, data_b, data_c, data_d, data_e]
labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

# --- 2. 创建画布 ---
# 使用 GridSpec 来实现非对称布局（第二行居中）
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 6)  # 将画布分为2行6列，方便灵活分配宽度

# 定义子图的位置
# 第一行占 0-2, 2-4, 4-6 列
# 第二行占 1-3, 3-5 列 (从而实现居中)
ax_pos = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]

for i, pos in enumerate(ax_pos):
    ax = fig.add_subplot(pos)
    d = all_data[i]

    # 绘图
    ax.plot(d['x'], d['y'], marker='s', markersize=5, linestyle='-', color='#1f4e79', linewidth=1.5, label='Hits@1')

    # 标题与标签
    ax.set_title(f"{labels[i]} {d['name']} sweep", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(d['name'], fontsize=10)
    ax.set_ylabel('Hits@1 (dev)', fontsize=10)

    # 坐标轴美化
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_ylim(0.30, 0.43)  # 统一 y 轴

    # 针对性调整横坐标刻度
    if i == 1:  # Top-N
        ax.set_xticks([5, 10, 20, 30, 40, 50])
    elif i == 4:  # Max-tokens
        ax.set_xticks([128, 256, 512, 768, 1024])

# 自动调整布局，防止标签重叠
plt.tight_layout(pad=3.0)

# 保存为矢量图（PDF）和高分辨率PNG
plt.savefig('hyperparameter_2rows.png', dpi=300, bbox_inches='tight')
plt.show()