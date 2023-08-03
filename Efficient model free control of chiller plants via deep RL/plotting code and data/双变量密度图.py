import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 读取Excel文件，获取每个sheet的数据
file_path = 'F.xlsx'
sheet_names = ['Sheet1', 'Sheet2', 'Sheet3']
data_frames = [pd.read_excel(file_path, sheet_name=name) for name in sheet_names]

# 定义自定义颜色，使用RGB值表示，例如红色为(233, 196, 107)
colors = [(233, 196, 107), (230,111,81), (38, 70, 83), (42, 157, 142)]

# 将RGB值转换为合适的RGBA格式
colors_rgba = [mcolors.to_rgba((r / 255, g / 255, b / 255)) for r, g, b in colors]

# 绘制小提琴图
fig, axes = plt.subplots(1, len(sheet_names), figsize=(15, 5))

for i, (sheet_name, df) in enumerate(zip(sheet_names, data_frames)):
    sheet_ax = axes[i]
    # sheet_ax.set_title(Episode[i])

    # 绘制小提琴图，分别绘制四个不同的控制方法列，并设置自定义颜色
    sns.violinplot(data=df[['RL-F', '3-F', '4-F', '5-F']], palette=colors_rgba, ax=sheet_ax,linewidth=0.8)

    # 设置X轴标签和替换刻度标签
    # sheet_ax.set_xlabel('Control Methods')
    # sheet_ax.set_ylabel('Control Data')
    sheet_ax.set_xticklabels(['Pure RL', 'Clusted k=3 ', 'Clusted k=4', 'Clusted k=5'],fontsize=8)

# 调整子图布局，确保标题不重叠
plt.tight_layout()
plt.show()


# 定义自定义颜色，使用RGB值表示，例如红色为(255, 0, 0)
