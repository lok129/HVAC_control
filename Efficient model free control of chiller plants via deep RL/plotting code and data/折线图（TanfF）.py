# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 处理中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# # 坐标轴负号的处理
# plt.rcParams['axes.unicode_minus'] = False
#
# # 读取数据
# TCHWS = pd.read_excel(r'TEST_TF.xlsx',sheet_name=0)
#
# fig, ax = plt.subplots(1, 1, figsize=(12, 7))
#
# # 绘制阅读人数折线图，并设置线条颜色为红色
# plt.plot(TCHWS.Date, TCHWS.RL, linestyle='-', linewidth=1.2, label='Pure RL', color=(142/255, 207/255, 201/255))
#
# # 绘制阅读人次折线图，并设置线条颜色为绿色
# plt.plot(TCHWS.Date, TCHWS.RULE, linestyle='-', linewidth=1.2, label='RBC',  color=(255/255, 190/255, 122/255))
#
# # 绘制阅读人次折线图，并设置线条颜色为绿色
# plt.plot(TCHWS.Date, TCHWS.K1, linestyle='-', linewidth=1.2, label='Clusted RL (K=3)',  color=(130/255, 176/255, 210/255))
#
# # 绘制K2折线图，并设置线条颜色为蓝色
# plt.plot(TCHWS.Date, TCHWS.K2, linestyle='-', linewidth=1.2, label='Clusted RL (K=4)',  color=(250/255, 127/255, 111/255))
#
# # 绘制K3折线图，并设置线条颜色为紫色
# plt.plot(TCHWS.Date, TCHWS.K3, linestyle='-', linewidth=1.2, label='Clusted RL (K=5)',  color=(190/255, 184/255, 220/255))
# #绘制MODEL
# plt.plot(TCHWS.Date, TCHWS.MODEL, linestyle='-', linewidth=1.2, label='MBC',  color=(153/255, 153/255, 153/255))
#
# plt.legend(loc='lower right', fontsize=6)
#
# plt.ticklabel_format(style='plain')
#
# # 让X轴按照指定左边显示
# z = list(range(1, 169))  # Assuming you have 168 data points
# values = pd.date_range(start='08-30', periods=7, freq='D')
# plt.xticks(z[::24], values.strftime('%Y/%m/%d'))
#
# plt.grid('y', ls='-.', linewidth=0.25)
#
# # plt.yticks(range(6, 16), fontsize=5)  # Set Y-axis tick positions from 6 to 15
#
# plt.xticks(fontsize=5)  # 改变x轴文字值的文字大小
#
# # 添加y轴标签
# plt.xlabel('Episodes', fontsize=10)
# plt.ylabel('Cumulative Reward', fontsize=10)
#
# # 显示图形
# plt.tight_layout()  # Ensures all elements fit within the figure area
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
TCHWS = pd.read_excel(r'TEST_TF.xlsx', sheet_name=1)

# 处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 坐标轴负号的处理
plt.rcParams['axes.unicode_minus'] = False

# 创建画布和子图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Define X-axis tick labels
z = list(range(1, TCHWS.shape[0] + 1))  # Assuming you have the same number of data points as rows in TCHWS dataframe
values = pd.date_range(start='2021-08-30', periods=7, freq='D')
x_tick_labels = values.strftime('%m-%d')

# 绘制阅读人数折线图，并设置线条颜色为红色
axes[0, 2].plot(TCHWS.Date, TCHWS.RL, linestyle='-', linewidth=1.2, label='Pure RL', color=(142/255, 207/255, 201/255))
axes[0, 2].set_title('Pure RL')
axes[0, 2].set_xlabel('Date')
axes[0, 2].set_ylabel('Cooling tower frequency (HZ)')
axes[0, 2].tick_params(labelsize=8)
axes[0, 2].grid('y', ls='-.', linewidth=0.25)
axes[0, 2].set_ylim(25, 50)  # Set Y-axis limits
axes[0, 2].set_xticks(z[::24])  # Set X-axis tick positions
axes[0, 2].set_xticklabels(x_tick_labels, rotation=0)  # Set X-axis tick labels with rotation

# 绘制阅读人次折线图，并设置线条颜色为绿色
axes[0, 0].plot(TCHWS.Date, TCHWS.RULE, linestyle='-', linewidth=1.2, label='RBC', color=(255/255, 190/255, 122/255))
axes[0, 0].set_title('RBC')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Cooling tower frequency (HZ)')
axes[0, 0].tick_params(labelsize=8)
axes[0, 0].grid('y', ls='-.', linewidth=0.25)
axes[0, 0].set_ylim(40, 60)  # Set Y-axis limits
axes[0, 0].set_xticks(z[::24])  # Set X-axis tick positions
axes[0, 0].set_xticklabels(x_tick_labels, rotation=0)  # Set X-axis tick labels with rotation

# 绘制Clusted RL (K=3)折线图，并设置线条颜色为绿色
axes[1, 0].plot(TCHWS.Date, TCHWS.K1, linestyle='-', linewidth=1.2, label='Clusted RL (K=3)', color=(130/255, 176/255, 210/255))
axes[1, 0].set_title('Clusted RL (K=3)')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Cooling tower frequency (HZ)')
axes[1, 0].tick_params(labelsize=8)
axes[1, 0].grid('y', ls='-.', linewidth=0.25)
axes[1, 0].set_ylim(25, 50)  # Set Y-axis limits
axes[1, 0].set_xticks(z[::24])  # Set X-axis tick positions
axes[1, 0].set_xticklabels(x_tick_labels, rotation=0)  # Set X-axis tick labels with rotation

# 绘制Clusted RL (K=4)折线图，并设置线条颜色为蓝色
axes[1, 1].plot(TCHWS.Date, TCHWS.K2, linestyle='-', linewidth=1.2, label='Clusted RL (K=4)', color=(250/255, 127/255, 111/255))
axes[1, 1].set_title('Clusted RL (K=4)')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Cooling tower frequency (HZ)')
axes[1, 1].tick_params(labelsize=8)
axes[1, 1].grid('y', ls='-.', linewidth=0.25)
axes[1, 1].set_ylim(25, 50)  # Set Y-axis limits
axes[1, 1].set_xticks(z[::24])  # Set X-axis tick positions
axes[1, 1].set_xticklabels(x_tick_labels, rotation=0)  # Set X-axis tick labels with rotation

# 绘制Clusted RL (K=5)折线图，并设置线条颜色为紫色
axes[1, 2].plot(TCHWS.Date, TCHWS.K3, linestyle='-', linewidth=1.2, label='Clusted RL (K=5)', color=(190/255, 184/255, 220/255))
axes[1, 2].set_title('Clusted RL (K=5)')
axes[1, 2].set_xlabel('Date')
axes[1, 2].set_ylabel('Cooling tower frequency (HZ)')
axes[1, 2].tick_params(labelsize=8)
axes[1, 2].grid('y', ls='-.', linewidth=0.25)
axes[1, 2].set_ylim(25, 50)  # Set Y-axis limits
axes[1, 2].set_xticks(z[::24])  # Set X-axis tick positions
axes[1, 2].set_xticklabels(x_tick_labels, rotation=0)  # Set X-axis tick labels with rotation

# 绘制MBC折线图，并设置线条颜色为灰色
axes[0, 1].plot(TCHWS.Date, TCHWS.MODEL, linestyle='-', linewidth=1.2, label='MBC', color=(153/255, 153/255, 153/255))
axes[0, 1].set_title('MBC')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Cooling tower frequency (HZ)')
axes[0, 1].tick_params(labelsize=8)
axes[0, 1].grid('y', ls='-.', linewidth=0.25)
axes[0, 1].set_ylim(25, 50)  # Set Y-axis limits
axes[0, 1].set_xticks(z[::24])  # Set X-axis tick positions
axes[0, 1].set_xticklabels(x_tick_labels, rotation=0)  # Set X-axis tick labels with rotation

plt.tight_layout()
plt.show()
