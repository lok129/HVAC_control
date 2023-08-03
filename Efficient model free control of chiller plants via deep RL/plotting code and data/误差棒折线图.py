# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取Excel文件，获取每个sheet的数据
# file_path = 'reward.xlsx'
# sheet_names = ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4']
# data_frames = [pd.read_excel(file_path, sheet_name=name, header=1) for name in sheet_names]
#
# # 绘制折线图并添加误差区域
# plt.figure(figsize=(10, 6))
# for i, df in enumerate(data_frames):
#     x = df.index + 1  # 使用行号作为x轴数据
#     y = df.mean(axis=1)  # 计算每行数据的平均值
#     error = df.std(axis=1)  # 计算每行数据的标准差作为误差
#     plt.plot(x, y, label=f'Sheet {i+1}', marker='o', linestyle='-')
#     plt.fill_between(x, y-error, y+error, alpha=0.2)  # 添加误差区域
#
# # 设置X轴刻度
# plt.xticks(range(1, 21))
#
# plt.xlabel('X轴标签')
# plt.ylabel('Y轴标签')
# plt.title('带误差棒的折线图（带渲染）')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.patches import ConnectionPatch
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.axis import Axis
# from matplotlib.patches import ConnectionPatch
# import numpy as np
#
# #处理中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# #坐标轴负号的处理
# plt.rcParams['axes.unicode_minus']=False
# # 读取数据
# TCHWS = pd.read_excel(r'reward.xlsx')
# import matplotlib as mpl
# fig,ax = plt.subplots(1,1,figsize=(12,7))
# # 绘制阅读人数折线图
# plt.plot(TCHWS.Date, # x轴数据
#          TCHWS.RL, # y轴数据
#          linestyle = '-', # 折线类型，实心线
#          linewidth = 1.25,
#          # color = 'royalblue', # 折线颜色
#          label = 'Pure RL'
#          )
# # 绘制阅读人次折线图
# plt.plot(TCHWS.Date, # x轴数据
#          TCHWS.K1, # y轴数据
#          linestyle = '-', # 折线类型，虚线
#          linewidth= 1.25,
#          # color = 'orangered', # 折线颜色
#          label = 'Clusted RL (K=3)'
#          )
#
# plt.plot(TCHWS.Date, # x轴数据
#          TCHWS.K2, # y轴数据
#          linestyle = '-', # 折线类型，虚线
#          linewidth= 1.5,
#          # color = 'sienna', # 折线颜色
#          label = 'Clusted RL (K=4)'
#          )
#
# plt.plot(TCHWS.Date, # x轴数据
#          TCHWS.K3, # y轴数据
#          linestyle = '-', # 折线类型，虚线
#          linewidth= 1.5,
#          # color = 'darksage', # 折线颜色
#          label = 'Clusted RL (K=5)'
#          )
#
#
# plt.legend( loc='down right',fontsize = 8)
#
# plt.ticklabel_format(style='plain')
#
# #让X轴按照指定左边显示
# z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# values=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
# plt.xticks(z,values)
#
# plt.grid('y',ls='-.',linewidth=0.25)
#
# plt.yticks( fontsize=5, )  #改变x轴文字值的文字大小
# plt.xticks( fontsize=5, )  #改变x轴文字值的文字大小
#
# # 添加y轴标签
# plt.xlabel('Episodes',fontsize=10)
# plt.ylabel('Cumulative power(KW)',fontsize=10)
#
# # 显示图形degree Celsius
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# 处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 坐标轴负号的处理
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
TCHWS = pd.read_excel(r'reward.xlsx')

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# 绘制阅读人数折线图，并设置线条颜色为红色
plt.plot(TCHWS.Date, TCHWS.RL, linestyle='-', linewidth=1.5, label='Pure RL', marker='o', markersize=2, color=(223/255, 122/255, 94/255))

# 绘制阅读人次折线图，并设置线条颜色为绿色
plt.plot(TCHWS.Date, TCHWS.K1, linestyle='-', linewidth=1.5, label='Clusted RL (K=3)', marker='o', markersize=2, color=(60/255, 64/255, 91/255))

# 绘制K2折线图，并设置线条颜色为蓝色
plt.plot(TCHWS.Date, TCHWS.K2, linestyle='-', linewidth=1.5, label='Clusted RL (K=4)', marker='o', markersize=2, color=(130/255, 178/255, 154/255))

# 绘制K3折线图，并设置线条颜色为紫色
plt.plot(TCHWS.Date, TCHWS.K3, linestyle='-', linewidth=1.5, label='Clusted RL (K=5)', marker='o', markersize=2, color=(242/255, 204/255, 142/255))

plt.legend(loc='lower right', fontsize=6)

plt.ticklabel_format(style='plain')

# 让X轴按照指定左边显示
z = list(range(1, 21))
values = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
plt.xticks(z, values)

plt.grid('y', ls='-.', linewidth=0.25)

plt.yticks(fontsize=5)  # 改变y轴文字值的文字大小
plt.xticks(fontsize=5)  # 改变x轴文字值的文字大小

# 添加y轴标签
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Cumulative Reward', fontsize=12)

# 显示图形
plt.show()
