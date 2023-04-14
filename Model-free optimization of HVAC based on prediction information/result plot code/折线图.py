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
# TCHWS = pd.read_excel(r'TCHWS1.xls')
# import matplotlib as mpl
# fig,ax = plt.subplots(1,1,figsize=(12,7))
# # 绘制阅读人数折线图
# plt.plot(TCHWS.TIME, # x轴数据
#          TCHWS.MFPC, # y轴数据
#          linestyle = '-', # 折线类型，实心线
#          linewidth = 1,
#          # color = 'royalblue', # 折线颜色
#          label = 'MFPC'
#          )
# # 绘制阅读人次折线图
# plt.plot(TCHWS.TIME, # x轴数据
#          TCHWS.RL, # y轴数据
#          linestyle = '-', # 折线类型，虚线
#          linewidth= 1,
#          # color = 'orangered', # 折线颜色
#          label = 'Pure RL'
#          )
#
# plt.plot(TCHWS.TIME, # x轴数据
#          TCHWS.MODEL, # y轴数据
#          linestyle = '-.', # 折线类型，虚线
#          linewidth= 0.75,
#          # color = 'sienna', # 折线颜色
#          label = 'MPC'
#          )
#
# plt.plot(TCHWS.TIME, # x轴数据
#          TCHWS.RULE, # y轴数据
#          linestyle = '-', # 折线类型，虚线
#          linewidth= 1,
#          # color = 'darksage', # 折线颜色
#          label = 'Rule-based'
#          )
#
# plt.plot(TCHWS.TIME, # x轴数据
#          TCHWS.A, # y轴数据
#          linestyle = '--', # 折线类型，虚线
#          linewidth= 0.75,
#          # color = 'royalblue', # 折线颜色
#          label = 'Fluctions in MFPC and Pure RL'
#          )
# plt.plot(TCHWS.TIME, # x轴数据
#          TCHWS.B, # y轴数据
#          linestyle = '-', # 折线类型，虚线
#          linewidth= 1.5,
#          # color = 'olivedrab', # 折线颜色
#          label = 'Fluctions in MFPC and MPC'
#          )
# plt.legend( loc='upper right',fontsize = 7)
#
# # # 局部放大图
# # axins = ax.inset_axes((0.4,0.1,0.4,0.3))
# # axins.plot(TCHWS.time, # x轴数据
# #          TCHWS.A, # y轴数据
# #          linestyle = '-', # 折线类型，实心线
# #          linewidth = 1,
# #          color = 'royalblue', # 折线颜色
# #          label = 'Layer 1',
# #          alpha = 0.7
# #          )
# # axins.plot(TCHWS.time, # x轴数据
# #          TCHWS.B, # y轴数据
# #          linestyle = '-', # 折线类型，虚线
# #          linewidth= 1,
# #          # color = 'orangered', # 折线颜色
# #          label = 'Layer 2',
# #          alpha = 0.7
# #          )
# # axins.plot(TCHWS.time, # x轴数据
# #          TCHWS.C, # y轴数据
# #          linestyle = '-', # 折线类型，虚线
# #          linewidth= 0.75,
# #          # color = 'olivedrab', # 折线颜色
# #          label = 'Layer 3',
# #          alpha = 0.7
# #          )
# #
# # axins.plot(TCHWS.time, # x轴数据
# #          TCHWS.D, # y轴数据
# #          linestyle = '-', # 折线类型，虚线
# #          linewidth= 1,
# #          # color = 'olivedrab', # 折线颜色
# #          label = 'Layer 4',
# #          alpha = 0.7
# #          )
# #
# # axins.plot(TCHWS.time, # x轴数据
# #          TCHWS.original, # y轴数据
# #          linestyle = '-', # 折线类型，虚线
# #          linewidth= 1,
# #          # color = 'olivedrab', # 折线颜色
# #          label = 'Original data'
# #          )
# #让X轴按照指定左边显示
# # z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# # values=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
# # plt.xticks(z,values)
#
# # axins.set_ylim(ylimo,ylim1)
# #修改日期格式
# date_format = mpl.dates.DateFormatter("%m-%d")
# ax.xaxis.set_major_formatter(date_format)
# # axins.xaxis.set_major_formatter(date_format)
# #修改X轴日期间隔天数
# # xlocator = mpl.ticker.MultipleLocator(7)
# # ax.xaxis.set_major_locator(xlocator)
# # z = [-3,0,3,6,9,12,15]
# # values=['-3','0','3','6','9','12','15']
# # plt.yticks(z,values)
# plt.grid('y',ls='-.',linewidth=0.5)
# #修改日期表示的斜率
# # plt.xticks(rotationr0)
# #对于X轴，只显示x中各个数对应的刻度值
# plt.xticks( fontsize=7, )  #改变x轴文字值的文字大小
#
# # plt.legend( loc='upper right',fontsize = 5)
# # ax.set(facecolor = "white")
# # ax.spines['right'].set_color('black')  # 右边框设置无色
# # ax.spines['top'].set_color('black')  # 上边框设置无色
# # ax.spines['left'].set_color('black')  # 右边框设置无色
# # ax.spines['bottom'].set_color('black')  # 上边框设置无色
#
# # 添加y轴标签
# plt.xlabel('Date')
# plt.ylabel('Temperature (degree Celsius)',fontsize=10)
#
# # 显示图形degree Celsius
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.patches import ConnectionPatch
import numpy as np

#处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#坐标轴负号的处理
plt.rcParams['axes.unicode_minus']=False
# 读取数据
TCHWS = pd.read_excel(r'total_p.xls')
import matplotlib as mpl
fig,ax = plt.subplots(1,1,figsize=(12,7))
# 绘制阅读人数折线图
plt.plot(TCHWS.Date, # x轴数据
         TCHWS.MFPC, # y轴数据
         linestyle = '-', # 折线类型，实心线
         linewidth = 1,
         # color = 'royalblue', # 折线颜色
         label = 'MFPC'
         )
# 绘制阅读人次折线图
plt.plot(TCHWS.Date, # x轴数据
         TCHWS.DQN, # y轴数据
         linestyle = '-', # 折线类型，虚线
         linewidth= 1,
         # color = 'orangered', # 折线颜色
         label = 'Pure RL'
         )

plt.plot(TCHWS.Date, # x轴数据
         TCHWS.Rule, # y轴数据
         linestyle = '-', # 折线类型，虚线
         linewidth= 1,
         # color = 'sienna', # 折线颜色
         label = 'Rule-based'
         )

plt.plot(TCHWS.Date, # x轴数据
         TCHWS.Model, # y轴数据
         linestyle = '-', # 折线类型，虚线
         linewidth= 1,
         # color = 'darksage', # 折线颜色
         label = 'MPC'
         )

# plt.plot(TCHWS.TIME, # x轴数据
#          TCHWS.A, # y轴数据
#          linestyle = '--', # 折线类型，虚线
#          linewidth= 0.75,
#          # color = 'royalblue', # 折线颜色
#          label = 'Fluctions in MFPC and Pure RL'
#          )
# plt.plot(TCHWS.TIME, # x轴数据
#          TCHWS.B, # y轴数据
#          linestyle = '-', # 折线类型，虚线
#          linewidth= 1.5,
#          # color = 'olivedrab', # 折线颜色
#          label = 'Fluctions in MFPC and MPC'
#          )
plt.legend( loc='upper right',fontsize = 10)

plt.ticklabel_format(style='plain')

#让X轴按照指定左边显示
z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
values=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
plt.xticks(z,values)

# axins.set_ylim(ylimo,ylim1)
#修改日期格式
# date_format = mpl.dates.DateFormatter("%m-%d")
# ax.xaxis.set_major_formatter(date_format)
# axins.xaxis.set_major_formatter(date_format)
#修改X轴日期间隔天数
# xlocator = mpl.ticker.MultipleLocator(7)
# ax.xaxis.set_major_locator(xlocator)
# z = [-3,0,3,6,9,12,15]
# values=['-3','0','3','6','9','12','15']
# plt.yticks(z,values)
plt.grid('y',ls='-.',linewidth=0.5)
#修改日期表示的斜率
# plt.xticks(rotationr0)
#对于X轴，只显示x中各个数对应的刻度值
plt.xticks( fontsize=7, )  #改变x轴文字值的文字大小

# plt.legend( loc='upper right',fontsize = 5)
# ax.set(facecolor = "white")
# ax.spines['right'].set_color('black')  # 右边框设置无色
# ax.spines['top'].set_color('black')  # 上边框设置无色
# ax.spines['left'].set_color('black')  # 右边框设置无色
# ax.spines['bottom'].set_color('black')  # 上边框设置无色

# 添加y轴标签
plt.xlabel('Episodes',fontsize=10)
plt.ylabel('Cumulative power(KW)',fontsize=10)

# 显示图形degree Celsius
plt.show()

