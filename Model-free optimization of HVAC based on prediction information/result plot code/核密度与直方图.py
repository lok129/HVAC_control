import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xlrd


def excel2matrix(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows, ncols))
    for i in range(nrows):
        rows = table.row_values(i)
        datamatrix[i,:] = rows
    return datamatrix

pathX = 'MFPC_t.xls'
y = excel2matrix(pathX)
# print(x)
# print(x.shape)

#多数据直方图
plt.hist(x=y,
         bins=10,
         edgecolor='w',  # 指定直方图的边框色
         # color=['r', 'o','g'],  # 指定直方图的填充色
         label=['Episode 1', 'Episode 5','Episode 10'],  # 为直方图呈现图例
         density=True,  # 是否将纵轴设置为密度，即频率
         alpha=0.4,  # 透明度
         rwidth=1,  # 直方图宽度百分比：0-1
         stacked=True)  # 当有多个数据时，是否需要将直方图呈堆叠摆放，默认水平摆放
plt.xlabel('Temperature(degrees celsius)')
plt.ylabel('Distribution density')
plt.grid(linestyle='-.')
z = [6,7,8,9,10,11,12,13,14,15]
values=['6','7','8','9','10','11','12','13','14','15']
plt.xticks(z,values)
ax = plt.gca()  # 获取当前子图
# ax.spines['right'].set_color('none')  # 右边框设置无色
# ax.spines['top'].set_color('none')  # 上边框设置无色
plt.legend()
plt.show()


# #读取excle
# filename = "MFPC_t.xls"
# data = pd.read_excel(filename)
#
# fig,axes = plt.subplots(1,3,figsize=(15,5))
#
# sns.histplot(data["MFPC"],kde=True,color="blue",ax = axes[0],alpha=0.5)
# sns.histplot(data["MFPC1"],kde=True,color="blue",ax = axes[1],alpha=0.5)
# sns.histplot(data["MFPC2"],kde=True,color="blue",ax = axes[2],alpha=0.5)
#
# axes[0].set_title('Round 1')
# axes[1].set_title('Round 2')
# axes[2].set_title('Round 3')
#
# # 设置横纵坐标标签
# for ax in axes:
#     ax.set_xlabel('Variable Value')
#     ax.set_ylabel('Frequency')
#
# # 显示图形
# plt.show()