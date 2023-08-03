# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取 Excel 文件
# file_path = "TEST-PC.xlsx"
# df = pd.read_excel(file_path)
#
# # 将日期列设置为索引
# df.set_index("DATA", inplace=True)
#
# # 绘制柱状图
# plt.figure(figsize=(12, 6))
#
# # 确定每个柱状图的宽度
# bar_width = 0.1
#
# # 计算每个柱状图的位置
# positions = [pos + bar_width * (i - 3) for pos in range(1, len(df.index) + 1) for i in range(len(df.columns))]
#
# # 循环绘制每个方法的柱状图
# for i, method in enumerate(df.columns):
#     # 确定每个柱状图的横坐标位置
#     x = [pos + bar_width * i for pos in positions]
#     # 绘制当前方法的柱状图
#     plt.bar(x, df[method], width=bar_width, label=method)
#
# # 设置图表标签
# plt.xlabel('时间')
# plt.ylabel('变量值')
# plt.title('柱状图')
# plt.xticks([pos + bar_width * 3 for pos in range(1, len(df.index) + 1)], df.index, rotation=45)
#
# # 显示图例
# plt.legend()
#
# # 显示图表
# plt.tight_layout()
# plt.show()
#
# # #雷达图
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 从 Excel 文件加载数据
# data_file = '123456.xlsx'
# df = pd.read_excel(data_file)
#
# # 将 'DATA' 列设置为索引
# df.set_index('DATA', inplace=True)
#
# # 提取方法名和数据用于绘制图表
# methods = df.columns
# num_methods = len(methods)
#
# # 设置图表
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # 将日期转换为字符串形式
# date_labels = df.index.strftime('%Y-%m-%d')
#
# # 计算每个方法的偏移量，以便柱状图分开显示
# num_offsets = len(methods)
# bar_width = 0.2
# bar_offsets = [-bar_width * (num_offsets // 2) + (i * bar_width) for i in range(num_offsets)]
#
# # 绘制每个方法的柱状图和折线图
# for i, method in enumerate(methods):
#     bar_positions = [pos + bar_offsets[i] for pos in range(len(df.index))]
#     ax.bar(bar_positions, df[method], width=bar_width, label=method, alpha=0.6)
#     ax.plot(range(len(df.index)), df[method], marker='o', linestyle='-', color='black', label=f"{method} (Line)")
#
# # 设置 x 轴标签和 y 轴标签
# ax.set_xlabel('日期')
# ax.set_ylabel('能耗')
#
# # 设置 x 轴刻度和标签
# plt.xticks(range(len(df.index)), date_labels, rotation=45)
#
# # 添加图例
# ax.legend()
#
# # 显示图表
# plt.tight_layout()
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# #热力图
# 从 Excel 文件加载数据
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 从 Excel 文件加载数据
# data_file = '123456.xlsx'
# df = pd.read_excel(data_file)
#
# # 将 'DATA' 列设置为索引
# df.set_index('DATA', inplace=True)
#
# # 设置图表大小
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # 绘制热力图
# sns.heatmap(df.T, cmap='coolwarm', annot=True, fmt=".1f", linewidths=.5, ax=ax)
#
# # 设置 x 轴标签和 y 轴标签
# ax.set_xlabel('日期')
# ax.set_ylabel('方法')
#
# # 添加标题
# ax.set_title('六种方法能耗热力图')
#
# # 调整布局，使图表更美观
# plt.tight_layout()
#
# # 显示图表
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 从 Excel 文件加载数据
data_file = '12345.xlsx'
df = pd.read_excel(data_file)

# 将 'DATA' 列设置为索引
df.set_index('DATA', inplace=True)

# 将日期的格式从 '2023-08-01' 转换为 '8-1' 形式
df.index = df.index.strftime('%m-%d')

# 设置图表大小
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制热力图，去掉标注数字，并使用蓝色色系
sns.heatmap(df.T, cmap='OrRd', annot=False, fmt=".1f", linewidths=.5, ax=ax)


# 调整布局，使图表更美观
plt.tight_layout()

# 显示图表
plt.show()
