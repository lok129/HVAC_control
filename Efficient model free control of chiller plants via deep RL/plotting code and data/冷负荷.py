import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
excel_file = 'CLS-TWB.xls'  # 替换为你的Excel文件路径
df = pd.read_excel(excel_file)

# 提取数据
dates = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M')
dates = dates.dt.date  # 获取日期部分并去掉时间
cold_load = df['CL']
wet_bulb_temp = df['Twb']

# 创建画布和轴对象
fig, ax1 = plt.subplots()

# 绘制散点图（冷负荷），使用RGB颜色(0, 0, 255)，即纯蓝色
ax1.scatter(dates, cold_load, color=(76/255, 152/255, 206/255), label='Cooling load', s=10)  # Adjust marker size with 's' parameter
ax1.set_xlabel('Date')
ax1.set_ylabel('Cooling Load(KW)', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 创建第二个轴对象，并共享X轴
ax2 = ax1.twinx()

# 绘制折线图（湿球温度），使用RGB颜色(255, 0, 0)，即纯红色
ax2.plot(dates, wet_bulb_temp, color=(247/255, 108/255, 81/255), label='Wet-bulb Temperature', linewidth=2)  # Adjust line thickness with 'linewidth' parameter
ax2.set_ylabel('Wet-bulb Temperature(℃)', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 设置X轴日期显示格式（仅显示月和日）
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

# 设置X轴刻度水平
plt.xticks(rotation=0)
# 显示图形
# plt.title('Cold Load vs. Wet Bulb Temperature')
plt.show()
