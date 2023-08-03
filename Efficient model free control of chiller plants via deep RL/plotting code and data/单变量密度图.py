import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'''
开始接收数据
接受1、3、6、10这四年的PLR的值，这部分代码已经写好了
'''
#接受大冷机第一年的数据
import xlrd
data_p = xlrd.open_workbook_xls("Pchiller.xls")
RBC = data_p.sheet_by_name('RBC')
MBC = data_p.sheet_by_name('MBC')
P_D_1 = data_p.sheet_by_name("DQN-1")
P_D_5 = data_p.sheet_by_name('DQN-5')
P_D_10 = data_p.sheet_by_name('DQN-10')
P_K_3_1 = data_p.sheet_by_name('K-3-1')
P_K_3_5= data_p.sheet_by_name('K-3-5')
P_K_3_10 = data_p.sheet_by_name('K-3-10')
P_K_4_1 = data_p.sheet_by_name('K-4-1')
P_K_4_5 = data_p.sheet_by_name('K-4-5')
P_K_4_10 = data_p.sheet_by_name('K-4-10')
P_K_5_1 = data_p.sheet_by_name('K-5-1')
P_K_5_5 = data_p.sheet_by_name('K-5-5')
P_K_5_10 = data_p.sheet_by_name('K-5-10')

RULE= []
MODEL = []
DQN_1 = []
DQN_5 = []
DQN_10 = []
K_3_1 = []
K_3_5 =[]
K_3_10 = []
K_4_1 = []
K_4_5 =[]
K_4_10 = []
K_5_1 = []
K_5_5 =[]
K_5_10 = []

for index in range(RBC.nrows):
        RULE.append(float(RBC.cell_value(index,0)))
        MODEL.append(float(MBC.cell_value(index, 0)))
        DQN_1.append(float(P_D_1.cell_value(index, 0)))
        DQN_5.append(float(P_D_5.cell_value(index,0)))
        DQN_10.append(float(P_D_10.cell_value(index, 0)))
        K_3_1.append(float(P_K_3_1.cell_value(index, 0)))
        K_3_5.append(float(P_K_3_5.cell_value(index, 0)))
        K_3_10.append(float(P_K_3_10.cell_value(index, 0)))
        K_4_1.append(float(P_K_4_1.cell_value(index, 0)))
        K_4_5.append(float(P_K_4_5.cell_value(index, 0)))
        K_4_10.append(float(P_K_4_10.cell_value(index, 0)))
        K_5_1.append(float(P_K_5_1.cell_value(index, 0)))
        K_5_5.append(float(P_K_5_5.cell_value(index, 0)))
        K_5_10.append(float(P_K_5_10.cell_value(index, 0)))


'''
开始画图
'''
# sns.set(style='ticks')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# fontsize_num=10
# fig,axes=plt.subplots(1,3,sharey=True)#fig是整个画布，axes是子图,1，2表示1行两列
# sns.kdeplot(K_3_1,shade=True,legend=True,label="Clusting(K=3)",ax=axes[0])
# sns.kdeplot(K_4_1,shade=True,legend=True,label="Clusting(K=4)",ax=axes[0])
# sns.kdeplot(K_5_1,shade=True,legend=True,label="Clusting(K=5)",ax=axes[0])
# sns.kdeplot(DQN_1,shade=True,legend=True,label="Pure RL",ax=axes[0])
# # sns.kdeplot(RULE,shade=True,legend=True,label="Rule-based",ax=axes[0])
# # sns.kdeplot(MODEL,shade=True,legend=True,label="MPC",ax=axes[0])
#
# sns.kdeplot(K_3_5,shade=True,legend=True,label="Clusting(K=3)",ax=axes[1])
# sns.kdeplot(K_4_5,shade=True,legend=True,label="Clusting(K=4)",ax=axes[1])
# sns.kdeplot(K_5_5,shade=True,legend=True,label="Clusting(K=5)",ax=axes[1])
# sns.kdeplot(DQN_1,shade=True,legend=True,label="Pure RL",ax=axes[1])
# # sns.kdeplot(RULE,shade=True,legend=True,label="Rule-based",ax=axes[1])
# # sns.kdeplot(MODEL,shade=True,legend=True,label="MPC",ax=axes[1])
#
# sns.kdeplot(K_3_10,shade=True,legend=True,label="Clusting(K=3)",ax=axes[2])
# sns.kdeplot(K_4_10,shade=True,legend=True,label="Clusting(K=4)",ax=axes[2])
# sns.kdeplot(K_5_10,shade=True,legend=True,label="Clusting(K=5)",ax=axes[2])
# sns.kdeplot(DQN_1,shade=True,legend=True,label="Pure RL",ax=axes[2])
# # sns.kdeplot(RULE,shade=True,legend=True,label="Rule-based",ax=axes[2])
# # sns.kdeplot(MODEL,shade=True,legend=True,label="MPC",ax=axes[2])
#
# plt.subplots_adjust(wspace=0.2)    #子图很有可能左右靠的很近，调整一下左右距离
# fig.set_figwidth(12)                #这个是设置整个图（画布）的大小
# fig.set_figheight(4)                #这个是设置整个图（画布）的大小
#
# axes[0].legend(fontsize=fontsize_num-3)
# axes[1].legend(fontsize=fontsize_num-3)
# axes[2].legend(fontsize=fontsize_num-3)
#
# # axes[0].set_title("Episode 1",fontsize=fontsize_num)
# axes[0].set_xlabel("",fontsize=fontsize_num)
# axes[0].set_ylabel("",fontsize=15)
# axes[0].tick_params(labelsize=fontsize_num)#刻度字体大小13
#
# # axes[1].set_title("Episode 5",fontsize=fontsize_num)
# axes[1].set_xlabel("",fontsize=fontsize_num)
# axes[1].set_ylabel("",fontsize=10)
# axes[1].tick_params(labelsize=fontsize_num)#刻度字体大小13
# plt.setp(axes[2].get_yticklabels(),visible=False)
# # axes[1].spines['left'].set_visible(False)
# axes[1].tick_params(left=False)
# axes[2].tick_params(left=False)
# # axes[2].set_title("Episode 10",fontsize=fontsize_num)
# axes[2].set_xlabel("",fontsize=fontsize_num)
# axes[2].set_ylabel("Density",fontsize=10)
# axes[2].tick_params(labelsize=fontsize_num)#刻度字体大小13
# plt.savefig("P_chiller.jpg",dpi=220)
# plt.show()
# 开始画图

# 开始画图
sns.set(style='ticks')
plt.rcParams['font.sans-serif'] = ['SimHei']
fontsize_num = 12
fig, axes = plt.subplots(1, 3, sharey=True)  # fig是整个画布，axes是子图,1，2表示1行两列

# 定义RGB颜色值
color1 = (233/255, 196/255, 107/255)
color2 = (233/255, 111/255, 81/255)
color3 = (38/255, 70/255, 83/255)
color4 = (42/255, 157/255, 142/255)

# 第一列图
sns.kdeplot(K_3_1, shade=True, color=color1, label="Clusted (K=3)", ax=axes[0])
sns.kdeplot(K_4_1, shade=True, color=color2, label="Clusted (K=4)", ax=axes[0])
sns.kdeplot(K_5_1, shade=True, color=color3, label="Clusted (K=5)", ax=axes[0])
sns.kdeplot(DQN_1, shade=True, color=color4, label="Pure RL", ax=axes[0])

# 第二列图
sns.kdeplot(K_3_5, shade=True, color=color1, label="Clusted (K=3)", ax=axes[1])
sns.kdeplot(K_4_5, shade=True, color=color2, label="Clusted (K=4)", ax=axes[1])
sns.kdeplot(K_5_5, shade=True, color=color3, label="Clusted (K=5)", ax=axes[1])
sns.kdeplot(DQN_5, shade=True, color=color4, label="Pure RL", ax=axes[1])

# 第三列图
sns.kdeplot(K_3_10, shade=True, color=color1, label="Clusted (K=3)", ax=axes[2])
sns.kdeplot(K_4_10, shade=True, color=color2, label="Clusted (K=4)", ax=axes[2])
sns.kdeplot(K_5_10, shade=True, color=color3, label="Clusted (K=5)", ax=axes[2])
sns.kdeplot(DQN_10, shade=True, color=color4, label="Pure RL", ax=axes[2])

plt.subplots_adjust(wspace=0.2)  # 子图很有可能左右靠的很近，调整一下左右距离
fig.set_figwidth(12)  # 这个是设置整个图（画布）的大小
fig.set_figheight(4)  # 这个是设置整个图（画布）的大小

axes[0].legend(fontsize=fontsize_num - 3)
axes[1].legend(fontsize=fontsize_num - 3)
axes[2].legend(fontsize=fontsize_num - 3)

axes[0].set_xlabel("", fontsize=fontsize_num)
axes[0].set_ylabel("", fontsize=15)
axes[0].tick_params(labelsize=fontsize_num)  # 刻度字体大小13
# axes[0].set_xticks(range(0, 25, 5))  # 设置X轴刻度范围和间隔
axes[0].set_xticks(range(0, 401, 100))  # 设置X轴刻度范围和间隔
# axes[0].set_yticks([0,0.002, 0.004, 0.006, 0.008])  # 设置Y轴刻度范围

axes[1].set_xlabel("", fontsize=fontsize_num)
axes[1].set_ylabel("", fontsize=10)
axes[1].tick_params(labelsize=fontsize_num)  # 刻度字体大小13
# axes[1].set_xticks(range(0, 25, 5))  # 设置X轴刻度范围和间隔
axes[1].set_xticks(range(0, 401, 100))  # 设置X轴刻度范围和间隔
# axes[1].set_yticks([0,0.002, 0.004, 0.006, 0.008])  # 设置Y轴刻度范围

plt.setp(axes[2].get_yticklabels(), visible=False)
axes[1].tick_params(left=False)
axes[2].tick_params(left=False)

axes[2].set_xlabel("", fontsize=fontsize_num)
axes[2].set_ylabel("", fontsize=10)
axes[2].tick_params(labelsize=fontsize_num)  # 刻度字体大小13
# axes[2].set_xticks(range(0, 25, 5))  # 设置X轴刻度范围和间隔
axes[2].set_xticks(range(0, 401, 100))  # 设置X轴刻度范围和间隔
# axes[2].set_yticks([0,0.002, 0.004, 0.006, 0.008])  # 设置Y轴刻度范围

plt.savefig("P_chiller.jpg", dpi=220)
plt.show()