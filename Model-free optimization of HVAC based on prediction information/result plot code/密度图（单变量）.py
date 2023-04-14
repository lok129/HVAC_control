import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'''
开始接收数据
接受1、3、6、10这四年的PLR的值，这部分代码已经写好了
'''
#接受大冷机第一年的数据
import xlrd
data_m = xlrd.open_workbook_xls("PCHILLER.xls")
data_d = xlrd.open_workbook_xls('PCHILLER_DQN.xls')
data_r = xlrd.open_workbook_xls('PCHILLER-RMM.xls')
P_M_1 = data_m.sheet_by_name("Sheet1")
P_M_5 = data_m.sheet_by_name('Sheet2')
P_M_10 = data_m.sheet_by_name('Sheet3')
P_D_1 = data_d.sheet_by_name('Sheet1')
P_D_5 = data_d.sheet_by_name('Sheet2')
P_D_10 = data_d.sheet_by_name('Sheet3')
P_R_1 = data_r.sheet_by_name('Sheet1')
P_MO_1 = data_r.sheet_by_name('Sheet2')
MFPC_1 =[]
DQN_1 = []
RULE= []
MODEL = []
MFPC_5 = []
MFPC_10 = []
DQN_5 = []
DQN_10 = []
for index in range(P_M_1.nrows):
        MFPC_1.append(float(P_M_1.cell_value(index,0)))
        MFPC_5.append(float(P_M_5.cell_value(index, 0)))
        MFPC_10.append(float(P_M_10.cell_value(index, 0)))
        DQN_1.append(float(P_D_1.cell_value(index,0)))
        DQN_5.append(float(P_D_5.cell_value(index, 0)))
        DQN_10.append(float(P_D_10.cell_value(index, 0)))
        RULE.append(float(P_R_1.cell_value(index, 0)))
        MODEL.append(float(P_MO_1.cell_value(index, 0)))

# import xlrd
# data = xlrd.open_workbook_xls("p-5.xls")
# t_chws_data = data.sheet_by_name("Sheet1")
# rule_based_5 = []
# manual_based_5 = []
# DQN_5 = []
# DQN_F_5 = []
#
# for index in range(t_chws_data.nrows):
#     if t_chws_data.cell_value(index,1) == 1:
#         rule_based_5.append(float(t_chws_data.cell_value(index,0)))
#     if t_chws_data.cell_value(index,1) == 2:
#         manual_based_5.append(float(t_chws_data.cell_value(index,0)))
#     if t_chws_data.cell_value(index,1) == 3:
#         DQN_5.append(float(t_chws_data.cell_value(index,0)))
#     if t_chws_data.cell_value(index, 1) == 4:
#         DQN_F_5.append(float(t_chws_data.cell_value(index, 0)))
#


'''
开始画图
'''
sns.set(style='ticks')
plt.rcParams['font.sans-serif'] = ['SimHei']
fontsize_num=10
fig,axes=plt.subplots(1,3,sharey=True)#fig是整个画布，axes是子图,1，2表示1行两列
sns.kdeplot(MFPC_1,shade=True,legend=True,label="MFPC",ax=axes[0])
sns.kdeplot(DQN_1,shade=True,legend=True,label="Pure RL",ax=axes[0])
sns.kdeplot(RULE,shade=True,legend=True,label="Rule_based",ax=axes[0])
sns.kdeplot(MODEL,shade=True,legend=True,label="MPC",ax=axes[0])

sns.kdeplot(MFPC_5,shade=True,legend=True,label="MFPC",ax=axes[1])
sns.kdeplot(DQN_5,shade=True,legend=True,label="Pure RL",ax=axes[1])
sns.kdeplot(RULE,shade=True,legend=True,label="Rule_based",ax=axes[1])
sns.kdeplot(MODEL,shade=True,legend=True,label="MPC",ax=axes[1])

sns.kdeplot(MFPC_10,shade=True,legend=True,label="MFPC",ax=axes[2])
sns.kdeplot(DQN_10,shade=True,legend=True,label="Pure RL",ax=axes[2])
sns.kdeplot(RULE,shade=True,legend=True,label="Rule_based",ax=axes[2])
sns.kdeplot(MODEL,shade=True,legend=True,label="MPC",ax=axes[2])

plt.subplots_adjust(wspace=0.2)    #子图很有可能左右靠的很近，调整一下左右距离
fig.set_figwidth(12)                #这个是设置整个图（画布）的大小
fig.set_figheight(4)                #这个是设置整个图（画布）的大小

axes[0].legend(fontsize=fontsize_num-2)
axes[1].legend(fontsize=fontsize_num-2)
axes[2].legend(fontsize=fontsize_num-2)

axes[0].set_title("Episode 1",fontsize=fontsize_num)
axes[0].set_xlabel("P_Chiller(KW)",fontsize=fontsize_num)
axes[0].set_ylabel("Density",fontsize=15)
axes[0].tick_params(labelsize=fontsize_num)#刻度字体大小13

axes[1].set_title("Episode 5",fontsize=fontsize_num)
axes[1].set_xlabel("P_Chiller(KW)",fontsize=fontsize_num)
axes[1].set_ylabel("Density",fontsize=10)
axes[1].tick_params(labelsize=fontsize_num)#刻度字体大小13
plt.setp(axes[2].get_yticklabels(),visible=False)
# axes[1].spines['left'].set_visible(False)
axes[1].tick_params(left=False)
axes[2].tick_params(left=False)
axes[2].set_title("Episode 10",fontsize=fontsize_num)
axes[2].set_xlabel("P_Chiller(KW)",fontsize=fontsize_num)
axes[2].set_ylabel("Density",fontsize=10)
axes[2].tick_params(labelsize=fontsize_num)#刻度字体大小13
plt.savefig("P_chiller.jpg",dpi=220)
plt.show()
