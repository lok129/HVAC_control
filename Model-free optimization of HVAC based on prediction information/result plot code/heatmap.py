import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
df = pd.read_excel('pppchiller.xls',sheet_name='Sheet1')
col1 = df['mfpc'].tolist()
result = np.array(col1).reshape((102,24))
result2 = []
for i in range(len(result[0])):
    b=[]
    for j in range(len(result)):
        b.append(result[j][i])
    result2.append(b)
Index = ["00:00","01:00","02:00","03:00","04:00","05:00","06:00","07:00","08:00","09:00","10:00","11:00","12:00","13:00","14:00","15:00","16:00","17:00","18:00","19:00","20:00","21:00","22:00","23:00"]
Columns = [i for i in range(102)]
date= pd.DataFrame(data=result2,index=Index,columns=Columns)
writer = pd.ExcelWriter("mfpc.xlsx")
date.to_excel(writer)
writer.save()
print(date)
sns.heatmap(data=date,square=False,cmap='PuBu')

plt.xticks(fontsize=7,rotation=0)
plt.yticks(fontsize=7)
plt.xlabel('Days',fontsize = 8)
plt.ylabel('Time',fontsize = 8)

plt.show()
# 单变量 密度图
# data = pd.read_excel("DQN_TCHWS.xlsx",index_col=0)
# sns.distplot(data,hist = True, kde = True, rug = True)
# plt.show()
#六边形的散点图
# sns.jointplot(x='X',y='Y',data=data,kind='hex',height=5,space=0)
#圆点散点图
# sns.scatterplot(x='Expert manual',y='model -free',data=data)
#变量回归图
# sns.set(color_codes=True)
# data = pd.read_csv('10%_csv.csv')
# data.head()
# plt.figure(figsize = (7,7))
#
# sns.regplot(x="0", y="1", data=data)
# plt.show()
