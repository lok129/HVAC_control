import pandas as pd
from sklearn.cluster import KMeans

# 读取CSV文件，并将第一列作为时间索引
data = pd.read_csv('env_df30min.csv', index_col=0)

# 提取冷负荷和湿球温度数据
X = data[['CL', 'Twb']].values

# 设置聚类的K值
k_load = 5  # 冷负荷聚类的K值
k_temperature = 2  # 湿球温度聚类的K值

# 进行冷负荷聚类
kmeans_load = KMeans(n_clusters=k_load)
labels_load = kmeans_load.fit_predict(X)

# 进行湿球温度聚类
kmeans_temperature = KMeans(n_clusters=k_temperature)
labels_temperature = kmeans_temperature.fit_predict(X)

# 将聚类结果添加到原始数据中
data['LoadCluster'] = labels_load
data['TemperatureCluster'] = labels_temperature

# 根据时间索引，合并子集数据
result = data.groupby(['LoadCluster', 'TemperatureCluster']).groups

# 输出结果
# for cluster_pair, indices in result.items():
#     load_cluster, temperature_cluster = cluster_pair
#     subset = data.loc[indices]
#     print(f"Load Cluster: {load_cluster}, Temperature Cluster: {temperature_cluster}")
#     print(subset)
#     print("========================")
with pd.ExcelWriter('output(30)_5.xlsx') as writer:
    for cluster_pair, indices in result.items():
        load_cluster, temperature_cluster = cluster_pair
        subset = data.loc[indices]
        subset.to_excel(writer, sheet_name=f'Cluster_{load_cluster}_{temperature_cluster}')

print("保存成功！")