# import pandas as pd
# import numpy as np
#
# # 先用pandas读入csv
# data = pd.read_csv("env_df30min.csv")
# # 再使用numpy保存为npy
# np.save("env_df30min.npy", data)

# 两种方法都能打开
import pickle
import numpy as np

f = open('chiller_model.pkl','rb')
data = pickle.load(f)
print(data)

# img_path = './train_data.pkl'
# img_data = np.load(img_path)
# print(img_data)

