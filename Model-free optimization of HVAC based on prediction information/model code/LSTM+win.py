import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn import preprocessing
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.models import Sequential
from sklearn.metrics import mean_squared_error
import math

# Load data
data=pd.read_csv('env_df1h.csv')
data.drop(['chiller_num','pump_num','tower_num'],axis=1,inplace = True)
data['time'] = pd.to_datetime(data['time']) # 转换时间格式
series = data.set_index(['time'],drop=True)# 将时间作为index

print(series)

# Prepare data
window_size = 24 # 滑动窗口大小
X = []
y = []
for i in range(window_size, len(series)):
    X.append(series.iloc[i-window_size:i, [0, 1]].values) # 取出当前窗口的数据
    y.append(series.iloc[i, 1]) # 取出当前窗口的目标值
X = np.array(X)
y = np.array(y)
train_size = int(len(X) * 0.8) # 训练集和测试集的划分比例
train_X, train_y = X[:train_size], y[:train_size]
test_X, test_y = X[train_size:], y[train_size:]



# Define model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # 定义LSTM层
        self.fc = nn.Linear(hidden_size, output_size) # 定义全连接层

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # 初始化LSTM的hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # 初始化LSTM的cell state
        out, _ = self.lstm(x, (h0, c0)) # LSTM前向传播
        out = self.fc(out[:, -1, :]) # 取出最后一个时间步的输出
        return out

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设置使用GPU或CPU
model = LSTM(2, 128, 1, 1).to(device) # 定义模型 这边根据自己需求调整
criterion = nn.MSELoss() # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 定义优化器

num_epochs = 20
batch_size = 32
train_loss = []

for epoch in range(num_epochs):
    for i in range(0, len(train_X), batch_size):
        batch_X = torch.tensor(train_X[i:i+batch_size]).to(device)
        batch_y =torch.tensor(train_y[i:i+batch_size]).unsqueeze(1).to(device)

        # Forward pass
        outputs = model(batch_X.float())
        loss = criterion(outputs, batch_y.float())
        train_loss.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}')

# Plot training loss
import matplotlib.pyplot as plt
plt.plot(train_loss)
plt.xlabel('Iterations')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.show()

# Evaluation
model.eval()
test_X, test_y = torch.tensor(test_X).to(device),torch.tensor(test_y).to(device)
with torch.no_grad():
    y_pred = model(test_X.float())
    test_loss = criterion(y_pred, test_y.unsqueeze(1).float())
print(f'Test Loss: {test_loss.item():.4f}')

# Inverse transform
y_pred = y_pred.cpu().numpy()
test_y = test_y.cpu().numpy()
y_pred = (y_pred * data['CL'].std()) + data['CL'].mean()
test_y = (test_y * data['CL'].std()) + data['CL'].mean()
print(test_y)
print(y_pred)


# Plot predictions
plt.plot
# plt.plot(y_pred, label='Predicted')
# plt.plot(test_y, label='Actual')
# plt.xlabel('Time (hours)')
# plt.ylabel('Cold Load (kW)')
# plt.title('Predicted vs Actual Values')
# plt.legend()
# plt.show()