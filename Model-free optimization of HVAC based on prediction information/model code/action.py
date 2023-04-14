import numpy as np
# 创建动作的选择变量
action_all = []
for T_chws in range(6,16):
            action_all.append([T_chws])
print(len(action_all))
action_all = np.array(action_all)
np.save("action_all",action_all)
