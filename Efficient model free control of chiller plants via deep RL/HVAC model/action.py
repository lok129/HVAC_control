import numpy as np
# 创建动作的选择变量
action_all = []
for T_chws in range(6,16):
    for f_tower in range(25,51):
            action_all.append([T_chws,f_tower])
print(len(action_all))
action_all = np.array(action_all)
np.save("action_all",action_all)
