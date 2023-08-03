# import numpy as np
# # 创建动作的选择变量
# action_all = []
# for T_chws in range(6,16):
#     for t_f in range(25,51):
#             action_all.append([T_chws,t_f])
# print(len(action_all))
# action_all = np.array(action_all)
# np.save("action_all",action_all)
# import numpy as np
# import pandas as pd
# loaddate =  np.load("action_all.npy")
# print(loaddate)

import numpy as np
import pandas as pd

data = pd.read_csv('env.csv')
np.save('env_1h.npy',data)
