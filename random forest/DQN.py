import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model
import math
import xlwt

reward_shaping  = True
showFigures     = True
printDetails    = True
ALPHA           = 0.01
ALPHA_decay     = 0.9
ALPHA_ratio     = 0.01
#ALPHA_ratio_decay = 0.9

GAMMA           = 0.01
BATCH_SIZE      = 32
BATCH_SIZE_RATIO = 32
TARGET_UPDATE   = 200
MEMORY_CAPACITY_artio = 600
MEMORY_CAPACITY = 2000
EPS             = 1
EPS_Ratio       = 1
EPS_MIN: float  = 0.01
EPS_DECAY       = 0.0001#差不多2000步之后不在探索
EPS_DECAY_ratio = 0.0005#差不多200步之后不再探索
LearnStep = 0
episodes        = 20
cap_Chiller = 1060
############################# Setup the problem #############################
env = model.chiller_plants()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############################# Neural Net class #############################
class Net(nn.Module):
    def __init__(self,STATE_SIZE,ACTION_SIZE):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 32)
        self.fc2 = nn.Linear(32, ACTION_SIZE)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class choose_tchws:
    def __init__(self,STATE_SIZE,ACTION_SIZE,action_space):
        self.STATE_SIZE = STATE_SIZE
        self.ACTION_SIZE = ACTION_SIZE
        self.action_space = action_space
        self.policy_net = Net(STATE_SIZE,ACTION_SIZE).to(device)  # policy network
        self.target_net = Net(STATE_SIZE,ACTION_SIZE).to(device)  # target network
        self.target_net.load_state_dict(self.policy_net.state_dict())  # copy the policy net parameters to target net
        self.target_net.eval()  # .eval()的作用是使得网络的学习参数不发生改变。
        self.learn_counter = 0  # for target updating
        self.mem_counter = 0  # for counting memor
        self.memory = np.zeros((MEMORY_CAPACITY, STATE_SIZE * 2 + 2))  # initialize replay memory
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=ALPHA_ratio)  # optimizer
        self.loss_func = nn.MSELoss()  # loss function
        self.success_counter = 0  # counting # of consecutive successes
        self.isSuccess = False  # is successed
        self.isStartedLearning = True
        self.listScore = []  # store all scores
    def select_action(self, state):#
        rand_num = random.random()
        global EPS_Ratio
        global EPS_DECAY_ratio
        EPS_Ratio = EPS_Ratio - EPS_DECAY_ratio
        if EPS_Ratio >= EPS_MIN:
            EPS_Ratio = EPS_Ratio
        else:
            EPS_Ratio = EPS_MIN
        if rand_num > EPS_Ratio: # greedy
                state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
                list = self.policy_net.forward(state)
                index = torch.argmax(list[0][0:260])
                return index
        else:
                state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
                return torch.tensor([[random.randrange(0,260)]], device=device, dtype=torch.long).item()
    def store_transition(self, s, a, r, s1):
        transition = np.hstack((s, [a, r], s1))
        index = self.mem_counter % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.mem_counter += 1
    def learn(self):
        if self.learn_counter % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_counter += 1
        sample_index = np.random.choice(min(self.mem_counter,MEMORY_CAPACITY), BATCH_SIZE)
        b_memory= self.memory[sample_index, :]
        b_s  = torch.FloatTensor(b_memory[:, :self.STATE_SIZE]).to(device)
        b_a  = torch.LongTensor(b_memory[:, self.STATE_SIZE:self.STATE_SIZE+1].astype(int)).to(device)
        b_r  = torch.FloatTensor(b_memory[:, self.STATE_SIZE+1:self.STATE_SIZE+2]).to(device)
        b_s1 = torch.FloatTensor(b_memory[:, -self.STATE_SIZE:]).to(device)
        Q      = self.policy_net(b_s).gather(1, b_a)  # shape (batch, 1)
        Q1     = self.target_net(b_s1).detach()     # detach from graph, don't backpropagate
        target = b_r + GAMMA * Q1.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss   = self.loss_func(Q,target)
        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

class DQN:
    def __init__(self):
        self.action = np.load("action_all.npy",allow_pickle=True)
    def train(self):
        Agent = choose_tchws(2,260,self.action)
        R_list = []
        num = 0
        for episode in range(episodes):
            s = env.reset()
            print(s)
            r = 0
            step = 0
            row = 0
            row_loss = 0
            book = xlwt.Workbook()
            loss_sheet = book.add_sheet('loss', cell_overwrite_ok=True)
            R_data_sheet = book.add_sheet('R'+str(episode), cell_overwrite_ok=True)
            P_chiller_sheet = book.add_sheet('P_chiller' + str(episode), cell_overwrite_ok=True)
            Ptower_sheet = book.add_sheet('Ptower'+str(episode), cell_overwrite_ok=True)
            T_chws_sheet = book.add_sheet('T_chws'+str(episode), cell_overwrite_ok=True)
            t_f_sheet = book.add_sheet('t_f' + str(episode), cell_overwrite_ok=True)
            Ptotal_sheet = book.add_sheet('Ptotal'+str(episode), cell_overwrite_ok=True)
            Tchwr_sheet = book.add_sheet('Tchwr' + str(episode), cell_overwrite_ok=True)
            Tcwr_sheet = book.add_sheet('Tcwr' + str(episode), cell_overwrite_ok=True)


            while True:
                global LearnStep
                LearnStep += 1
                step += 1
                action_index = Agent.select_action(s)
                a = self.action[action_index]

                s_,R,Pchiller,Ptower,TCHWS_SET,t_f,Ptotal,Tchwr,Tcwr,Done = env.step(a)

                R_data_sheet.write(row, 0, str(R))
                P_chiller_sheet.write(row, 0, str(Pchiller))
                Ptower_sheet.write(row, 0, str(Ptower))
                T_chws_sheet.write(row, 0, str(TCHWS_SET))
                t_f_sheet.write(row, 0, str(t_f))
                Ptotal_sheet.write(row,0,str(Ptotal))
                Tchwr_sheet.write(row, 0, str(Tchwr))
                Tcwr_sheet.write(row, 0, str(Tcwr))

                Agent.store_transition(s, action_index,R,s_)
                if Agent.mem_counter > BATCH_SIZE:
                    loss = Agent.learn()
                    loss_sheet.write(row, 0, str(loss.item()))
                    row_loss += 1
                row += 1
                book.save("T_chws-T_f"+str(episode)+ '.xls')
                if Done:
                    r += R
                    print("Epi:",episode,"Reward:",r,"EPS_Ratio",EPS_Ratio)
                    break
                s = s_

if __name__ == "__main__":
    DQN = DQN()
    DQN.train()