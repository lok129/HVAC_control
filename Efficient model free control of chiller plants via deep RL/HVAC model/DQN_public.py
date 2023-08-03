import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model
import math
import xlwt
from torchsummary import summary
import time

reward_shaping = True
showFigures = True
printDetails = True
ALPHA = 0.01
ALPHA_decay = 0.9
ALPHA_ratio = 0.01
GAMMA = 0.01
BATCH_SIZE = 32
BATCH_SIZE_RATIO = 32
TARGET_UPDATE = 200
MEMORY_CAPACITY = 2000
EPS = 1
EPS_Ratio = 1
EPS_MIN = 0.01
EPS_DECAY = 0.0001
EPS_DECAY_ratio = 0.0007
LearnStep = 0
episodes = 20

# Setup the problem
env = model.model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Net class
class Net(nn.Module):
    def __init__(self, STATE_SIZE, ACTION_SIZE):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 128)
        self.fc2 = nn.Linear(128, ACTION_SIZE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SharedExperiencePool:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = np.zeros((capacity, env.state_size * 2 + 2))
        self.mem_counter = 0

    def store_transition(self, s, a, r, s1):
        transition = np.hstack((s, [a, r], s1))
        index = self.mem_counter % self.capacity
        self.memory[index, :] = transition
        self.mem_counter += 1

    def sample(self, batch_size):
        sample_index = np.random.choice(min(self.mem_counter, self.capacity), batch_size)
        return self.memory[sample_index, :]

# choose_tchws class
class choose_tchws:
    def __init__(self, STATE_SIZE, ACTION_SIZE, action_space):
        self.STATE_SIZE = STATE_SIZE
        self.ACTION_SIZE = ACTION_SIZE
        self.action_space = action_space
        self.policy_net = Net(STATE_SIZE, ACTION_SIZE).to(device)  # policy network
        self.target_net = Net(STATE_SIZE, ACTION_SIZE).to(device)  # target network
        self.target_net.load_state_dict(self.policy_net.state_dict())  # copy the policy net parameters to target net
        self.target_net.eval()  # .eval()的作用是使得网络的学习参数不发生改变。
        self.learn_counter = 0  # for target updating
        self.memory = np.zeros((MEMORY_CAPACITY, STATE_SIZE * 2 + 2))  # initialize replay memory
        self.shared_memory = SharedExperiencePool(MEMORY_CAPACITY)  # initialize shared experience pool
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=ALPHA_ratio)  # optimizer
        self.loss_func = nn.MSELoss()  # loss function
        self.success_counter = 0  # counting # of consecutive successes
        self.isSuccess = False  # is successed
        self.isStartedLearning = True
        self.listScore = []  # store all scores

    def select_action(self, state):
        rand_num = random.random()
        global EPS_Ratio
        global EPS_DECAY_ratio
        EPS_Ratio = EPS_Ratio - EPS_DECAY_ratio
        if EPS_Ratio >= EPS_MIN:
            EPS_Ratio = EPS_Ratio
        else:
            EPS_Ratio = EPS_MIN
        if rand_num > EPS_Ratio:  # greedy
            state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
            list = self.policy_net.forward(state)
            index = torch.argmax(list[0][0:260])
            return index
        else:
            state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
            return torch.tensor([[random.randrange(0, 260)]], device=device, dtype=torch.long).item()

    def store_transition(self, s, a, r, s1):
        transition = np.hstack((s, [a, r], s1))
        index = self.mem_counter % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.mem_counter += 1
        self.shared_memory.store_transition(s, a, r, s1)  # store transition in shared memory

    def learn(self):
        if self.learn_counter % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_counter += 1
        batch_size = BATCH_SIZE
        if self.shared_memory.mem_counter < BATCH_SIZE:
            batch_size_ratio = self.shared_memory.mem_counter
        else:
            batch_size_ratio = BATCH_SIZE_RATIO
        if self.mem_counter > BATCH_SIZE:
            batch_size -= batch_size_ratio
            batch = np.vstack((self.memory[:BATCH_SIZE], self.shared_memory.sample(batch_size_ratio)))
        else:
            batch = self.memory[:self.mem_counter]
        np.random.shuffle(batch)
        b_s = torch.FloatTensor(batch[:, :self.STATE_SIZE]).to(device)
        b_a = torch.LongTensor(batch[:, self.STATE_SIZE:self.STATE_SIZE + 1].astype(int)).to(device)
        b_r = torch.FloatTensor(batch[:, self.STATE_SIZE + 1:self.STATE_SIZE + 2]).to(device)
        b_s1 = torch.FloatTensor(batch[:, -self.STATE_SIZE:]).to(device)

        Q = self.policy_net(b_s).gather(1, b_a)  # shape (batch, 1)
        Q1 = self.target_net(b_s1).detach()  # detach from graph, don't backpropagate
        target = b_r + GAMMA * Q1.max(1)[0].view(batch_size, 1)  # shape (batch, 1)
        loss = self.loss_func(Q, target)
        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

# DQN class
class DQN:
    def __init__(self):
        self.action = np.load("action_all.npy", allow_pickle=True)

    def train(self):
        Agent = choose_tchws(2, 260, self.action)
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
            CLs_data_sheet = book.add_sheet('CLs' + str(episode), cell_overwrite_ok=True)
            CLc_data_sheet = book.add_sheet('CLc' + str(episode), cell_overwrite_ok=True)
            P_chiller_sheet = book.add_sheet('P_chiller' + str(episode), cell_overwrite_ok=True)
            P_tower_sheet = book.add_sheet('P_tower' + str(episode), cell_overwrite_ok=True)
            R_sheet = book.add_sheet('R' + str(episode), cell_overwrite_ok=True)
            R_p_sheet = book.add_sheet('R_P' + str(episode), cell_overwrite_ok=True)
            R_c_sheet = book.add_sheet('R_C' + str(episode), cell_overwrite_ok=True)
            T_chws_sheet = book.add_sheet('T_chws' + str(episode), cell_overwrite_ok=True)
            f_tower_sheet = book.add_sheet('f_tower' + str(episode), cell_overwrite_ok=True)
            T_chwr_sheet = book.add_sheet('T_chwr' + str(episode), cell_overwrite_ok=True)
            T_outdoor_sheet = book.add_sheet('T_outdoor' + str(episode), cell_overwrite_ok=True)
            P_chiller_total_sheet = book.add_sheet('P_chiller_total' + str(episode), cell_overwrite_ok=True)
            NUMBER_sheet = book.add_sheet('number' + str(episode), cell_overwrite_ok=True)

            numeric_style = xlwt.XFStyle()
            numeric_style.num_format_str = '0.000'

            while True:
                global LearnStep
                LearnStep += 1
                CLs = s[0]
                step += 1
                action_index = Agent.select_action(s)
                T_chws = self.action[action_index][0]
                f_tower = self.action[action_index][1]

                S_, CLc, Done, P_chiller, P_tower, R, R_p, R_c, T_chwr, T_outdoor, P_total, Chiller_number = env.step(self.action[action_index])

                CLs_data_sheet.write(row, 0, float(CLs), numeric_style)
                CLc_data_sheet.write(row, 0, float(CLc), numeric_style)
                R_sheet.write(row, 0, float(R), numeric_style)
                P_chiller_sheet.write(row, 0, float(P_chiller), numeric_style)
                P_tower_sheet.write(row, 0, float(P_tower), numeric_style)
                R_p_sheet.write(row, 0, float(R_p), numeric_style)
                R_c_sheet.write(row, 0, float(R_c), numeric_style)
                T_chws_sheet.write(row, 0, float(T_chws), numeric_style)
                f_tower_sheet.write(row, 0, float(f_tower), numeric_style)
                T_chwr_sheet.write(row, 0, float(T_chwr), numeric_style)
                T_outdoor_sheet.write(row, 0, float(T_outdoor), numeric_style)
                P_chiller_total_sheet.write(row, 0, float(P_total), numeric_style)
                NUMBER_sheet.write(row, 0, float(Chiller_number), numeric_style)

                Agent.store_transition(s, action_index, R, S_)
                r += R
                if Agent.mem_counter > BATCH_SIZE:
                    loss = Agent.learn()
                    loss_sheet.write(row_loss, 0, str(loss.item()))
                    row_loss += 1
                row += 1
                book.save("T_chws - P_chiller " + str(episode) + '.xls')
                if Done:
                    print("Epi:", episode, "Reward:", r, "EPS_Ratio", EPS_Ratio)
                    break
                s = S_
        Agent.save_model("model.pth1")

if __name__ == "__main__":
    DQN = DQN()
    start_time = time.time()
    DQN.train()
    end_time = time.time()
    train_time = end_time - start_time
    print("训练时间", train_time, "秒")
