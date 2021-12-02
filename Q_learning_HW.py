import gym.spaces
import numpy as np
from gym import Env
from gym.utils import seeding
from pprint import pprint


class matrixChain(Env):
    def __init__(self,n = 3):
        self.n = n
        self.positions = [[0 for _ in range(self.n)] for _ in range(self.n)]
        self.current_pos = [0,0]
        for i in range(self.n):
            for j in range(self.n):
                if i == 0 and j == 2:
                    self.positions[i][j] = 10
                elif (i == 0 and j == 1) or (i==1 and j ==1):
                    self.positions[i][j] = -1
                else:
                    self.positions[i][j] = 1
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(self.n),
            gym.spaces.Discrete(self.n))
        )
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        assert self.action_space.contains(action)
        if action == 0:
            movement = (1,0)
            reward = self.getReward(self.positions,self.current_pos,movement)
        elif action == 1:
            movement = (-1,0)
            reward = self.getReward(self.positions,self.current_pos,movement)
        elif action == 2:
            movement = (0,-1)
            reward = self.getReward(self.positions,self.current_pos,movement)
        elif action == 3:
            movement = (0,1)
            reward = self.getReward(self.positions,self.current_pos,movement)
        done = False
        return self.current_pos,reward,done,{}

    def getReward(self, position,current_pos,movement):
        move = [current_pos[0] + movement[0] , current_pos[1] + movement[1] ]
        if move[0] >= 0 and move[0] < self.n and move[1] >= 0 and move[1] < self.n :
            reward = position[move[0]][move[1]]
            self.current_pos = move
        else: 
            pos = self.current_pos
            reward = position[pos[0]][pos[1]]
            self.current_pos = current_pos
        return reward
    
    def reset(self):
        self.current_pos = [0,0]
        return self.current_pos

        

def QRL(env):
    state_space = (env.observation_space[0].n ,env.observation_space[1].n )
    action_space = env.action_space.n 

    Q_table = np.zeros((state_space[0],state_space[1],action_space))

    epsilon = 0.2 # greedy 100% 
    epsilon_min = 0.005 #minimum greedy 0.05%
    epsilon_decay = 0.99993 # decay multiplied with epsilon after each episode 

    episodes = 50  #amount of games 
    max_steps = 200 #maximum steps per episode 
    learning_rate = 1
    Gamme = 0.2

    for episode in range(episodes):
        state = env.reset()
        done = False 
        score = 0 
        for _ in range(max_steps):
            if np.random.uniform(0,1) > epsilon:
                action = np.argmax(Q_table[state[0]][state[1]])
            else:
                action = env.action_space.sample()

            next_state , reward , done , _ = env.step(action)
            score += reward
            second_part = learning_rate*(reward+Gamme*np.max(Q_table[next_state[0]][next_state[1]] ))
            Q_table[state[0]][state[1]][action] = (1 - learning_rate)* Q_table[state[0]][state[1]][action] + second_part
            # Q_table[state,action] = (1 - learning_rate)* Q_table[state,action] + learning_rate*(reward+Gamme*np.max(Q_table[next_state,:]))

            state = next_state
        if epsilon >= epsilon_min:
            epsilon*= epsilon_decay
    t = Q_table.tolist()
    for i in range(len(t)):

        tmp = [list(map(lambda x:round(x,4) , y)) for y in t[i]]
        print(tmp)

if __name__ == "__main__":
    matrix_chain = matrixChain()
    QRL(matrix_chain)
