import gym.spaces
import numpy as np

env = gym.make("Taxi-v3")

state_space = env.observation_space.n 
action_space = env.action_space.n 

Q_table = np.zeros((state_space,action_space))

epsilon = 1.0  # greedy 100% 
epsilon_min = 0.005 #minimum greedy 0.05%
epsilon_decay = 0.99993 # decay multiplied with epsilon after each episode 

episodes = 50000  #amount of games 
max_steps = 100 #maximum steps per episode 
learning_rate = 0.65 
Gamme = 0.65 

for episode in range(episodes):
    state = env.reset()
    done = False 
    score = 0 
    for _ in range(max_steps):
        if np.random.uniform(0,1) > epsilon:
            action = np.argmax(Q_table[state,:])
        else:
            action = env.action_space.sample()
        
        next_state , reward , done , _ = env.step(action)
        score += reward

        Q_table[state,action] = (1 - learning_rate)* Q_table[state,action] + learning_rate*(reward+Gamme*np.max(Q_table[next_state,:]))

        state = next_state
    if epsilon >= epsilon_min:
        epsilon*= epsilon_decay
print(Q_table.tolist())