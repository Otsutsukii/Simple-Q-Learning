import numpy as np
import random

reward = np.array([[-1, -1, -1, -1,  0, -1],
                   [-1, -1, -1,  0, -1, 100],
                   [-1, -1, -1,  0, -1, -1],
                   [-1,  0,  0, -1,  0, -1],
                   [ 0, -1, -1,  0, -1, 100],
                   [-1,  0, -1, -1,  0, 100]])


Q_matrix = np.zeros([6, 6])
gamma = 0.8
s_final = 5

for i in range(100):
    # Q_matrix[1, 5] = 100
    s_init = random.randint(0, 5)
    # print(s_init)
    # s_init = 3
    actions = np.where(reward[s_init] != -1)[0]
    random.shuffle(actions)
    state_prev = s_init
    state_now = actions[0]  # 5
    # state_now = 1

    count = 0

    for i in range(1000):
        actions = np.where(reward[state_now] != -1)[0]
        reward_max, action_max = -1, -1
        for action in actions:
            if Q_matrix[state_now, action] > reward_max:
                reward_max = Q_matrix[state_now, action]
                action_max = action

        Q_matrix[state_prev, state_now] = reward[state_prev, state_now] + \
                                          gamma * Q_matrix[state_now, action_max]
        state_prev = state_now
        state_now = action_max

    # while((state_prev != s_final) and count < 1000):
    #     count += 1
    #     actions = np.where(reward[state_now] != -1)[0]
    #     reward_max, action_max = -1, -1
    #     for action in actions:
    #         if Q_matrix[state_now, action] > reward_max:
    #             reward_max = Q_matrix[state_now, action]
    #             action_max = action
    #     print(action)
    #     Q_matrix[state_prev, state_now] = reward[state_prev, state_now] + \
    #                                       gamma * Q_matrix[state_now, action_max]
    #     state_prev = state_now
    #     state_now = action_max

print(Q_matrix)

