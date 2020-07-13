import random
import gym
import numpy as np
from IPython.display import clear_output

print('Enviorment : ')
input1 = str(input())
env = gym.make(input1).env
print('State space : '+ str(env.observation_space))
print('Action space : ' + str(env.action_space))
state = env.reset()
temp = []

done = False

while not done:		 # Generating random states and their actions with corresponding rewards
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    temp.append({
        'state': state,
        'action': action,
        'reward': reward
        }
    )

for i in range(2):
	print(temp[i])