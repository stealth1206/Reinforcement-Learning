from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import gym
import matplotlib.pyplot as plt

def prepro(I):       # preprocessing of image : 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
  
  I = I[35:195]               # crop
  I = I[::2,::2,0]            # downsample by factor of 2
  I[I == 144] = 0             # erase background (background type 1)
  I[I == 109] = 0             # erase background (background type 2)
  I[I != 0] = 1               # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r, gamma):         # Finding the discounted rewards
  r = np.array(r)
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):    # we go from last reward to first one so we don't have to do exponentiations
    if r[t] != 0: running_add = 0             # if the game ended, reset the reward sum
    running_add = running_add * gamma + r[t]  
    discounted_r[t] = running_add
  discounted_r -= np.mean(discounted_r)     #normalizing the result
  discounted_r /= np.std(discounted_r)     
  return discounted_r

# creates a generic neural network architecture
model = Sequential()
# hidden layer takes a pre-processed frame as input, and has 200 units
model.add(Dense(units=200,input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))   # output layer
# compile the model using traditional Machine Learning losses and optimizers
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Intialization of the enviorment using gym
env = gym.make("Pong-v0")
observation = env.reset()
prev_input = None

up_action = 2
down_action = 3

# Hyperparameters
gamma = 0.99

# initialization of variables used in the main loop
x_train, y_train, rewards = [],[],[]
reward_sum = 0
episode_nb = 0
r = []
mean = []
best = []
batch = []
# main loop
while (True):

    # preprocess the observation, set input as difference between images
    cur_input = prepro(observation)
    x = cur_input - prev_input if prev_input is not None else np.zeros(80 * 80)
    prev_input = cur_input
    
    # forward the policy network and sample action according to the proba distribution i.e. exploration and exploitation
    proba = model.predict(np.expand_dims(x, axis=1).T)
    action = up_action if np.random.uniform() < proba else down_action
    y = 1 if action == 2 else 0

    # log the input and label to train later
    x_train.append(x)
    y_train.append(y)

    observation, reward, done, info = env.step(action)
    rewards.append(reward)
    reward_sum += reward
    r.append(reward_sum)
    # end of an episode
    if done:
        print('At the end of episode', episode_nb, 'the total reward was :', reward_sum)
        
        episode_nb += 1
    
        # training
        model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=1, sample_weight=discount_rewards(rewards, gamma))

        # taking the average rewards
        if episode_nb%200 == 0 and episode_nb>0:
            mean.append(np.mean(r))
            batch.append(episode_nb)
            best.append(np.amax(r))
            r = []

        if episode_nb == 10000:
            break

        # Reinitialization
        x_train, y_train, rewards = [],[],[]
        observation = env.reset()
        reward_sum = 0
        prev_input = None

#Printing the plots
plt.plot(batch,mean,'r',label="Mean of Rewards")
plt.plot(batch,best,'b',label="Best of Rewards")
plt.title('Pong Learning Curve')
plt.xlabel('Time_Steps')
plt.ylabel('Mean Reward for Batch of size 200')
plt.legend()
plt.show()
env.close()
