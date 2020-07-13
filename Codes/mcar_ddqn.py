import random
import gym
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt


class Agent:
	def __init__(self, state_size, action_size):             #intialising the agent
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95     # discount rate
		self.epsilon = 1.0    # exploration rate
		self.epsilon_min = 0.001
		self.epsilon_decay = 0.997
		self.train_start = 1000
		self.learning_rate = 0.5
		self.batch_size = 64
		self.model = self._build_model()
		self.target_model = self._build_model()
		#self.update_target_model()

	def _build_model(self): 	# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(64, input_dim=2))
		model.add(Activation('relu'))

		model.add(Dense(64))
		model.add(Activation('relu'))

		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss="mean_squared_error",
					  optimizer=Adam(lr=self.learning_rate))
		return model

	# def update_target_model(self):	# copy weights from model to target_model

	# 	self.target_model.set_weights(self.model.get_weights())

	def rem_tuple(self, state, action, reward, next_state, done,a):		# keeping memory of tuple of state,action,reward,next_state 
		self.memory.append((state, action, reward, next_state, done,a))

	def act(self, state):							# Gives action on the basis of random choice i.e exploration or exploitation
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size),np.random.randint(2)
		else :
			if np.random.random()<0.5:
				return np.argmax(self.model.predict(state)[0]),0
			else:
				return np.argmax(self.target_model.predict(state)[0]),1

	def replay(self):							# Fitting the model by finding Q-values and target Q-values
		if len(self.memory) < self.train_start:
			return
		minibatch = random.sample(self.memory, self.batch_size)
		update_input = np.zeros((self.batch_size, self.state_size))
		update_target = np.zeros((self.batch_size, self.action_size))
		for i in range(self.batch_size):
			state, action, reward, next_state, done,a = minibatch[i]
			if a==0:
				target = self.model.predict(state)[0]
				target = np.reshape(target,[3,1])
				if done:
					target[action] = reward
				else:
					Q_future  = self.model.predict(next_state)[0]
					target[action] = reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(Q_future)]
				self.model.fit(state,target, epochs=1, verbose=0)
			elif a==1:
				target = self.target_model.predict(state)[0]
				target = np.reshape(target,[3,1])
				if done:
					target[action] = reward
				else:
					Q_future  = self.target_model.predict(next_state)[0]
					target[action] = reward + self.gamma * self.model.predict(next_state)[0][np.argmax(Q_future)]
				print(target)
				self.target_model.fit(state,target, epochs=1, verbose=0)
		# 		update_input[i] = state
		# 		update_target[i] = target	
		# self.model.fit(update_input, update_target,batch_size=self.batch_size, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def save(self, name):		# Saving the model
		self.model.save(name)


EPISODES = 1001      # number of epsiodes(iterations)
STEP = 200			 # number of steps in each episode
def main():
	env = gym.make('MountainCar-v0')				# importing the MountainCar-v0 enviorment
	state_size = env.observation_space.shape[0]			
	action_size = env.action_space.n 				# number of actions defined
	agent = Agent(state_size, action_size)		# Importing the agent
	print('state size:' ,state_size)
	print('action size: ', action_size)
	done= False
	average = []
	best = []
	batch = []
	r = []
	for episode in range(EPISODES):
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		score = 0
		for step in range(STEP):
			#env.render()
			print(agent.act(state))
			action,aa= agent.act(state)
			next_state, reward, done, info = env.step(action)
			next_state = np.reshape(next_state, [1, state_size])
			agent.rem_tuple(state, action, reward, next_state, done,aa)     # saving the tuple into memory
			score += reward
			state = next_state
			if done: 
				# agent.update_target_model()			# updating the target model after each step
				break
			agent.replay()					# Fitting the model
		r.append(score)						
		# Test every 100 episodes
		if episode%100 == 0:				# saving the model
			print('saving the model')
			agent.save("mountain_car-dqn.h5")
			
			
		if episode%100 == 0 and episode> 10:									# Testing of episodes
			best.append(np.amax(r))
			r = []
			model = load_model('mountain_car-dqn.h5')
			total_reward = 0
			for i in range(10):
				state = env.reset()
				state = np.reshape(state, [1, state_size])
				for j in range(STEP):
					#env.render()
					action = model.predict(state)				# predicting from the trained model
					action = np.argmax(action)
					z_state,reward,done,info = env.step(action)
					z_state = np.reshape(z_state, [1, state_size])
					state=z_state
					total_reward += reward
					if done:
						break
			ave_reward = total_reward/10		#claculating average reward
			average.append(ave_reward)
			batch.append(episode)
			#print('episode: ',episode,'Evaluation Average Reward:',ave_reward)

	# printing the plots 		
	plt.plot(batch,average,'r',label="Mean of Rewards")
	plt.plot(batch,best,'b',label="Best of Rewards")
	plt.title('MountainCar-v0')
	plt.xlabel('Time_Steps')
	plt.ylabel('Mean Reward')
	plt.legend()
	plt.show()
	env.close()			
if __name__ == "__main__":
	main()
	