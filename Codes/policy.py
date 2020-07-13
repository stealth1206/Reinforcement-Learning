import gym
import numpy as np
import matplotlib.pyplot as plt
import copy 

#Taking input from the users
print("Enviorment : ")
env_name = input()
print("Number of Iterations : ")
NUM_EPISODES = int(input())
print("Discount Factor : ")
GAMMA = float(input())
print("Batch Size : ")
Batch_size = int(input())
print("Experiment Type : ")
exp_type = str(input())

# Setting up for the options choosen
dsa = False     		# dsa : Advantage-Normalisation
rtg = False				# rtg : Rewards-to-go
if exp_type == 'dsa':
	dsa = True
elif exp_type == 'rtg':
	rtg = True

#Hyperparameters
LEARNING_RATE = 0.005

# Create gym and seed numpy
env = gym.make(env_name)
nA = env.action_space.n
nS = env.observation_space
np.random.seed(1)

flag = 8		# For coping with dimension of two different enviorments
itr = 1000
if nA == 2:
	flag = 4
	itr = 200

# Initializing weight
w = np.random.rand(flag, nA)

# Our policy that maps state to action parameterized by w
def policy(state,w):
	h = state.dot(w)
	expo = np.exp(h)
	return expo/np.sum(expo)

# Vectorized softmax Jacobian
def softmax_grad(softmax):
	s = softmax.reshape(-1,1)
	return np.diagflat(s) - np.dot(s, s.T)

# Function to calculate Advantage-Normalisation of rewards-to-go values
def advantage_normalisation(rewards) :
	discounted_rewards =[]
	base = [None]*itr
	for i in range(len(rewards)):    # Finding rewards-to-go
		r=np.ones_like(rewards[i])
		running_add = 0
		for t in reversed(range(0, len(rewards[i]))):
			running_add = running_add * GAMMA + rewards[i][t]
			r[t] = running_add
		discounted_rewards.append(r)
	
	for i in range(itr):	
		sum=0
		for j in range(len(discounted_rewards)):
			if(i<len(discounted_rewards[j])):
				sum=sum+discounted_rewards[j][i]
		base[i]=sum/Batch_size 				# Calculating baseline
	for i in range(len(discounted_rewards)):
		for j in range(len(discounted_rewards[i])):
			discounted_rewards[i][j]-= base[j]
		discounted_rewards[i] -= np.mean(discounted_rewards[i])  # Normalising the rewards-to-go
		discounted_rewards[i] /= np.std(discounted_rewards[i])
		#discounted_rewards[i]+=2	

	return discounted_rewards


# Main loop and intializing main loop variables
r = []
g = []
e_r = []
batch = []
mean = []
for e in range(NUM_EPISODES):

	state = env.reset()[None,:]

	grads = []	
	rewards = []	
	# Keep track of game score to print
	score = 0

	while True:

		#env.render()

		# Sample from policy and take action in environment
		probs = policy(state,w)
		action = np.random.choice(nA,p=probs[0])
		next_state,reward,done,_ = env.step(action)
		next_state = next_state[None,:]

		# Compute gradienprint("2D random values array : \nt and save with reward in memory for our weight updates
		dsoftmax = softmax_grad(probs)[action,:]
		dlog = dsoftmax / probs[0,action]
		grad = state.T.dot(dlog[None,:])

		grads.append(grad)
		rewards.append(reward)		
		#count = count +1
		score+=reward

		# update your old state to the new state
		state = next_state

		if done:
			break
	r.append(rewards)
	g.append(grads)
	# Weight update using advantage normalisation
	if dsa and (e+1)%Batch_size == 0 and e>0 :
		r = advantage_normalisation(r)
		#print(r)
		for i in range(len(g)):
			for j in range(len(g[i])):
				w += LEARNING_RATE * g[i][j] * r[i][j]	
			
		r = []
		g = [] 
	#weight update using rewards-to-go
	elif rtg :
		for i in range(len(grads)):
			w += LEARNING_RATE * grads[i] * sum([ r * (GAMMA ** t) for t,r in enumerate(rewards[i:])])
	# Simple weight updates
	else :
		for i in range(len(grads)):
			w += LEARNING_RATE * grads[i] * sum([ r * (GAMMA ** t) for t,r in enumerate(rewards[0:])])

		
	print("EP: " + str(e) + " Score: " + str(score))
	# Append for logging and print
	e_r.append(score) 
	# finding average rewards
	if e%200 == 0 and e>0:
            mean.append(np.mean(e_r))
            batch.append(e)
            e_r = [] 

# Printing the plots
plt.plot(batch,mean,'r',label="Mean of Rewards")
plt.title('Advantage Normalisation Batch Size = 64')
plt.xlabel('Time_Steps')
plt.ylabel('Mean Reward')
plt.legend()
plt.show()
env.close()