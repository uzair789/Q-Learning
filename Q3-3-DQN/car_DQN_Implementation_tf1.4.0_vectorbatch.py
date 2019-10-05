#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential
from keras.layers import Dense
import random
from collections import deque

import matplotlib.pyplot as plt

import os

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
		self.env_name = environment_name
		if self.env_name == 'CartPole-v0':
			lr = 0.001
			self.model = Sequential()
			self.model.add(Dense(64, activation = 'tanh', input_shape=(4,)))
			self.model.add(Dense(128, activation = 'tanh', ))
			self.model.add(Dense(128, activation = 'tanh', ))
			self.model.add(Dense(64, activation = 'tanh'))
			self.model.add(Dense(2))
			
		else:
			# put the architecture for mountain car here
			self.model = Sequential()
			self.model.add(Dense(64, activation = 'tanh', input_shape=(2,)))
			self.model.add(Dense(128, activation = 'tanh', ))
			self.model.add(Dense(64, activation = 'tanh'))
			self.model.add(Dense(3))
			lr = 0.001

		self.model.compile(loss=keras.losses.mean_squared_error,
				optimizer=tf.train.AdamOptimizer(learning_rate =lr),
				metrics=['accuracy'])
		
	def get_model(self):
		return self.model


	def save_model_weights(self, suffix):
		# Helper function to save your model / weights. 
		self.model.save_weights(suffix+'_model.h5')

	def load_model(self, model_file):
		# Helper function to load an existing model.
		# e.g.: torch.save(self.model.state_dict(), model_file)
		pass

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		# e.g.: self.model.load_state_dict(torch.load(model_file))
		pass


class Replay_Memory():

	def __init__(self, memory_size, burn_in):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		
		# Hint: you might find this useful:
		# 		collections.deque(maxlen=memory_size)
		self.memory_size = memory_size
		self.replay_memory = deque(maxlen=memory_size)
		
	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		random_indices = np.random.randint(len(self.replay_memory), size=batch_size)
		#batch = []
		#for idx in random_indices:
		#	batch.append(self.replay_memory[idx])

		batch = [self.replay_memory[idx] for idx in random_indices]

		assert len(batch) == batch_size
		return np.array(batch)	

	def append(self, transition):
		# Appends transition to the memory. 	
		self.replay_memory.extend([transition])

	def get_size(self):
		return len(self.replay_memory)


class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.env_name = environment_name
		self.net = QNetwork(self.env_name)
		self.model = self.net.get_model()
		self.memory_size = 70000
		self.burn_in = 10000
		self.batch_size = 32
		self.number_episodes = 5000
		self.env = gym.make(self.env_name)
		self.epsilon = 0.5
		self.epsilon_step = 0.45 * pow(10,-5)
		self.test_every = 100
		if self.env_name == 'CartPole-v0':
			self.gamma = 0.99
		else:
			self.gamma = 1
		self.replay_mem = Replay_Memory(memory_size=self.memory_size, burn_in=self.burn_in)


	def epsilon_greedy_policy(self, q_values, eps=None):
		# Creating epsilon greedy probabilities to sample from.
		if eps:
			p = eps
		else:
			p = self.epsilon 


		if np.random.binomial(n=1, p=p):
			# sample random action
			action = self.env.action_space.sample()
		else:
			#sample max action
			action = self.greedy_policy(q_values)

		return action


	def action_to_one_hot(self, action):
		action_vec = np.zeros(self.env.action_space.n)
		action_vec[action] = 1
		return action_vec

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		action = np.argmax(q_values)
		return action

	def get_training_data(self, batch):
		y = []
		x = []
		for current_state, action, reward, next_state, done in batch:
			q_value_current_state = self.model.predict(np.array(current_state, ndmin=2))[0]
			q_value_next_state = self.model.predict(np.array(next_state, ndmin=2))[0]
			target = q_value_current_state
			if done:
				target[action] = reward
			else:
				target[action] = reward + self.gamma * max(q_value_next_state)
			y.append(target)
			x.append(current_state)	
	
		return [np.array(x), np.array(y)]		

	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# When use replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		print('Replay Memory size', self.replay_mem.get_size())
		stop_training = False
		loss = []
		acc = []
		video_points = [1, int(self.number_episodes*1/3), int(self.number_episodes*2/3), int(self.number_episodes*3/3)]
		
		mean_test_rewards = []
		training_loss = []
		for i in range(1,self.number_episodes+1):

			#generate the teset video
			if i in video_points:
				print(i, video_points, "Saving Video!!")
				test_video(self, self.env_name, i)


			done = False
			state = self.env.reset()
			c = 0
			state_rewards = []
			while not done:
				c += 1
				q_values = self.model.predict(np.array(state, ndmin=2))
				action = self.epsilon_greedy_policy(q_values)
				next_state, reward, done, info = self.env.step(action)
				transition = [state, action, reward, next_state, done]
				state = next_state
				self.replay_mem.append(transition)
				batch = self.replay_mem.sample_batch(self.batch_size)
				x, y = self.get_training_data(batch)
				
				assert len(x) == len(y)
				if not stop_training:	
					history = self.model.fit(x, y, epochs=1, batch_size=self.batch_size, verbose=0)
				loss.append(history.history['loss'][-1])
				acc.append(history.history['acc'][-1])
				state_rewards.append(reward)
				
				self.epsilon = max(self.epsilon - self.epsilon_step, 0.05)
			rewards_per_episode = (sum(state_rewards))
			loss_per_episode = (np.mean(loss))
			acc_per_episode = (np.mean(acc))
			training_loss.append(loss_per_episode)
			print('episode = %d | step = %d | loss = %f | acc = %f | reward per episode = %d | epsion = %f | stop_training = %s'%(i, c, loss_per_episode, acc_per_episode, rewards_per_episode, self.epsilon, stop_training))


			if i%self.test_every == 0:
				episode_rewards = self.test(num_episodes=20)
				mean_test_rewards.append(np.mean(episode_rewards))
				print('TESTING --> iteration = %d/%d | mean test reward over 20 episodes = %f | epsilon = %f | stop_training = %s' %(i, self.number_episodes,np.mean(episode_rewards), self.epsilon,stop_training))
				
				if len(mean_test_rewards)>=5:
					mean_test_rewards_avg = np.mean(mean_test_rewards[-5:])
					if mean_test_rewards_avg >= 190:
						stop_training = True
					else:
						stop_training = False		
				
		self.net.save_model_weights('car_64_128_64')	
		return [training_loss, mean_test_rewards]


	def test(self, num_episodes):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		episode_rewards = []
		for i in range(num_episodes):
			done = False
			state = self.env.reset()
			c = 0
			state_rewards = []
			while not done:
				c += 1
				q_values = self.model.predict(np.array(state, ndmin=2))
				action = self.greedy_policy(q_values)
				next_state, reward, done, info = self.env.step(action)
				#self.env.render()
				state = next_state
				state_rewards.append(reward)
				sum_rewards = sum(state_rewards)
			episode_rewards.append(sum_rewards)
			#print('Inside TESTING --> episode = %d/%d | steps = %d | episode reward = %d | epsilon = %f'%(i, num_episodes, c, sum_rewards, self.epsilon))
		#self.env.close()	
		
		return episode_rewards

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		transition_count = 0
		print(self.burn_in)
		while(transition_count <= self.burn_in):
			current_state = self.env.reset()
			done = False
			if not done and transition_count <= self.burn_in:
				action = self.env.action_space.sample() 
				next_state, reward, done, info = self.env.step(action)
				transition = [current_state, action, reward, next_state, done]
				current_state = next_state
				transition_count += 1
				self.replay_mem.append(transition)
		print('Done burn in')
		print(transition_count)

# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop. 
def test_video(agent, env, epi):
	# Usage: 
	# 	you can pass the arguments within agent.train() as:
	# 		if episode % int(self.num_episodes/3) == 0:
    #       	test_video(self, self.environment_name, episode)
    save_path = "./results/videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        q_values = agent.model.predict(np.array(state, ndmin=2))
        action = agent.epsilon_greedy_policy(q_values, 0.05)
        next_state, reward, done, info = env.step(action)
        state = next_state
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    agent.env.close()


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()

def plot_graph(data, title, xlabel, ylabel):
	plt.figure(figsize=(12,5))
	plt.title(title)
	plt.plot(data)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig('results/'+title+'.png')

def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it.
	env_name = 'MountainCar-v0' 
	dqn = DQN_Agent(env_name)
	dqn.burn_in_memory()

	[training_loss, mean_test_rewards] = dqn.train()


	plot_graph(training_loss, 'training_loss_for_'+env_name, 'episodes', 'training loss')
	plot_graph(mean_test_rewards, 'Mean_test_rewards_20_'+env_name, 'iterations', 'Mean rewards for 20 episodes')
	
	final_test_rewards = dqn.test(num_episodes=3)

	plot_graph(final_test_rewards, 'final_test_rewards_for_'+env_name, 'episodes', 'Test Rewards')



if __name__ == '__main__':
	main(sys.argv)

