#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential
from keras.layers import Dense
import random
from collections import deque

import matplotlib.pyplot as plt

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
		self.env_name = environment_name


	def make_model(self):
		if self.env_name == 'CartPole-v0':
			model = Sequential()
			model.add(Dense(64, activation = 'tanh', input_shape=(4,)))
			model.add(Dense(64, activation = 'tanh', input_shape=(64,)))
			model.add(Dense(2, activation = 'tanh', input_shape=(64,)))
			
		else:
			# put the architecture for mountain car here
			pass
		model.compile(loss=keras.losses.mean_squared_error,
				optimizer=tf.train.AdamOptimizer(learning_rate =0.001),
				metrics=['accuracy'])
		return model

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights. 
		pass

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
		batch = []
		for idx in random_indices:
			batch.append(self.replay_memory[idx])
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
		self.model = QNetwork(environment_name).make_model()
		self.memory_size = 50000
		self.burn_in = 10000
		self.batch_size = 32
		self.number_iterations = 10000
		self.env = gym.make(environment_name)
		self.epsilon = 0.5
		self.epsilon_step = 0.45**(-5)
		if environment_name == 'CartPole-v0':
			self.gamma = 0.99
		else:
			self.gamma = 1
		self.replay_mem = Replay_Memory(memory_size=self.memory_size, burn_in=self.burn_in)


	def anneal_epsilon(self):
		factor = 2
		print('epsilon before ', self.epsilon)
		self.epsilon = self.epsilon/factor
		print('epsilon after', self.epsilon)

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from. 
		if np.random.binomial(n=1, p=self.epsilon):
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

	def get_x_y(self, batch):
		y = []
		x = []
		for current_state, action, reward, next_state, done in batch:
			# for a terminal state
			q_value_current_state = self.model.predict(np.array(current_state, ndmin=2))[0]
			q_value_next_state = self.model.predict(np.array(next_state, ndmin=2))[0]
			target = q_value_current_state
			if done:
				target[action] = reward
			else:
				#target[max_action_idx] = reward + self.gamma * max(q_value_next_state)
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
		num_episodes = 5000
		loss = []
		acc = []
		mean_episode_rewards_c = []
		rewards_per_episode = []
		loss_per_episode = []
		acc_per_episode = []
		for i in range(num_episodes):
			done = False
			state = self.env.reset()
			c = 0
			r_c = []
			while not done:
				c += 1
				q_values = self.model.predict(np.array(state, ndmin=2))
				action = self.epsilon_greedy_policy(q_values)
				next_state, reward, done, info = self.env.step(action)
				#self.env.render()
				transition = [state, action, reward, next_state, done]
				state = next_state
				self.replay_mem.append(transition)
				batch = self.replay_mem.sample_batch(self.batch_size)
				x, y = self.get_x_y(batch)
				
				assert len(x) == len(y)
				history = self.model.fit(x, y, epochs=1, batch_size=self.batch_size, verbose=0)
				l = history.history['loss'][-1]
				a =  history.history['accuracy'][-1]
				loss.append(l)
				acc.append(a)
				r_c.append(reward)
				self.epsilon = self.epsilon - self.epsilon_step
			rewards_per_episode = sum(r_c)
			loss_per_episode = np.mean(loss)
			acc_per_episode = np.mean(acc)
			
			print('episode = %d | step = %d | loss = %f | acc = %f | epsilon = %f | mean_reward per episode = %f '%(i, c, loss_per_episode, acc_per_episode, self.epsilon, rewards_per_episode))

		return [np.mean(loss_per_episode), np.mean(acc_per_episode), np.mean(rewards_per_episode)]
			
			
			


	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		rewards_per_episode = []
		num_episodes = 20
		for i in range(num_episodes):
			done = False
			state = self.env.reset()
			c = 0
			r_c = []
			while not done:
				c += 1
				q_values = self.model.predict(np.array(state, ndmin=2))
				action = self.greedy_policy(q_values)
				next_state, reward, done, info = self.env.step(action)
				state = next_state
				#self.env.render()
				r_c.append(reward)
				sum_rewards = sum(r_c)
			rewards_per_episode.append(sum_rewards)
			print('Inside TESTING --> episode = %d/%d | steps = %d | episode reward = %f | epsilon = %f'%(i, num_episodes, c, sum_rewards, self.epsilon))
		#self.env.close()	
		
		return np.mean(rewards_per_episode)

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
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.epsilon_greedy_policy(state, 0.05)
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

def plot_graphs(rewards):
	plt.figure(figsize=(12,3))
	plt.title('test rewards')
	plt.plot(loss)
	plt.xlabel('iterations')
	plt.ylabel('rewards')

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
	env_name = 'CartPole-v0' 
	dqn = DQN_Agent(env_name)
	dqn.burn_in_memory()

	num_iterations = 100
	loss_all = []
	acc_all = []
	rewards_all = []

	# test
	rewards_all_test = []
	anneal_every = [10, 15, 35, 50]
	for i in range(num_iterations):
		[mean_loss, mean_acc, mean_rewards] = dqn.train()
		print('TRAINIGN 100 episode --> iteration = %d/%d | loss = %f | accuracy = %f | reward = %f'%(i, num_iterations, mean_loss, mean_acc, mean_rewards))
		# test code here
		test_every = 1
		if i%test_every == 0:
			mean_rewards_per_100_episodes = dqn.test()
			print('TESTING --> iteration = %d/%d | mean 20 episode reward = %f'%(i, num_iterations, mean_rewards_per_100_episodes))
			rewards_all_test.append(mean_rewards_per_100_episodes)
		if i in anneal_every:
			dqn.anneal_epsilon()	



	plot_graphs(rewards_all_test)



if __name__ == '__main__':
	main(sys.argv)

