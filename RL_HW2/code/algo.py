import numpy as np
import math
from abc import abstractmethod

class QAgent:
	def __init__(self,state_shape, action_shape,grid_num,discount_factor=0.99,learing_rate=0.1,epsilon=1,decay_epsilon=0.1):
		"""
        初始化 QAgent 类，设置 Q 表的维度和各种超参数。
        
        参数:
            state_shape: 状态空间的形状
            action_shape: 动作空间的大小
            learning_rate: 学习率
            discount_factor: 折扣因子 gamma
            epsilon: epsilon-greedy 策略中的初始探索率
            epsilon_decay: epsilon 衰减率
            epsilon_min: epsilon 最小值
        """
		self.state_shape = state_shape
		self.action_shape = action_shape
		self.gamma = discount_factor
		self.grid_num = grid_num

		self.Q = np.zeros((grid_num,grid_num,action_shape))
		# self.Q = np.zeros(())#定义Q表
		
		self.lr = learing_rate

		self.epsilon = epsilon
		self.decay_epsilon = decay_epsilon
		self.decay_epsilon_min = 0.01
	def select_action(self, ob) -> int:
		"""
        使用 epsilon-greedy 策略选择动作。
        
        参数:
            ob: 当前的状态
            
        返回:
            action: 选择的动作
        """
		x, y = map(int, ob)
		# epsilon-greedy 策略：以 epsilon 的概率随机选择动作，1-epsilon 的概率选择最大 Q 值的动作
		if np.random.rand() < self.epsilon:
			return np.random.randint(0,self.action_shape)
		else:
			return np.argmax(self.Q[x,y])

	def update(self, ob, action, reward, ob_next, done):
		"""
        更新 Q 表，根据当前状态、动作、奖励和下一个状态来计算目标值并进行更新。
        
        参数:
            ob: 当前的状态
			action: 选择的动作
			reward: 奖励
			ob_next: 下一个状态
			done: 是否结束
        """
		#计算公式: Q(s,a) = Q(s,a) + lr * (r + gamma * max(Q(s',a')) - Q(s,a))
		x, y = map(int, ob_next)
		if done: 
			self.Q[x,y] = reward
			return
		else:
			max_next_q = np.max(self.Q[x,y])
			target = reward + self.gamma * max_next_q
			self.Q[x,y,action] += self.lr * (target - self.Q[x,y,action])
	def epsilon_decay(self,num):
		"""
		衰减epsilon值
		"""
		self.epsilon = self.decay_epsilon_min + (self.epsilon-self.decay_epsilon_min) * math.exp(-1 * 10^-5 *num)