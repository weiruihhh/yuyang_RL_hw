import numpy as np
import random

# state_shape = 2
# action_shape = 4
# Q = np.zeros((state_shape, action_shape))
# print(Q)
# ob_next = [5.,5.]
# print(tuple(ob_next))
grid_num = 4
action_shape = 4
Q = np.zeros((grid_num,grid_num,action_shape))

print(Q[2,2])

print(random.randint(0,action_shape))
