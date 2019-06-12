import math

import numpy as np
from gym import spaces

from envs.DishRackEnv import rack_lower, rack_upper
from reality.RealEnv import RealEnv


class RealDishRackEnv(RealEnv):
    observation_space = spaces.Box(np.array([-3.] * 7 + [-float('inf')] * 3 + [rack_lower[2]]),
                                   np.array([3.] * 7 + [float('inf')] * 3 + [rack_upper[2]]),
                                   dtype=np.float32)
    state_to_estimate = [7, 8, 9, 10]
    normalize_low = rack_lower
    normalize_high = rack_upper
