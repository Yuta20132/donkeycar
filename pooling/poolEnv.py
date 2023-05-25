import gym
import numpy as np
import gym_donkeycar
import torch
from typing import Any, Dict, Optional, Tuple



#プーリングを行う
#38400次元から4056次元
class PoolWrapper(gym.Wrapper):
    def __init__(self,env:gym.Env):
        super().__init__(env)
        
        self.observation_space = gym.spaces.Box(low=0,high=255,shape=(39,52,3),dtype=np.uint8)
    def convert(self,image):
        n = 5
        m = 3
        pool_size = (n,n)
        pool_strides = (m,m)
        pool = np.zeros((int((image.shape[0]-pool_size[0])/pool_strides[0])+1,
                 int((image.shape[1]-pool_size[1])/pool_strides[1])+1, 
                 image.shape[2]))

        for i in range(pool.shape[0]):
            for j in range(pool.shape[1]):
                for k in range(pool.shape[2]):
                    pool[i, j, k] = np.max(image[i*pool_strides[0]:i*pool_strides[0]+pool_size[0], 
                                          j*pool_strides[1]:j*pool_strides[1]+pool_size[1], k])

        

        return pool
    def reset(self) ->np.ndarray:
        obs = self.env.reset()
        obs = self.convert(obs)
        return obs

    

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs,reward,done,infos = self.env.step(action)
        obs = self.convert(obs)


        return obs,reward,done,infos