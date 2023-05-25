import gym
import numpy as np
import gym_donkeycar
import torch
from typing import Any, Dict, Optional, Tuple
from vae.vae import VAE



class VAEWrapper(gym.Wrapper):
    def __init__(self,env:gym.Env):
        super().__init__(env)
        self.vae = VAE()
        model_path = 'vae_generated_track.pth'
        self.vae.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

        self.observation_space = gym.spaces.Box(low=-np.inf,high=np.inf,shape=(32,),dtype=np.float32)
    def convert(self,obs):
        #VAEのモデルに投げるため画像を変換するメソッド
        obs_ = np.array(obs).reshape((160,120,3))
        obs_ = obs_[0:160,40:120,:].reshape((1,80,160,3))
        obs_ = torch.from_numpy(obs_).permute(0,3,1,2).float()

        

        return obs_

    def convert_state_vae(self,img):
        #入力画像をEncodeし、32次元の潜在変数に変換するメソッド
        state = self.convert(img)
        state,_,_ = self.vae.encode(state)
        state /= 255.0
        #state_ = state_.clone().detach().cpu().numpy()
        return state.detach().numpy()

    def reset(self) ->np.ndarray:
        obs = self.env.reset()
        obs = self.convert_state_vae(obs)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs,reward,done,infos = self.env.step(action)
        obs = self.convert_state_vae(obs)


        return obs,reward,done,infos



