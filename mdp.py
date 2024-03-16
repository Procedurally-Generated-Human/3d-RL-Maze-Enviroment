import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt


class Maze3dEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, map, render_mode=None) -> None:
        self.map = map
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=0, high=50, shape=(60,), dtype=np.float64)
        self.action_space = gym.spaces.Discrete(4)
    
    def reset(self, seed=None):
        info = {}
        self.posx, self.posy, self.rot = 1.5, 1.5, np.pi/4
        self.exitx, self.exity = 3, 3
        observation = np.zeros(60)
        for i in range(60):
            rot_i = self.rot + np.deg2rad(i-30)
            x, y = self.posx, self.posy
            sin, cos = 0.02*np.sin(rot_i), 0.02*np.cos(rot_i)
            n = 0
            
            while 1:
                x, y, n = x + cos, y + sin, n +1
                if self.map[int(x)][int(y)]:
                    h = 1/(0.02*n)
                    observation[i] = h
                    break
        return observation, info
    

    def step(self, action):
        obs = np.zeros(60)
        reward = -0.01
        terminated = False
        info = {}

        x, y = (self.posx, self.posy)

        if action == 0: #up
            x, y = (x + 0.3*np.cos(self.rot), y + 0.3*np.sin(self.rot))
        elif action == 1: #down
            x, y = (x - 0.3*np.cos(self.rot), y - 0.3*np.sin(self.rot))
        elif action == 2: #left
            self.rot = self.rot - np.pi/8
        elif action == 3: #right
            self.rot = self.rot + np.pi/8

        if mapa[int(x)][int(y)] == 0:
            if int(self.posx) == self.exitx and int(self.posy) == self.exity:
                reward = 1
                terminated = True
            self.posx, self.posy = (x, y)
        
        for i in range(60):
            rot_i = self.rot + np.deg2rad(i-30)
            x, y = self.posx, self.posy
            sin, cos = 0.02*np.sin(rot_i), 0.02*np.cos(rot_i)
            n = 0
            
            while 1:
                x, y, n = x + cos, y + sin, n +1
                if self.map[int(x)][int(y)]:
                    h = 1/(0.02*n)
                    obs[i] = h
                    break

        if self.render_mode == "human":
            self._render_frame(obs)
        return obs, reward, terminated, False, info
        

    def _render_frame(self, obs):
        idx = 0
        for line in obs:
            plt.vlines(idx, -line, line, lw=8)
            idx += 1
        plt.axis('off'); plt.tight_layout(); plt.axis([0, 60, -1, 1])
        plt.show()
  


    #def close():


from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

mapa = [[1, 1, 1, 1, 1],
      [1, 0, 0, 0, 1],
      [1, 0, 1, 0, 1],
      [1, 0, 0, 0, 1],
      [1, 1, 1, 1, 1]]
ss = Maze3dEnv(map=mapa)
model = PPO("MlpPolicy", ss, verbose=1)
model.learn(total_timesteps=100000, log_interval=4)
ss2 = Maze3dEnv(map=mapa, render_mode="human")
obs, info = ss2.reset(seed=None)
for i in range(100):
    action = model.predict(obs)
    print(action[0])
    obs, rew, _, _, _ = ss2.step(action[0])


