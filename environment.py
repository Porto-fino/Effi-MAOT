from __future__ import division
import gym
import numpy as np
from cv2 import resize
from gym.spaces.box import Box
import distutils.version
from random import choice
import gym_unrealcv

def create_env(env_id, args, rank=-1): 

    env = gym.make(env_id) # id = UnrealMCRoomLarge-DiscreteColorGoal-v5 /////'UnrealUrbanTree-DiscreteColorGoal-v1'
    print ('build env')

    if args.rescale is True:  #满足
        env = Rescale(env, args) #rescale of (240,320,3)

    if 'img' in args.obs:  #True 
        env = UnrealRescale(env, args) #rescale of (3,80,80)
    return env

class Rescale(gym.Wrapper):
    def __init__(self, env, args):
        super(Rescale, self).__init__(env)
        if type(env.observation_space) == list: # [Box(240,320,3),Box(240,320,3)] True
            self.mx_d = 255.0
            self.mn_d = 0.0
            shape = env.observation_space[0].shape #(240,320,3)
        # else:
        #     self.mx_d = env.observation_space.high
        #     self.mn_d = env.observation_space.low
        #     shape = env.observation_space.shape
        self.obs_range = self.mx_d - self.mn_d #255
        self.new_maxd = 1.0
        self.new_mind = -1.0
        self.observation_space = [Box(self.new_mind, self.new_maxd, shape) for i in range(len(self.observation_space))] #Box内每一维数据范围是[-1,1]
        self.args = args
        self.inv_img = self.choose_rand_seed() #true random true / false
        self.flip = self.choose_rand_seed() #false

    def rescale(self, x):
        obs = x.clip(self.mn_d, self.mx_d)
        new_obs = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                    ) / self.obs_range) + self.new_mind
        return new_obs

    def _reset(self):
        ob = self.env.reset()

        self.flip = self.choose_rand_seed()
        if self.flip:
            ob = self.flip_img(ob)

        self.inv_img = self.choose_rand_seed()
        if self.inv_img:
            ob = 255 - ob
        ob = self.rescale(np.float32(ob))
        return ob

    def _step(self, action):
        if self.flip:
            action = self.flip_action(action)
        ob, rew, done, info = self.env.step(action)
        if self.inv_img:
            ob = 255 - ob
        if self.flip:
            ob = self.flip_img(ob)
        ob = self.rescale(np.float32(ob))
        return ob, rew, done, info

    def choose_rand_seed(self):
        return choice([True, False])

    def flip_img(self, img):
        return np.fliplr(img)

    def flip_action(self, action):
        ac = action
        if action == 0:
            ac = 1
        elif action == 1:
            ac = 0
        elif action == 2:
            ac = 3
        elif action == 3:
            ac = 2
        return ac

class UnrealRescale(gym.ObservationWrapper):
    def __init__(self, env, args):
        gym.ObservationWrapper.__init__(self, env)

        self.input_size = args.input_size  #80
        self.use_gym_10_api = distutils.version.LooseVersion(gym.__version__) >= distutils.version.LooseVersion('0.10.0')

        if self.use_gym_10_api: ###
            self.observation_space = [Box(-1.0, 1.0, [3, self.input_size, self.input_size], dtype=np.uint8) for i in range(len(self.observation_space))] # 3 80 80
        # else:
        #     self.observation_space = [Box(-1.0, 1.0, [3, self.input_size, self.input_size]) for i in range(len(self.observation_space))]

    def process_frame_ue(self, frame, size=80):

        frame = frame.astype(np.float32)
        frame = resize(frame, (size, size))
        frame = np.transpose(frame, (2, 1, 0))

        return frame

    def observation(self, observation): #observation (2, 240, 320, 3)
        obses = []
        for i in range(len(observation)):
            obses.append(self.process_frame_ue(observation[i], self.input_size)) #resize
        return np.array(obses)