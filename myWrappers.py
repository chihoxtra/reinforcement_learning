import gym
import torch
import copy
import numpy as np
from collections import deque
from torchvision import transforms

class StackEnv():
    """ Virtual Wrapper """

    def __init__(self, env, input_shape=(84,84), depth=4, skipframe=4):

        self.env = env #the parent class of env
        self.input_shape = input_shape #dimension of the output
        self.depth = depth #how many frames per stack
        self.max_skipframe = 4 #how many frame to skip

        # changing the size and and gray scale the input
        self.encode_states = transforms.Compose([
                             transforms.ToPILImage(),
                             transforms.Grayscale(num_output_channels=1),
                             transforms.Resize(self.input_shape),
                             transforms.ToTensor()
                             ])

        self.oddEvenStack = True #do we combine odd and even frames

        self.states_stack = deque([torch.zeros(self.input_shape) for z in range(self.depth)],
                            maxlen=self.depth)

        #for odd even frame combination
        self.last_state = np.zeros(self.env.observation_space.shape, dtype=np.uint8)


    def state2StackedInputs(self, current_state):
        encoded_state = self.encode_states(current_state) #210x160x3 -> 1x84x84

        #combine the max element between this frame and last frame
        last_encoded_state = self.encode_states(self.last_state)

        merged_states = torch.stack((encoded_state.squeeze(), last_encoded_state.squeeze()), dim=0).max(dim=0)[0]
        self.states_stack.append(merged_states) #last in the stack and first is out hence

        #update last state
        self.last_state = copy.deepcopy(current_state)

        #convert to tensor for input
        state_inputs = torch.stack([self.states_stack[i] for i in range(len(self.states_stack))]).numpy()

        return state_inputs


    def reset(self):
        # reset the parent env
        initial_state = self.env.reset()
        state_inputs = self.state2StackedInputs(initial_state)

        return state_inputs

    def step(self, action):
        #skip thru frames and sum rewards; get last states and done
        reward_hist = []
        for i in range(self.max_skipframe):
            next_state, reward, done, _ = self.env.step(action)

            # get previous frame as last frame
            if i == self.max_skipframe - 2:
                self.last_state = next_state
            # merge last key frame in stack with previous frame, add to stack
            elif i == self.max_skipframe - 1:
                next_state_inputs = self.state2StackedInputs(next_state)
            #append awards
            reward_hist.append(reward)
        return next_state_inputs, sum(reward_hist), done
