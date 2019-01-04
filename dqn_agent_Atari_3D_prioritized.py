import numpy as np
import random
from collections import namedtuple, deque

from model_Atari_3D_prioritized_duel import QNetwork
#from model_Atari_3D import QNetwork
#from model_fc_unity import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)        # replay buffer size
BATCH_SIZE = 32               # minibatch size
REPLAY_MIN_SIZE = int(5e3)    # min len of memory before replay start
GAMMA = 0.99                  # discount factor
TAU = 1e-3                    # for soft update of target parameters
LR = 1e-4                     # learning rate #25e4
UPDATE_EVERY = 16             # how often to update the network
MIN_DECAY_STEP = int(1e5)     # decay start from this step
DECAY_STEP = int(1e4)         # how many steps before another decay: priority-> random
DECAY_GAMMA = 0.9995          # LR decay by how much
TD_ERROR_EPS = 1e-5           # make sure TD error is not zero
PRIORITY_DISCOUNT = 0.8       # balance between prioritized and random sampling #0.7
SAMPLING_BIAS = 0.6           # adjustment on weight update #0.5
USE_DUEL = True               # use duel network?
REWARD_CLIP = True            # use reward clipping?
ERROR_CLIP = False            # clip error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, USE_DUEL).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, USE_DUEL).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, REWARD_CLIP)

        # Keep track of repeated actions
        self.last_actions = deque(maxlen=10)

        # Initialize time step (for updating every UPDATE_EVERY steps and others)
        self.t_step = 0

        # object reference to constant values:
        self.p_discount = PRIORITY_DISCOUNT
        self.sampling_bias = SAMPLING_BIAS
        self.reward_clip = REWARD_CLIP
        self.error_clip = ERROR_CLIP
        print("current device: {}".format(device))


    def get_TD_values(self, local_net, target_net, s, a, r, ns, d, isLearning=False):

        ns, s = ns.float(), s.float()
        with torch.no_grad(): #for sure no grad for this part
            #print(type(ns))
            ns_target_vals = target_net(ns.float().to(device))

            #0:the value, 1: argmax, unsqueeze to match the side of TD current
            ns_target_max_val = ns_target_vals.max(dim=1)[0]
            ns_target_max_val = ns_target_max_val.unsqueeze(dim=-1)

            assert(ns_target_max_val.requires_grad == False)

            td_targets = r + ((1-d) * GAMMA * ns_target_max_val)

        if isLearning: # if it is not under learning mode
            local_net.train()
            td_currents_vals = local_net(s.float().to(device))

            td_currents = torch.gather(td_currents_vals, 1, a.to(device))
        else:
            local_net.eval()
            with torch.no_grad():
                td_currents_vals = local_net(s.to(device))

                td_currents = torch.gather(td_currents_vals, 1, a.to(device))

        local_net.train() #resume training for local network

        return td_targets, td_currents


    def step(self, state, action, reward, next_state, done):
        # internal rourtine
        def toBatchDim(v):
            return torch.from_numpy(v).unsqueeze(0)

        # get the td values to compute td errors
        td_target, td_current = self.get_TD_values(self.qnetwork_local,
                                                   self.qnetwork_target,
                                                   toBatchDim(state),
                                                   toBatchDim(np.array(action)).reshape(1,1),
                                                   reward,
                                                   toBatchDim(next_state),
                                                   done,
                                                   isLearning=False)

        # store the abs magnitude of td error, add eps to make sure it is non-zero
        td_error = torch.abs(td_target - td_current).cpu().numpy()

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, td_error)

        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1

        # decay only starts after certain threshold
        if self.t_step > MIN_DECAY_STEP and self.t_step % DECAY_STEP == 0:
            #self.optimizer.state_dict()['param_groups'][0]['lr'] *= DECAY_GAMMA
            self.p_discount *= DECAY_GAMMA
            self.sampling_bias *= DECAY_GAMMA

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > REPLAY_MIN_SIZE:
            experiences, sampling_adj = self.memory.sample(self.p_discount,
                                                           TD_ERROR_EPS)
            self.learn(experiences, sampling_adj, GAMMA, SAMPLING_BIAS)

        if self.t_step % UPDATE_EVERY == 0:
            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            action =  np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))

        # try to avoid too many zeros actions
        if len(self.last_actions) == 10 and np.sum(self.last_actions) == 0 and action == 0:
            action = random.choice(np.arange(self.action_size))
            print('\rall zeros!\r')

        return action

    def learn(self, experiences, sampling_adj, gamma, sampling_bias):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        td_targets, td_currents = self.get_TD_values(self.qnetwork_local,
                                                     self.qnetwork_target,
                                                     states, actions,
                                                     rewards, next_states,
                                                     dones,
                                                     isLearning=True)


        if self.error_clip: #error clipping
            td_error = td_targets - td_currents
            td_error_clipped = td_error.clamp(min=-1,max=-1)
            loss = (td_error_clipped**2).mean()
        else:
            loss = F.mse_loss(td_currents, td_targets).to(device) #element wise

        # adjust the loss for sampling bias by priority
        reduced_adj = torch.from_numpy(np.array(sampling_adj)).float()
        adjusted_loss = loss * (reduced_adj ** self.sampling_bias)

        self.optimizer.zero_grad()
        adjusted_loss.backward()
        # update the parameters
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        #self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, reward_clip=False):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            priority_factor (float): discount factor for priority sampling
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action",
                                     "reward", "next_state", "done", "td_error"])
        self.seed = random.seed(seed)
        self.reward_clip = reward_clip

    def add(self, state, action, reward, next_state, done, td_error):
        """Add a new experience to memory."""
        if self.reward_clip:
            reward = reward/10.0 #reward clipping
            #reward = max(min(reward, 1.0), -1.0) #reward clipping

        e = self.experience(np.expand_dims(state,0), action, reward,
                            np.expand_dims(next_state,0), done, td_error)
        self.memory.append(e)

    def sample(self, p_discount,td_eps):
        """Sample a batch of experiences from memory."""
        #experiences = random.sample(self.memory, k=self.batch_size)
        td_list = np.array([e[5] for e in self.memory]) # a list of td_error
        td_list = (td_list ** p_discount) + td_eps #apply discount factor and make sure not zero

        p_dist = td_list/np.sum(td_list) #normalized probability
        p_dist = p_dist.squeeze() #p_dist is a 1D array

        assert((np.sum(p_dist) - 1) <  1e-4)
        assert(len(p_dist) == len(self.memory))

        # get sample of index from the p distribution
        sample_ind = np.random.choice(len(self.memory), self.batch_size, p=p_dist)

        experiences = [self.memory[i] for i in sample_ind]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        # for weight update adjustment
        selected_td_p = [p_dist[i] for i in sample_ind]
        sampling_adj = (1./np.sum(selected_td_p)) * (1./len(self.memory))

        return (states, actions, rewards, next_states, dones), sampling_adj


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
