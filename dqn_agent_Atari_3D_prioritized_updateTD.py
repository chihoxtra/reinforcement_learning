import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

from model_Atari_3D_prioritized_duel import QNetwork
#from model_Atari_3D import QNetwork
#from model_fc_unity import QNetwork

"""
This version updated with TD error updates and corrected weight adjustment
"""

BUFFER_SIZE = int(1e5)        # replay buffer size
BATCH_SIZE = 32               # minibatch size
REPLAY_MIN_SIZE = BATCH_SIZE  # min len of memory before replay start #int(5e3)
GAMMA = 0.99                  # discount factor
TAU = 1e-3                    # for soft update of target parameters
LR = 1e-4                     # learning rate #25e4
UPDATE_EVERY = 8              # how often to update the network
MIN_DECAY_STEP = int(1e5)     # decay start from this step
DECAY_STEP = int(1e4)         # how many steps before another decay: priority-> random
DECAY_GAMMA = 0.9995          # LR decay by how much
TD_ERROR_EPS = 1e-5           # make sure TD error is not zero
P_REPLAY_ALPHA = 0.8          # balance between prioritized and random sampling #0.7
P_REPLAY_BETA = 0.5           # adjustment on weight update #0.5
LEARNING_LOOP = 2             # number of learning cycle per step
USE_DUEL = True               # use duel network? V and A?
USE_DOUBLE = True             # use double network to select TD value?
REWARD_SCALE = False          # use reward clipping?
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
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, REWARD_SCALE)

        # Keep track of repeated actions
        self.last_actions = deque(maxlen=10)

        # Initialize time step (for updating every UPDATE_EVERY steps and others)
        self.t_step = 0

        # object reference to constant values:
        self.p_replay_alpha = P_REPLAY_ALPHA
        self.p_replay_beta = P_REPLAY_BETA
        self.reward_scale = REWARD_SCALE
        self.error_clip = ERROR_CLIP
        print("current device: {}".format(device))
        print("use duel network (a and v): {}".format(USE_DUEL))
        print("use double network: {}".format(USE_DOUBLE))

    def get_TD_values(self, local_net, target_net, s, a, r, ns, d, isLearning=False):

        ###### TD TARGET #######
        s, ns = s.float(), ns.float() #to satisfy the network requirement
        with torch.no_grad(): #for sure no grad for this part

            ns_target_vals = target_net(ns.float().to(device))

            #0:the value, 1: argmax, unsqueeze to match the side of TD current
            if USE_DOUBLE:
                # use local values + target's pick for new target value
                ns_target_max_arg = ns_target_vals.max(dim=1)[1]
                ns_target_max_arg = ns_target_max_arg.unsqueeze(dim=-1).to(device)
                ns_local_vals = local_net(ns.float().to(device))

                ns_target_max_val = torch.gather(ns_local_vals, 1, ns_target_max_arg)
            else:
                ns_target_max_val = ns_target_vals.max(dim=1)[0]
                ns_target_max_val = ns_target_max_val.unsqueeze(dim=-1)

            assert(ns_target_max_val.requires_grad == False)

            td_targets = r + ((1-d) * GAMMA * ns_target_max_val)

        ###### TD CURRENT #######
        if isLearning: # if it is under learning mode need backprop
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
        """ handle memory update, learning and target network params update"""
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
            self.p_replay_alpha *= DECAY_GAMMA
            self.p_replay_beta = min(self.p_replay_beta/DECAY_GAMMA, 1)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > REPLAY_MIN_SIZE:
            for i in range(LEARNING_LOOP):
                experiences, weight, ind = self.memory.sample(self.p_replay_alpha,
                                                              self.p_replay_beta,
                                                              TD_ERROR_EPS)
                self.learn(experiences, weight, ind, GAMMA)

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

    def learn(self, experiences, weight, ind, gamma):
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
            td_errors = td_targets - td_currents
            td_errors_clipped = td_error.clamp(min=-1,max=-1)
            loss = (td_errors_clipped**2).mean()
        else:
            loss = F.mse_loss(td_currents, td_targets).to(device) #element wise

        # adjust the loss for sampling bias by priority
        adjusted_loss = loss * weight

        self.optimizer.zero_grad()
        adjusted_loss.backward()
        # update the parameters
        self.optimizer.step()

        # update the td error in memory
        with torch.no_grad():
            td_errors_update = np.array([torch.abs(td_targets - td_currents).cpu().numpy()])
            self.memory.update(td_errors_update, ind)

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

    def __init__(self, action_size, buffer_size, batch_size, seed, reward_scale=False):
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
        self.reward_scale = reward_scale

    def add(self, state, action, reward, next_state, done, td_error):
        """Add a new experience to memory."""
        if self.reward_scale:
            reward = reward/10.0 #scale reward by factor of 10
            #reward = max(min(reward, 1.0), -1.0) #reward clipping

        e = self.experience(np.expand_dims(state,0), action, reward,
                            np.expand_dims(next_state,0), done, td_error)
        self.memory.append(e)

    def update(self, td_errors, index):
        """ update the td error values while restoring orders"""
        #td_errors = td_errors.squeeze()
        original_len = len(self.memory)
        for i in range(len(index)):
            self.memory.rotate(-index[i])
            e = self.memory.popleft()
            #print(e.td_error - td_errors[:,i,:])
            td_updated = td_errors[:,i,:]

            e_update = self.experience(e.state, e.action, e.reward,
                                       e.next_state, e.done, td_updated)

            self.memory.appendleft(e_update)
            self.memory.rotate(index[i])
        assert(original_len == len(self.memory))


    def sample(self, p_replay_alpha, p_replay_beta, td_eps):
        """Sample a batch of experiences from memory."""
        #experiences = random.sample(self.memory, k=self.batch_size)
        td_list = np.array([e[5] for e in self.memory]) # a list of td_error

        td_list = (td_list ** p_replay_alpha) + td_eps #apply discount factor and make sure not zero

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
        # checker: the mean of selected TD errors should be greater than the mean of overall TD errors
        #print(np.mean([e.td_error for e in experiences]), np.mean([e.td_error for e in self.memory]))
        normalizer = torch.from_numpy(np.array(max(selected_td_p))).float()
        adjustment = ((1./np.mean(selected_td_p)) * (1./len(self.memory)))
        adjustment = torch.from_numpy(np.array(adjustment)).float()
        weight = adjustment ** p_replay_beta
        weight =  weight/normalizer

        return (states, actions, rewards, next_states, dones), weight, sample_ind


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
