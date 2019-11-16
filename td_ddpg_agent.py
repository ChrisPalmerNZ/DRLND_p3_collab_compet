import numpy as np
import random
import copy
from collections import namedtuple, deque

from td_ddpg_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

POLICY_FREQ = 2         # Frequency of delay (twin delayed DDPG)

BUFFER_SIZE = int(1e6)  # replay buffer size                     - default int(2e5), try to int(1e6)
BATCH_SIZE = 1024       # minibatch size                         - default 128 
GAMMA = 0.99            # discount factor                        - default 0.99
TAU = 5e-3              # for soft update of target parameters   - default 1e-3
LR_ACTOR = 1e-3         # learning rate of the actor             - default 1e-3 (i.e. Adam default)
LR_CRITIC = 1e-3        # learning rate of the critic            - default 1e-3
OPTIM = 'Adam'          # optimizer to use                       - default is Adam, experiment with AdamW?
WEIGHT_DECAY = 0        # L2 weight decay                        - default for Adam = 0
AWEIGHT_DECAY = 0.01    # L2 weight decay for AdamW              - default for AdamW = 0.01
AMSGRAD = False         # AMSGrad variant of optimizer           - default False
LEAKINESS = 0.01        # leakiness, leaky_relu used if > 0      - default for leaky_relu is 0.01
USEKAIMING = False      # kaiming normal weight initialization   - default False

# Suggested on slack:
LEARN_EVERY = 1         # learning timestep interval (20 for the Continuous Reacher task, 1 for Tennis)
LEARN_NUM   = 1         # number of learning passes  (10 for the Continuous Reacher task, 1 for Tennis)
GRAD_CLIPPING = 1       # Gradient Clipping                      - default 1

# Ornstein-Uhlenbeck noise parameters
OU_SIGMA  = 0.01        # 0.1 # default 0.2
OU_THETA  = 0.15        # default 0.15
NOISE_CLIP = 0.5        # Noise clipping

# 
EPSILON       = 1.0     # for epsilon in the noise process (act step)
EPSILON_DECAY = 1e-6    # 1e-6    # decay rate (learn step), default 1e-6, 0 for no decay
# End - suggested on slack

USE_CUDA = True
device = torch.device("cuda:0" if USE_CUDA and torch.cuda.is_available() else "cpu")

NOISE_FUNC = 0           # What noise function? 0 = Ornstein-Uhlenbeck process, 1 = Gaussian Noise

# Initialize all random seeds
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA and torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

class ReplayBuffer(object):
    """Replay buffer, storing transitions/experience tuples"""

    def __init__(self, buffer_size=1000000, batch_size=100):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        # maximum size of memory
        self.buffer_size = buffer_size
        # Initialize stored experiences memory
        self.memory = deque(maxlen=self.buffer_size)  # internal memory (deque)
        # batch size
        self.batch_size = batch_size
        # Experience named tuple
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process.
        Params
        ======
            mu (float)    : long-running mean
            theta (float) : speed of mean reversion
            sigma (float) : volatility parameter
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        # Below approach does not work at all with TD3 - but with DDPG it works better than the (classic) approach above
        # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, max_action):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int):  dimension of each state
            action_size (int): dimension of each action
            max_action (int): Actors use this to get back to original dimensionality
        """
        # Actor NN
        self.actor_local = Actor(state_size, action_size, max_action, leakiness=LEAKINESS).to(device)
        self.actor_target = Actor(state_size, action_size, max_action, leakiness=LEAKINESS).to(device)
        self.actor_target.load_state_dict(self.actor_target.state_dict())
        if OPTIM == "AdamW" and hasattr(optim,'AdamW'):
            self.actor_optimizer = optim.AdamW(self.actor_local.parameters(), lr=LR_ACTOR)
        else:    
            self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters())

        # Critic NN
        self.critic_local = Critic(state_size, action_size, leakiness=LEAKINESS).to(device)
        self.critic_target = Critic(state_size, action_size, leakiness=LEAKINESS).to(device)
        self.critic_target.load_state_dict(self.critic_target.state_dict())
        if OPTIM == "AdamW" and hasattr(optim,'AdamW'):
            self.critic_optimizer = torch.optim.AdamW(self.critic_local.parameters(), lr=LR_ACTOR)
        else:    
            self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=LR_ACTOR)
                
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action       
        
        # Noise function: Ornstein-Uhlenbeck process
        # Depending on NOISE_FUNC we may use Gaussian Noise instead 
        self.noise = OUNoise(action_size, SEED)

        self.epsilon = EPSILON

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
    
    def select_action(self, state):
        state = torch.Tensor(state.reshape(1,-1)).to(device)
        # Need numpy format for noise addition and clipping
        return self.actor_local(state).cpu().data.numpy().flatten()

    def step(self, states, actions, rewards, next_states, dones, timestep, num_agents):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        for i in range(num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestep % LEARN_EVERY == 0:
            for _ in range(LEARN_NUM):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, timestep)
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += np.clip((self.epsilon * self.noise.sample()),-NOISE_CLIP, NOISE_CLIP)

        return np.clip(action, -self.max_action, self.max_action)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, timestep):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            timestep (needed for TD3 as it updates actor by timestep % POLICY_FREQ)
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.forward(next_states)
     
        # For the 2 Critic Targets, take (s', a') as inputs and return Q-values Qt1(s', a') and Qt2(s', a') 
        Q1_targets_next, Q2_targets_next = self.critic_target(next_states, actions_next)
        
        # Use the minimum of the 2 Q-values
        Q_targets_next = torch.min(Q1_targets_next, Q2_targets_next)   

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Critic models take the tuple (s, a) as input and return two Q values Q1(s, a) and Q2(s, a)
        Q1_expected, Q2_expected = self.critic_local.forward(states, actions)
 
        # Compute loss from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
        critic_loss = F.mse_loss(Q1_expected, Q_targets) + F.mse_loss(Q2_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # gradient clipping for critic
        if GRAD_CLIPPING > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIPPING)
        
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Once every (delay - default two) iterations, update the Actor model with gradient ascent on output of first Critic model
        if timestep % POLICY_FREQ == 0:
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local.Q1(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)    

            # --------------------- and update epsilon decay ----------------------- # 
            if EPSILON_DECAY > 0:                
                self.epsilon -= EPSILON_DECAY
                self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

