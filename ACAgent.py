import torch
from ACNetwork import ACNetwork

from importlib import reload
reload(sys.modules['ACNetwork'])
from ACNetwork import ACNetwork


def format_observations(observation, keys=("glyphs", "blstats")):
    
    '''convert the observation from env to the -> glyphs_matrix,around_agent, blstats'''
    
    observations = {}
    for key in keys:
        entry = observation[key]
        entry = torch.from_numpy(entry)
        entry = entry.view((1, 1) + entry.shape)  # (...) -> (T,B,...).
        observations[key] = entry
    return observations


class ACAgent:
    def __init__(self, observation_space, action_space):
        """Loads the agent"""
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = ACNetwork(observation_space, self.action_space.n)

    def act(self, observation):
        # Perform processing to observation

        # get the distribution over actions for state and the value of the state
        state = torch.FloatTensor(state).to(device)
        dist, value = self.model(state)
        
        # sample an action from the distribution
        action = dist.sample()

        return action.item() # action.cpu().numpy()
