import os
import torch
from AbstractAgent import AbstractAgent
from nle.nethack import ACTIONS

import gym
from gym import logger as gymlogger

from setup import colab_setup
from train import train

# probably should be argparse.argparser
hyper_params = {
    "learning-rate": 0.00048,  # learning rate for RMSprob
    "discount-factor": 0.99,  # discount factor
    "max_steps": int(1e6),  # total number of steps to run the environment for
    "hidden_size": 256,  # number of transitions to optimize at the same time
    "learning-starts": 10000,  # number of steps before learning starts
    "num_step_td_update": 5,  # number of iterations between every optimization step
    "seed": 10,  # number of iterations between every optimization step


    "verbose": True, # whether to print or not
    "print-freq":1, # how frequent to print the results (in epochs)
    
    "save-freq":2, # how frequent to save the model (in epochs)
    "save-name":'ac_2', # the name of the ac to save it by
    "save-dir":'Nethack-AC', # the name of the directory to to save models
    "model_name": None, # to continue trainining, set this to the model name or -1 to train from last model

    "env": "NetHackScore-v0", # the name of the environment
    "env_seed": 10 # environment random seed to use during training
    "observation_keys": ("glyphs", "blstats"), # the observation keys to use in the environment
    "actions": ACTIONS, # set the actions to use, default = all,
    "num_envs": 8,
    
    
    "colab": False #whether this is running in colab or not

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), # if GPU is available, use that, preffered

    "plot_too": False, "whehter to create plots of the rewards after training"
}


# Set up the colab and save directory ## move to function??
if hyper_params["colab"]:
    # setup colab by installing nle libraries and dependancies
    colab_setup()

    # Imports specifically for google colab/drive
    from google.colab import files, drive
    drive.mount('/content/gdrive')
    hyper_params['save-dir'] = os.path.join(
        '/content/gdrive/My Drive/Colab Notebooks/', hyper_params['save-dir'])

os.makedirs(hyper_params['save-dir'], exist_ok=True) # if save dir doesn't exist, create it
path = lambda fname: os.path.join(hyper_params['save-dir'], fname) # gives full save path for file name


def make_env(seed, rank):
    def _thunk():
        env = gym.make(env_name)
        env.seed(seed+rank)
        return env

    return _thunk

def main():
    # create the gym environmet to use
    env = gym.make(
        hyper_params['env'],
        observation_keys=hyper_params['observation_keys'],
        actions = hyper_params['actions']
    )
    # set a seed --good for comparison and all
    env.seed(hyper_params['env_seed'])

    # train the model
    train(env, agent, hyper_params)
    


    if hyper_params['verbose']: 
        print("Done!!!")



if __name__ == "__main__":
    main()