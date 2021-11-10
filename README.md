# ppo_torch
A modular and minimal implementation of Proximal Policy Optimization (PPO) using PyTorch. This purpose of this repo is to provide an implementation of PPO that gives a satisfactory performance on most of the OpenAI gym environments, while being easy to understand and build upon. To make this implementation easier to understand an explanation of the code structure is provided below and a tutorial is provided to better understand the usage of each module.

Since this is a minimal implementation, it has the following limitations:

- given the need to interact with gym environments, all of the calculations are performed on CPU (GPU support is planned).
- only one worker is used (also planning for multi-worker, but I need to better understand how it works).


## Code Structure
There are total of 6 files, with the following dependecy structure

    train.py
    ├──ppo.py
    │  ├──actor.py
    │  │  └──mlp.py
    │  └──critic.py
    │     └──mlp.py
    └──replay_buffer.py

- `mlp.py` defines `mlp()` which creates a neural network with given sizes and activation functions
- `actor.py` creates the actor model for both continuous and discrete action space
- `critic.py` creates the critic model
- `ppo.py` defines the actor-critic model and has the utility function to save and load the model, as well as calculate the loss function for actor and critic
- `replay_buffer.py` creates the replay buffer, and has the utility function to compute the generalized advantage estimation (GAE)
- `train.py` defines the training loop, the update loop and testing function

## Citing 
Please use this bibtex if you want to cite this repository in your publications :

    @misc{ppo_torch,
        author = {Dai, Bolun},
        title = {Modular & Minimal PyTorch Implementation of Proximal Policy Optimization},
        year = {2021},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/BolunDai0216/ppo_torch}},
    }
