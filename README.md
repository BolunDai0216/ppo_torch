# ppo_torch
A modular and minimal implementation of Proximal Policy Optimization (PPO) using PyTorch. This purpose of this repo is to provide an implementation of PPO that gives a satisfactory performance on most of the OpenAI gym environments, while being easy to understand and build upon. To make this implementation easier to understand an explanation of the code structure is provided below and a tutorial is provided to better understand the usage of each module.

Since this is a minimal implementation, it has the following limitations:

- given the need to interact with gym environments, all of the calculations are performed on CPU (GPU support is planned).
- only one worker is used (also planning for multi-worker, but I need to better understand how it works).


## Code Structure
