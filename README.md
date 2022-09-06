# RL for Wizards

A pocket-repo for RL stuff; Algorithms, pre-trained Agents and Pointers to information


## Info
Directory Structure
- "algorithms" contains PyTorch Implementations of each method (ideally importable and well commented, with useful helpers like saving weights)
- "agents" contains saved model weights for Environments 
  - Save New Weights/Controllers as roughly: "{ENV_NAME}-{ALGORITHM}-{STEPS}"
- "housekeeping" has all of the items for... housekeeping (requirements.txt for env replication, etc.)


## Setup
Mainly working with Torch & Gym through Conda install, but see following for full reqs install w pip:
- `pip install -r housekeeping/requirements.txt` -- on CPU
- `pip install -r housekeeping/requirements-gpu.txt` -- on GPU (check your cuda!)


## Links + Resources
- OpenAI Gym Docs- https://www.gymlibrary.dev/
- SeungEunRho's minimalRL- https://github.com/seungeunrho/minimalRL (great torch implementations of the Classic Stuff)



