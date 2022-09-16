# Agent Focused RL

A pocket-repo for RL stuff; Algorithms, pre-trained Agents and Pointers to information


## Hit-List
Algorithms yet to be implemented
1. REINFORCE
2. VPG
  - one with the schulman Generalized Advantage Estimation (VPG) & one w/o (REINFORCE) -- technically same algorithm but slightly different
3. TBD


## Info
Directory Structure
- "algorithms" contains PyTorch Implementations of each method (has a test use case if file is run @ the command line, intended to be imported thought)
- "agents" contains saved model weights for Environments 
  - Save New Weights/Controllers as roughly: "{ENV_NAME}-{ALGORITHM}-{STEPS}"
- "core" has utilities and code that is shared across Algorithms (environment interaction, etc.)
- "housekeeping" has all of the items for... housekeeping (requirements.txt for env replication, etc.)
- "intel" has some notes and links to helpful resources, papers and information


## Setup
Mainly working with Torch & Gym through Conda install, but see following for full reqs install w pip:
- `pip install -r housekeeping/requirements.txt` -- on CPU
- `pip install -r housekeeping/requirements-gpu.txt` -- on GPU (check your cuda!)


## Links + Resources
Useful stuff to refer to
- OpenAI Gym Docs- https://www.gymlibrary.dev/ (this link somehow rarely comes up if google this)
- SeungEunRho's minimalRL- https://github.com/seungeunrho/minimalRL (great torch implementations)
- garage- https://github.com/rlworkgroup/garage (toolkit for RL research, formerly called "rllab")
- ray- https://github.com/ray-project/ray (RL for Industry, great documentation)
- OpenAI SpinningUp- https://spinningup.openai.com/en/latest/index.html (nice to reference, actually should just read the papers)
  - Github- https://github.com/openai/spinningup 

