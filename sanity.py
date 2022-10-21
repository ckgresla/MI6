# Test Performance of an Algorithm on Toy Example -- to be used whilst in development 

import torch

from mi6.core.tests import * 
from mi6.algorithms.REINFORCE import *


cartpole(REINFORCE, render=True)

