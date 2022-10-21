# Test Performance of an Algorithm on Toy Example -- to be used whilst in development 

import torch

from mi6.core.tests import * 
from mi6.algorithms.REINFORCE import REINFORCE as algorithm
#from mi6.algorithms.VPG import VPG as algorithm #let's nuke these and return to a better implementation later, w consistent agent objects


cartpole(algorithm, render=True)

