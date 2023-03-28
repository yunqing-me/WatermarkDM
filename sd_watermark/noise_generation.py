import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time

import torch
from torch import distributed as dist
import torch.nn as nn




noise = torch.randn(2, 4, 64, 64)
torch.save(noise, f'noise_2.pt')


# project_in = nn.Sequential(
#     nn.Linear(3, 2),
#     nn.GELU()
# )
# combo_net = nn.Sequential(
#     project_in,
#     nn.Dropout(),
#     nn.Linear(2, 1)
# )

# print(combo_net)