import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time

import torch
from torch import distributed as dist
import torch.nn as nn



# for visualization with fixed text prompts, during finetuning

noise = torch.randn(2, 4, 64, 64)
torch.save(noise, f'noise_2.pt')

noise = torch.randn(9, 4, 64, 64)
torch.save(noise, f'noise_9.pt')

noise = torch.randn(16, 4, 64, 64)
torch.save(noise, f'noise_16.pt')
