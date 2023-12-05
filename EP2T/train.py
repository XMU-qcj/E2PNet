import os
import json
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from EP2T import EP2T

EP2T_model = EP2T().cuda()
data = torch.randint(0, 255, (6, 8192, 4)).float().cuda()
print(data.shape)
res = EP2T_model(data)
print(res.shape)