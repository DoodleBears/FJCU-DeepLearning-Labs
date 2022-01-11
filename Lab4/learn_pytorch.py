import torch
from torch.utils.data import Dataset
 
# 這兩個是資料處裡常用的套件
import numpy as np
import pandas as pd


if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    print(z)