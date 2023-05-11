import pickle
import torch
import random
# specify the file path
file_path = "sim_scores.pkl"
with open(file_path, 'rb') as f:
    serialized_object = torch.load(f)
    print(serialized_object)