import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

# local import 
from data import scale_max_dict, scale_min_dict, get_test_data, get_test_dataloaders
from model import SalesForecastNet

def inference(model, test_loader, device):
    predictions = []

    with torch.no_grad():
        for X in tqdm(iter(test_loader)):
            X = X.to(device)

            output = model(X)

            output = output.cpu().numpy()

            predictions.extend(output)

    return np.array(predictions)
