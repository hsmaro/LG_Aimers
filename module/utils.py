import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Warmup Scheduler
class WarmupLR(optim.lr_scheduler.LambdaLR):

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_end_steps: int,
        last_epoch: int = -1,
    ):

        def wramup_fn(step: int):
            if step < warmup_end_steps:
                return float(step) / float(max(warmup_end_steps, 1))
            return 1.0

        super().__init__(optimizer, wramup_fn, last_epoch)


# Inference
def inference(model, test_loader, device):
    predictions = []

    with torch.no_grad():
        for X in tqdm(iter(test_loader)):
            X = X.to(device)

            output = model(X)

            output = output.cpu().numpy()

            predictions.extend(output)

    return np.array(predictions)