from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np

# for gradient clipping
clip_value = 1.0

def train(model, optimizer, train_dataloader, val_dataloader, device, scheduler, args):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    best_loss = 9999999
    best_model = None

    for epoch in range(1, args.epochs):
        model.train()
        train_loss = []
        train_mae = []
        for X, Y in tqdm(iter(train_dataloader)):
            X = X.to(device)
            Y = Y.to(device)

            # Foward
            optimizer.zero_grad()
            # get prediction
            output = model(X)

            loss = criterion(output, Y)

            # back propagation

            loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            # Perform LR scheduler Work
            if scheduler is not None:
                scheduler.step()

            train_loss.append(loss.item())

        val_loss = validation(model, val_dataloader, criterion, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}]')

        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model
            print('Model Saved')
            
    return best_model, best_loss

def validation(model, val_dataloader, criterion, device):
    model.eval()
    val_loss = []

    with torch.no_grad():
        for X, Y in tqdm(iter(val_dataloader)):
            X = X.to(device)
            Y = Y.to(device)

            output = model(X)
            loss = criterion(output, Y)

            val_loss.append(loss.item())
    return np.mean(val_loss)