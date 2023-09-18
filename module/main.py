# Libary import
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import argparse
import logging

# Local import 
from data import get_train_val_data, get_test_data, get_test_dataloaders, CustomDataset
from model import SalesForecastNet
from train import train, validation
from utils import WarmupLR, inference

# import parrallel distribution
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "8888"

def main(rank, world_size, args):
    # 분산환경 초기화
    dist.init_process_group("gloo", rank=rank, world_size=world_size) # gloo, nccl, mpi
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_data = pd.read_csv("./data/train.csv")
    train_data.drop(["ID", "제품"], axis=1, inplace=True)

    # for numeric columns
    numeric_cols = train_data.columns[4:]

    # min max
    min_values = train_data[numeric_cols].min(axis=1)
    max_values = train_data[numeric_cols].max(axis=1)

    # normalization
    ranges = max_values - min_values
    ranges[ranges == 0] = 1

    # min max scaling
    train_data[numeric_cols] = (train_data[numeric_cols].subtract(min_values, axis=0)).div(ranges, axis=0)

    # max, min to dict
    scale_min_dict = min_values.to_dict()
    scale_max_dict = max_values.to_dict()

    # Label Encoder
    encoder = LabelEncoder()
    categorical_col = ["대분류", "중분류", "소분류", "브랜드"]

    for col in categorical_col:
        train_data[col] = encoder.fit_transform(train_data[col])
    
    train_input, train_target, val_input, val_target = get_train_val_data(train_data, args.window_size, args.predict_size)

    train_dataset = CustomDataset(train_input, train_target)
    val_dataset = CustomDataset(val_input, val_target)

    # 분산학습용
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    # model compile
    model = SalesForecastNet(input_size=args.input_size, hidden_size=args.hidden_size,
                            output_size=args.output_size, num_layers=args.num_layers, use_layernorm=args.use_layernorm).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = WarmupLR(optimizer, warmup_end_steps=1500)
    
    # 학습
    best_model, best_loss = train(model, optimizer, train_dataloader, val_dataloader, rank, scheduler, args)
    logging.info(f"Training complete. Best Validation loss : {best_loss:.5f}")
    
    if rank == 0:
        torch.save(best_model, "best_model.pth")
        logging.info("Model save successfully")

    # start inference
    test_input = get_test_data(train_data, args.window_size)
    test_dataloader = get_test_dataloaders(test_input, args.batch_size)

    pred = inference(best_model, test_dataloader, device)

    # 추론 결과 inverse scaling
    for idx in range(len(pred)):
        pred[idx, :] = pred[idx, :] * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]
    
    # 결과 후처리
    pred = np.round(pred, 0).astype(int)

    submit = pd.read_csv("./data/sample_submission.csv")
    submit.iloc[:, 1:] = pred
    submit.to_csv("final.csv", index=False)
    logging.info("Result save successfully")
    print("저장완료")

# Argument Parsing
if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logging.basicConfig(filename='app.log',   
                        level=logging.INFO,    
                        format='%(asctime)s - %(levelname)s - %(message)s',   
                        datefmt='%d-%b-%y %H:%M:%S') 
    parser = argparse.ArgumentParser(description="Train an LSTM model")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping value")
    parser.add_argument("--window_size", type=int, default=90, help="Window size to prediction")
    parser.add_argument("--predict_size", type=int, default=21, help="predction_size")
    parser.add_argument("--batch_size", type=int, default=2048, help="batch_size for dataloader")
    parser.add_argument("--input_size", type=int, default=5, help="model input size")
    parser.add_argument("--hidden_size", type=int, default=512, help="model hidden size")
    parser.add_argument("--output_size", type=int, default=21, help="model output size")
    parser.add_argument("--num_layers", type=int, default=4, help="model's num_layer")
    parser.add_argument("--use_layernorm", type=bool, default=True, help="use layer norm if True apply layer norm")

    args = parser.parse_args()
    logging.info("Starting the training process")
    logging.info("=== Hyperparameters ===")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Learning Rate: {args.lr}")
    logging.info(f"Clip: {args.clip}")
    logging.info(f"Window Size: {args.window_size}")
    logging.info(f"Predict Size: {args.predict_size}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Input Size: {args.input_size}")
    logging.info(f"Hidden Size: {args.hidden_size}")
    logging.info(f"Output Size: {args.output_size}")
    logging.info(f"Num Layers: {args.num_layers}")
    logging.info(f"Use Layer Norm: {args.use_layernorm}")
    logging.info("=======================")

    # 분산 학습 설정
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)