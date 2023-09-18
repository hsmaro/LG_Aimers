import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm



def make_train_data(data, train_size, predict_size):
    '''
    학습 기간 블럭, 예측 기간 블럭의 세트로 데이터를 생성
    data : 일별 판매량
    train_size : 학습에 활용할 기간
    predict_size : 추론할 기간
    '''
    num_rows = len(data)
    window_size = train_size + predict_size

    input_data = np.empty((num_rows * (len(data.columns) - window_size + 1), train_size, len(data.iloc[0, :4]) + 1))
    target_data = np.empty((num_rows * (len(data.columns) - window_size + 1), predict_size))

    for i in tqdm(range(num_rows)):
        encode_info = np.array(data.iloc[i, :4])
        sales_data = np.array(data.iloc[i, 4:])

        for j in range(len(sales_data) - window_size + 1):
            window = sales_data[j : j + window_size]
            temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window[:train_size]))
            input_data[i * (len(data.columns) - window_size + 1) + j] = temp_data
            target_data[i * (len(data.columns) - window_size + 1) + j] = window[train_size:]

    return input_data, target_data


def make_predict_data(data, train_size):
    '''
    평가 데이터(Test Dataset)를 추론하기 위한 Input 데이터를 생성
    data : 일별 판매량
    train_size : 추론을 위해 필요한 일별 판매량 기간 (= 학습에 활용할 기간)
    '''
    num_rows = len(data)

    input_data = np.empty((num_rows, train_size, len(data.iloc[0, :4]) + 1))

    for i in tqdm(range(num_rows)):
        encode_info = np.array(data.iloc[i, :4])
        sales_data = np.array(data.iloc[i, -train_size:])

        window = sales_data[-train_size : ]
        temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window[:train_size]))
        input_data[i] = temp_data

    return input_data


def get_train_val_data(train, window_size, predict_size):

    train_input, train_target = make_train_data(train, window_size, predict_size)

    # Train / Validation Split
    # 시계열 데이터이기에 순서에 맞춰서 잘라준다.
    data_len = len(train_input)
    val_input = train_input[-int(data_len*0.2):]
    val_target = train_target[-int(data_len*0.2):]
    train_input = train_input[:-int(data_len*0.2)]
    train_target = train_target[:-int(data_len*0.2)]

    return train_input, train_target, val_input, val_target

def get_test_data(train, window_size):
    test_input = make_predict_data(train, window_size)

    return test_input

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        if self.Y is not None:
            return torch.Tensor(self.X[index]), torch.Tensor(self.Y[index])
        return torch.Tensor(self.X[index])

    def __len__(self):
        return len(self.X)
    
def get_dataloaders(train_input, train_target, val_input, val_target, batch_size):
    train_dataset = CustomDataset(train_input, train_target)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset = CustomDataset(val_input, val_target)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader

def get_test_dataloaders(test_input, batch_size):

    test_dataset = CustomDataset(test_input, None)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return test_dataloader