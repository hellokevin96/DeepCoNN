# -*- coding: utf-8 -*-
import json
import pickle
from datetime import datetime
import os
from torch.utils.data import TensorDataset, DataLoader
from skorch import NeuralNetRegressor
import torch.nn as nn
import torch
import pandas as pd
import numpy as np


class FactorizationMachine(nn.Module):

    def __init__(self, sample_size: int, factor_size: int, is_cuda=False):
        super(FactorizationMachine, self).__init__()
        self.__linear = nn.Linear(sample_size, 1)
        # self.__v = torch.randn((sample_size, factor_size), requires_grad=True)
        self.__v = torch.normal(0, .001, (sample_size, factor_size),
                                requires_grad=True)
        if is_cuda:
            self.__v = self.__v.cuda()
        self.__drop = nn.Dropout(0.2)

    def forward(self, x):
        # linear regression
        w = self.__linear(x)

        # cross feature
        inter1 = torch.matmul(x, self.__v).sum(1, keepdim=True)
        inter2 = torch.matmul(x.pow(2), self.__v.pow(2)).sum(1, keepdim=True)
        inter = (inter1 - inter2) * 0.5

        return w + inter


def fm_data(data_path: str):
    # get user and item id
    # data_folder = 'data/music_instruments/fm/'
    data_folder = 'data/movie/fm/'
    train_data = pd.read_json(
        os.path.join(data_folder, 'train_user_item_rating.json'))
    # train_data = train_data.loc[:, ['user_id', 'item_id', 'rating']]
    valid_data = pd.read_json(
        os.path.join(data_folder, 'valid_user_item_rating.json'))
    # valid_data = valid_data.loc[:, ['user_id', 'item_id', 'rating']]
    test_data = pd.read_json(
        os.path.join(data_folder, 'test_user_item_rating.json'))
    # test_data = test_data.loc[:, ['user_id', 'item_id', 'rating']]
    train_data_size = len(train_data) + len(valid_data)

    data = pd.concat((train_data, valid_data, test_data)).reset_index(drop=True)

    # min_date = data['time'].min().item()
    # min_date = datetime.fromtimestamp(min_date)

    data['rated_item'] = data.apply(
        lambda r: (r['time'], r['item_id']), axis='columns')
    user_group = data.groupby('user_id')['rated_item'].apply(list)
    user_group = user_group.apply(lambda r: sorted(r, key=lambda l: l[0]))

    del data['rated_item']
    data = pd.merge(left=data,
                    left_on='user_id',
                    right=user_group,
                    right_index=True,
                    how='inner')
    data['rated_item'] = data.apply(
        lambda r:
            list(filter(lambda l: l[0] > r['time'],
                        r['rated_item'])),
        axis='columns')

    data['rated_item'] = data.apply(lambda r: [l[1] for l in r['rated_item']],
                                    axis='columns')

    # data = data.loc[:, ['user_id', 'item_id', 'rated_item', 'time', 'rating']]

    # train_data = pd.merge(left=train_data,
    #                       left_on=('user_id', 'item_id'),
    #                       right=data,
    #                       right_on=('user_id', 'item_id'),
    #                       how='left')
    #
    # valid_data = pd.merge(left=valid_data,
    #                       left_on=('user_id', 'item_id'),
    #                       right=data,
    #                       right_on=('user_id', 'item_id'),
    #                       how='left')
    #
    # test_data = pd.merge(left=test_data,
    #                      left_on=('user_id', 'item_id'),
    #                      right=data,
    #                      right_on=('user_id', 'item_id'),
    #                      how='left')

    with open(os.path.join(data_folder, 'dataset_meta_info.json'), 'r') as f:
        dataset_meta_info = json.load(f)
    user_size = dataset_meta_info['user_size']
    item_size = dataset_meta_info['item_size']
    # return user_size, item_size, train_data, min_date
    min_date = datetime.fromtimestamp(dataset_meta_info['mindate'])

    x_length = user_size+3*item_size+1

    data = data.reset_index(drop=True)
    data['index'] = data.index
    tensor_x = np.zeros((len(data), x_length))

    # user_id, item_id one-hot
    x = data.loc[:, ['user_id', 'item_id', 'index']].to_numpy()
    # x = torch.LongTensor(x)
    x[:, 1] += user_size
    x[:, 0] = x_length * x[:, 2] + x[:, 0]
    x[:, 1] = x_length * x[:, 2] + x[:, 1]
    x = np.delete(x, 2, 1).flatten()
    tensor_x.put(x, 1)

    # other movie rated
    rated_movie = data \
        .apply(lambda r: [(r['index'],
                           l + user_size + item_size)
                          for l in r['rated_item']],
               axis='columns').to_list()
    rated_movie_weight = [[1 / len(r)] * len(r) if len(r) > 0 else []
                          for r in rated_movie]
    rated_movie = sum(rated_movie, [])
    rated_movie = [x[0] * x_length + x[1] for x in rated_movie]
    rated_movie_weight = sum(rated_movie_weight, [])
    tensor_x.put(rated_movie, rated_movie_weight)

    # time(month)
    time = data['time'].apply(lambda r: datetime.fromtimestamp(r))
    time -= min_date
    time = time.apply(lambda r: r.days // 30).to_numpy()
    time_index = data['index'].to_numpy() * x_length \
                 + user_size + 2 * item_size
    tensor_x.put(time_index, time)

    # last rated movie
    last_item = data.apply(
        lambda r: (r['index'], r['rated_item'][-1:]),
        axis='columns')
    last_item = last_item.apply(
        lambda r: [r[0] * x_length + user_size + 2 * item_size + 1 + r[1][0]]
        if len(r[1]) > 0 else []).to_list()

    last_item = sum(last_item, [])
    tensor_x.put(last_item, 1)

    # tensor_x = torch.from_numpy(tensor_x).float()
    # y = torch.from_numpy(data['rating']
    #                      .to_numpy()).float().unsqueeze(1)
    tensor_x = tensor_x.astype(np.float32)
    y = data['rating'].to_numpy()
    y = np.expand_dims(y, axis=1).astype(np.float32)

    train_x = tensor_x[:train_data_size]
    train_y = y[:train_data_size]

    test_x = tensor_x[train_data_size:]
    test_y = y[train_data_size:]
    # print(tensor_x.shape)
    # print(y.shape)
    net_fm = NeuralNetRegressor(FactorizationMachine,
                                max_epochs=100,
                                lr=0.001,
                                optimizer=torch.optim.Adam,
                                module__sample_size=x_length,
                                module__factor_size=100,
                                # device='cuda',
                                module__is_cuda=False)
    net_fm.fit(train_x, train_y)
    test_pred = net_fm.predict(test_x).flatten()
    test_y = test_y.flatten()
    test_mse = np.sqrt((test_pred - test_y)**2).mean()
    print(test_mse)
    with open(os.path.join(data_folder, 'fm.pkl'), 'wb') as f:
        pickle.dump(net_fm, f)

    # score = net_fm.score(test_x, test_y)
    # print(score)


if __name__ == '__main__':
    # fm_train(*fm_data('s'))
    fm_data('s')
