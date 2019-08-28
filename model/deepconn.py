# -*- coding: utf-8 -*-
import copy
import json
import os
import random
import lmdb
import pandas as pd
import torch.nn as nn
import torch.nn
from torch.utils.data import DataLoader, Dataset
import logging
from util import data_split_pandas
from util.npl_util import gensim_model

logger = logging.getLogger('DeepCoNN.train_test')


class FactorizationMachine(nn.Module):

    # noinspection PyArgumentList
    def __init__(self, factor_size: int, fm_k: int):
        super(FactorizationMachine, self).__init__()
        self._linear = nn.Linear(factor_size, 1)
        self._v = torch.nn.Parameter(torch.randn((factor_size, fm_k)))
        self._drop = nn.Dropout(0.2)

    def forward(self, x):
        # linear regression
        w = self._linear(x).squeeze()

        # cross feature
        inter1 = torch.matmul(x, self._v)
        inter2 = torch.matmul(x**2, self._v**2)
        inter = (inter1**2 - inter2) * 0.5
        inter = self._drop(inter)
        inter = torch.sum(inter, dim=1)

        return w + inter


class Flatten(nn.Module):
    """
    squeeze layer for Sequential structure
    """

    def forward(self, x):
        return x.squeeze()


class DeepCoNNDataSet(Dataset):

    def __init__(self, data: pd.DataFrame, folder, zero_index: int,
                 review_length: int, device: torch.device):
        self._data = data.reset_index(drop=True)

        self._lmdb = lmdb.open(os.path.join(folder, 'lmdb'), readonly=True)

        with open(os.path.join(folder, 'user_to_review_ids.json'), 'r') as f:
            self._user_to_review_ids = json.load(f)

        with open(os.path.join(folder, 'item_to_review_ids.json'), 'r') as f:
            self._item_to_review_ids = json.load(f)

        self._zero_index = zero_index
        self._review_length = review_length
        self._device = device

    def __len__(self):
        return len(self._data)

    # noinspection PyArgumentList
    def __getitem__(self, x):
        """
        :param x:
        :return: review token ids of users and items with fixed length
        """
        uir = self._data.loc[x].to_dict()
        user = uir['user']
        item = uir['item']
        rating = uir['rating']

        user_review_ids = self._user_to_review_ids[user]
        item_review_ids = self._item_to_review_ids[item]

        with self._lmdb.begin() as txn:
            user_review_tokens = [txn.get(str(i).encode())
                                  for i in user_review_ids]
            item_review_tokens = [txn.get(str(i).encode())
                                  for i in item_review_ids]

        user_review_tokens = [json.loads(str(v, 'utf-8'))['token_ids']
                              for v in user_review_tokens]
        item_review_tokens = [json.loads(str(v, 'utf-8'))['token_ids']
                              for v in item_review_tokens]

        user_tokens = self.review_ids_to_token_ids(user_review_tokens)
        item_tokens = self.review_ids_to_token_ids(item_review_tokens)

        user_tokens = torch.LongTensor(user_tokens).to(self._device)
        item_tokens = torch.LongTensor(item_tokens).to(self._device)
        rating = torch.FloatTensor([rating]).to(self._device)

        return user_tokens, item_tokens, rating

    def close_lmdb(self):
        self._lmdb.close()

    def review_ids_to_token_ids(self, review_list: list):
        # _r = self.review[self.review.index.isin(review_ids)]
        random.shuffle(review_list)
        # result = sum(result, [])
        result = []
        for sen in review_list:
            if len(result) < self._review_length:
                result += sen
            else:
                break

        if len(result) > self._review_length:
            result = result[:self._review_length]
        elif len(result) < self._review_length:
            result = result + \
                     ([self._zero_index] * (self._review_length-len(result)))

        return result


class DataPreFetcher:

    def __init__(self, loader):
        self._loader = iter(loader)
        self._rating = None
        self._item_review = None
        self._user_review = None
        self.pre_load()

    def pre_load(self):
        try:
            self._user_review, self._item_review, self._rating \
                = next(self._loader)
        except StopIteration:
            self._rating = None
            self._item_review = None
            self._user_review = None
            return

    def next(self):
        # data = self._next_data
        user_review = self._user_review
        item_review = self._item_review
        rating = self._rating
        self.pre_load()
        return user_review, item_review, rating


class DeepCoNN(nn.Module):

    def __init__(self, review_length, word_vec_dim, fm_k, conv_length,
                 conv_kernel_num, latent_factor_num):
        """
        :param review_length: 评论单词数
        :param word_vec_dim: 词向量维度
        :param conv_length: 卷积核的长度
        :param conv_kernel_num: 卷积核数量
        :param latent_factor_num: 全连接输出的特征维度
        """
        super(DeepCoNN, self).__init__()

        # input_shape: (batch_size, 1, review_length, word_vec_dim)
        self.__user_conv = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=conv_kernel_num,
                kernel_size=(conv_length, word_vec_dim),
            ),
            # (batch_size, conv_kernel_num, review_length-conv_word_size + 1, 1)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(review_length-conv_length+1, 1)),
            Flatten(),
            nn.Linear(conv_kernel_num, latent_factor_num),
            nn.ReLU(),
        )

        self.__item_conv = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=conv_kernel_num,
                kernel_size=(conv_length, word_vec_dim),
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(review_length-conv_length+1, 1)),
            Flatten(),
            nn.Linear(conv_kernel_num, latent_factor_num),
            nn.ReLU(),
        )

        # input: (batch_size, 2*latent_factor_num)
        self.__factor_machine = FactorizationMachine(latent_factor_num * 2,
                                                     fm_k)

    def forward(self, user_review, item_review):
        user_latent = self.__user_conv(user_review)
        item_latent = self.__item_conv(item_review)

        # concatenate
        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        # print(concat_latent.is_cuda)
        prediction = self.__factor_machine(concat_latent)

        return prediction


def get_review_average_length(df: pd.DataFrame, review_column: str):
    df['sentences_length'] = df[review_column].apply(lambda x: len(x))
    return df['sentences_length'].mean()


# noinspection PyUnreachableCode,PyArgumentList
class DeepCoNNTrainTest:

    # noinspection PyUnresolvedReferences
    def __init__(self, epoch, batch_size, dir_path, device, model_args,
                 learning_rate, save_folder):
        """
        训练，测试DeepCoNN
        """
        self._epoch = epoch
        self._batch_size = batch_size
        self._review_length = model_args['review_length']
        self._dir_path = os.path.join(dir_path, 'DeepCoNN')
        self._device = torch.device(device)
        self._save_dir = os.path.join(dir_path, 'DeepCoNN', save_folder)

        logger.info('epoch:{:<8d} batch size:{:d}'.format(epoch, batch_size))

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

        # read data
        self._train_data = pd.read_csv(
            os.path.join(self._dir_path,  'train_user_item_rating.csv'))
        self._test_data = pd.read_csv(
            os.path.join(self._dir_path, 'test_user_item_rating.csv'))

        with open(os.path.join(self._dir_path, 'dataset_meta_info.json'),
                  'r') as f:
            dataset_meta_info = json.load(f)
        self._user_size = dataset_meta_info['user_size']
        self._item_size = dataset_meta_info['item_size']
        self._dataset_size = dataset_meta_info['dataset_size']

        # initial DeepCoNN model
        self._model = DeepCoNN(review_length=model_args['review_length'],
                               word_vec_dim=model_args['word_vector_dim'],
                               fm_k=model_args['fm_k'],
                               conv_length=model_args['conv_length'],
                               conv_kernel_num=model_args['conv_kernel_num'],
                               latent_factor_num=model_args['latent_factor_num']
                               ).to(self._device)

        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=learning_rate)
        self._loss_func = torch.nn.MSELoss()

        # load pretrained embedding
        logger.info('Initialize word embedding model for pytorch')
        embedding = torch.FloatTensor(gensim_model.vectors)
        zero_tensor = torch.zeros(size=embedding[:1].size())
        self._zero_index = embedding.size()[0]
        embedding = torch.cat((embedding, zero_tensor), dim=0)
        self._embedding \
            = nn.Embedding.from_pretrained(embedding).to(self._device)

        logger.info('Model initialized, start training...')

        # dataloader
        logger.info('Initialize dataloader.')
        data = pd.read_csv('{}/train_user_item_rating.csv'
                           .format(self._dir_path))
        test_data = pd.read_csv('{}/test_user_item_rating.csv'
                                .format(self._dir_path))

        train_data, valid_data = data_split_pandas(data, 0.9, 0.1)

        train_dataset = DeepCoNNDataSet(data=train_data,
                                        folder=self._dir_path,
                                        zero_index=self._zero_index,
                                        review_length=self._review_length,
                                        device=self._device)
        self._valid_dataset = DeepCoNNDataSet(data=valid_data,
                                              folder=self._dir_path,
                                              zero_index=self._zero_index,
                                              review_length=self._review_length,
                                              device=self._device)

        self._test_dataset = DeepCoNNDataSet(data=test_data,
                                             folder=self._dir_path,
                                             zero_index=self._zero_index,
                                             review_length=self._review_length,
                                             device=self._device)

        self._data_loader = DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)

    def train(self):
        valid_data_loader = DataLoader(self._valid_dataset,
                                       batch_size=self._batch_size,
                                       shuffle=False)

        logger.info('Start training.')
        best_valid_loss = float('inf')
        best_model_state_dict = None
        best_valid_epoch = 0
        for e in range(self._epoch):
            fetcher = DataPreFetcher(self._data_loader)
            user_tokens, item_tokens, rating = fetcher.next()

            train_loss = None
            while user_tokens is not None:
                user_review_vec = self._embedding(user_tokens).unsqueeze(dim=1)
                item_review_vec = self._embedding(item_tokens).unsqueeze(dim=1)

                pred = self._model(user_review_vec,
                                   item_review_vec)

                train_loss = self._loss_func(pred, rating.flatten())
                self._optimizer.zero_grad()
                train_loss.backward()
                self._optimizer.step()

                user_tokens, item_tokens, rating = fetcher.next()

            # validate
            fetcher = DataPreFetcher(valid_data_loader)
            valid_user_tokens, valid_item_tokens, valid_rating = fetcher.next()
            # pred = torch.FloatTensor()
            # rating = torch.FloatTensor()
            error = torch.FloatTensor().to(self._device)
            while valid_user_tokens is not None:
                user_review_vec = \
                    self._embedding(valid_user_tokens).unsqueeze(dim=1)
                item_review_vec = \
                    self._embedding(valid_item_tokens).unsqueeze(dim=1)

                batch_pred = self._model(user_review_vec,
                                         item_review_vec)

                batch_error = (batch_pred - valid_rating.flatten())
                error = torch.cat((error, batch_error))

                valid_user_tokens, valid_item_tokens, valid_rating \
                    = fetcher.next()

            error = torch.mean(error**2).item()  # mse
            if best_valid_loss > error:
                best_model_state_dict = copy.deepcopy(self._model.state_dict())
                best_valid_loss = error
                best_valid_epoch = e
            logger.info(
                'epoch: {}, train mse_loss: {:.5f}, valid mse_loss: {:.5f}'
                .format(e, train_loss, error))

        # save
        torch.save(best_model_state_dict,
                   os.path.join(self._save_dir,
                                'DeepCoNN.tar'))

        with open(os.path.join(self._save_dir, 'training.json'), 'w') as f:
            json.dump({'epoch': best_valid_epoch,
                       'valid_loss': best_valid_loss},
                      f)

    def test(self):
        self._model.load_state_dict(torch.load(os.path.join(self._save_dir,
                                                            'DeepCoNN.tar')))
        data_loader = DataLoader(self._test_dataset,
                                 batch_size=self._batch_size,
                                 shuffle=False)

        error = torch.FloatTensor([]).to(self._device)
        pred = torch.FloatTensor([]).to(self._device)
        for ur, ir, r in data_loader:
            user_review_vec = \
                self._embedding(ur).unsqueeze(dim=1)
            item_review_vec = \
                self._embedding(ir).unsqueeze(dim=1)

            batch_pred = self._model(user_review_vec,
                                     item_review_vec)

            pred = torch.cat((pred, batch_pred))

            batch_error = (batch_pred - r.flatten())
            error = torch.cat((error, batch_error))
        error = torch.mean(error**2).item()
        logger.info('Test MSE: {:.5f}'.format(error))
        with open(os.path.join(self._save_dir, 'test_result.json'), 'w') as f:
            json.dump({'mse': error}, f)
        self._test_data['predict'] = pred.tolist()
        self._test_data.to_csv(os.path.join(self._save_dir,
                                            'test_result_detail.csv'))



