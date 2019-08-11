# -*- coding: utf-8 -*-
import os

import pandas as pd
import torch.nn as nn
import torch.nn
from torch.utils.data import DataLoader, Dataset
import logging
from util.npl_util import gensim_model

logger = logging.getLogger('DeepCoNN.train_test')


def data_split(data: pd.DataFrame,
               train_ratio=.8, valid_ratio=.1, test_ratio=.1):
    assert (train_ratio + valid_ratio + test_ratio) == 1., 'ratio sum != 1'
    data = data.sample(frac=1).reset_index(drop=True)  # shuffle
    train_index = int(len(data) * train_ratio)
    valid_index = int(len(data) * (train_ratio + valid_ratio))

    train_data = data.loc[:train_index, :].reset_index(drop=True)
    valid_data = data.loc[train_index: valid_index, :].reset_index(drop=True)
    test_data = data.loc[valid_index:, :].reset_index(drop=True)

    return train_data, valid_data, test_data


class Flatten(nn.Module):
    """
    squeeze layer for Sequential structure
    """

    def forward(self, x):
        return x.squeeze()


class DataFrameDataSet(Dataset):

    def __init__(self, data: pd.DataFrame):
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, item):
        return self.__data.loc[item].to_list()
        pass


class FactorizationMachine(nn.Module):

    def __init__(self, factor_size: int, is_cuda=False):
        super(FactorizationMachine, self).__init__()
        self.__linear = nn.Linear(factor_size, 1)
        self.__v = torch.randn((factor_size, factor_size), requires_grad=True)
        if is_cuda:
            self.__v = self.__v.cuda()
        self.__drop = nn.Dropout()

    def forward(self, x):
        # linear regression
        w = self.__linear(x).squeeze()

        # cross feature
        inter1 = torch.matmul(x, self.__v)
        inter2 = torch.matmul(x**2, self.__v**2)
        inter = (inter1**2 - inter2) * 0.5
        inter = self.__drop(inter)
        inter = torch.sum(inter, dim=1)

        return w + inter


class DeepCoNN(nn.Module):

    def __init__(self, review_length, word_vec_dim, conv_length,
                 conv_kernel_num, latent_factor_num, is_cuda=False):
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
                                                     is_cuda)

    def forward(self, user_review, item_review):
        user_latent = self.__user_conv(user_review)
        item_latent = self.__item_conv(item_review)

        # concatenate
        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        # print(concat_latent.is_cuda)
        prediction = self.__factor_machine(concat_latent)

        return prediction

    def evaluate(self, user_review, item_review, rating):
        # todo
        pass


def get_review_average_length(df: pd.DataFrame, review_column: str):
    df['sentences_length'] = df[review_column].apply(lambda x: len(x))
    return df['sentences_length'].mean()


class DeepCoNNTrainTest:

    def __init__(self, epoch, batch_size, review_length, data_folder,
                 is_cuda=False):
        """
        训练，测试DeepCoNN
        :param epoch:
        :param batch_size:
        :param review_length:
        :param data_folder:
        :param is_cuda:
        """
        self.epoch = epoch
        self.batch_size = batch_size
        self.review_length = review_length
        self.is_cuda = is_cuda
        self.data_folder = data_folder

        logger.info('epoch:{:<8d} batch size:{:d}'.format(epoch, batch_size))

        # read data
        self.user_item_rating = pd.read_json(
            '{}/user_item_rating.json'.format(data_folder))
        self.review = pd.read_json(
            '{}/review.json'.format(data_folder))
        self.user_to_review_ids = pd.read_json(
            '{}/user_to_review_ids.json'.format(data_folder))
        self.item_to_review_ids = pd.read_json(
            '{}/item_to_review_ids.json'.format(data_folder))

        if os.path.exists(
                os.path.join(data_folder, 'train_user_item_rating.json')):

            train_data = pd.read_json(
                '{}/train_user_item_rating.json'.format(data_folder))
            self.valid_data = pd.read_json(
                '{}/valid_user_item_rating.json'.format(data_folder))
            self.test_data = pd.read_json(
                '{}/test_user_item_rating.json'.format(data_folder))
        else:
            train_data, self.valid_data, self.test_data \
                = data_split(self.user_item_rating)

            train_data.to_json(
                '{}/train_user_item_rating.json'.format(data_folder))
            self.valid_data.to_json(
                '{}/valid_user_item_rating.json'.format(data_folder))
            self.test_data.to_json(
                '{}/test_user_item_rating.json'.format(data_folder))

        logger.info('average sentences length: {:d}'.format(review_length))

        # initial DeepCoNN model
        self.model = DeepCoNN(review_length=review_length,
                              word_vec_dim=300,
                              conv_length=3,
                              conv_kernel_num=100,
                              latent_factor_num=100,
                              is_cuda=self.is_cuda)

        if self.is_cuda:
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
        self.loss_func = torch.nn.MSELoss()

        # load pretrained embedding
        logger.info('Initialize word embedding model for pytorch')
        embedding = torch.FloatTensor(gensim_model.vectors)
        zero_tensor = torch.zeros(size=embedding[:1].size())
        self.zero_index = len(embedding)
        embedding = torch.cat((embedding, zero_tensor), dim=0)
        self.embedding = nn.Embedding.from_pretrained(embedding)
        if self.is_cuda:
            self.embedding = self.embedding.cuda()

        logger.info('Model initialized, start training...')

        # dataloader
        logger.info('Initialize dataloader.')
        self.data_loader = DataLoader(DataFrameDataSet(train_data),
                                      batch_size=batch_size,
                                      shuffle=True)

    def train(self):
        for e in range(self.epoch):
            i = 0
            for data in self.data_loader:

                batch_data = pd.DataFrame(data={'user': data[0],
                                                'item': data[1],
                                                'rating': data[2]})

                batch_user_review_vectors, batch_item_review_vectors = \
                    self.uir_to_token_vectors(batch_data)

                rating = batch_data['rating'].to_list()
                rating = torch.Tensor(rating)
                if self.is_cuda:
                    rating = rating.cuda()

                pred = self.model(batch_user_review_vectors,
                                  batch_item_review_vectors)

                loss = self.loss_func(pred, rating)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                i += 1
                if i % 50 == 0:
                    logger.info(
                        'mse_loss: {:.5f}'.format(loss.cpu().data.numpy()))

            # validate
            valid_user_review_vectors, valid_item_review_vectors = \
                self.uir_to_token_vectors(self.valid_data
                                          .tail(200))  # memory limit

            valid_rating = self.valid_data['rating'].tail(200).to_list()
            valid_rating = torch.Tensor(valid_rating)
            if self.is_cuda:
                valid_rating = valid_rating.cuda()

            valid_pred = self.model(valid_user_review_vectors,
                                    valid_item_review_vectors)

            valid_loss = self.loss_func(valid_pred, valid_rating)
            logger.info(
                'valid mse_loss: {:.5f}'.format(valid_loss.cpu().data.numpy()))

        torch.save(self.model.cpu(),
                   os.path.join(self.data_folder, 'DeepCoNN.pkl'))

    def test(self):
        self.model = torch.load(os.path.join(self.data_folder, 'DeepCoNN.pkl'))
        if self.is_cuda:
            self.model.cuda()

        batch_size = 100
        test_pred = torch.Tensor()
        if self.is_cuda:
            test_pred = test_pred.cuda()
        for i in range(0, len(self.test_data), batch_size):
            test_user_review_vectors, test_item_review_vectors = \
                self.uir_to_token_vectors(self.test_data
                                          .loc[i: i+batch_size-1, :])

            batch_pred = self.model(test_user_review_vectors,
                                    test_item_review_vectors)

            test_pred = torch.cat((test_pred, batch_pred))

        test_rating = self.test_data['rating'].to_list()
        test_rating = torch.Tensor(test_rating)
        if self.is_cuda:
            test_rating = test_rating.cuda()

        print('actual rating:')
        print(test_rating.cpu()[:10])
        print('predicting rating:')
        print(test_pred.cpu()[:10])

        print(test_pred.shape, test_rating.shape)
        test_loss = self.loss_func(test_pred, test_rating)
        logger.info(
            'test mse_loss: {:.5f}'.format(test_loss.cpu().data.numpy()))

        self.test_data['predict_rating'] = test_pred.tolist()
        self.test_data.to_csv(os.path.join(self.data_folder, 'test_result.csv'))

        self.test_data['square_test'] \
            = (self.test_data['rating'] - self.test_data['predict_rating']) ** 2

        mse_on_rating = self.test_data.groupby('rating')['square_error'].mean()
        mse_on_rating.to_csv(os.path.join(self.data_folder,
                                          'test_mse_of_ratings'))

    def uir_to_token_vectors(self, uir: pd.DataFrame):
        """
        根据 <user,item,rating> triplets, 构造对应的评论词向量
        :param uir:
        :return:
        """
        batch_user_data = pd.merge(left=uir,
                                   right=self.user_to_review_ids,
                                   left_on='user',
                                   right_index=True,
                                   how='left')

        batch_item_data = pd.merge(left=uir,
                                   right=self.item_to_review_ids,
                                   left_on='item',
                                   right_index=True,
                                   how='left')

        batch_user_data['token_ids'] = batch_user_data['review_ids'] \
            .apply(lambda x: self.review_ids_to_token_ids(x))

        batch_item_data['token_ids'] = batch_item_data['review_ids'] \
            .apply(lambda x: self.review_ids_to_token_ids(x))

        batch_user_review = torch.LongTensor(
            batch_user_data['token_ids'].to_list())
        batch_item_review = torch.LongTensor(
            batch_item_data['token_ids'].to_list())

        if self.is_cuda:
            batch_item_review = batch_item_review.cuda()
            batch_user_review = batch_user_review.cuda()

        batch_user_review_vec = self.embedding(batch_user_review)
        batch_item_review_vec = self.embedding(batch_item_review)

        batch_user_review_vec = batch_user_review_vec.unsqueeze(dim=1)
        batch_item_review_vec = batch_item_review_vec.unsqueeze(dim=1)

        return batch_user_review_vec, batch_item_review_vec

    def review_ids_to_token_ids(self, review_ids: list):
        _r = self.review[self.review.index.isin(review_ids)]
        _r = _r.sample(frac=1.)['token_ids'].to_list()
        # result = sum(result, [])
        result = []
        for sen in _r:
            if len(result) < self.review_length:
                result += sen
            else:
                break

        if len(result) > self.review_length:
            result = result[:self.review_length]
        elif len(result) < self.review_length:
            result = result + \
                     ([self.zero_index] * (self.review_length-len(result)))

        return result
