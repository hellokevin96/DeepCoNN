# -*- coding: utf-8 -*-
import json
import logging
import os
import shutil

import lmdb
import pandas as pd
from util import data_split_pandas

from util import split_sentence_to_word_list, words2ids

logger = logging.getLogger('DeepCoNN.load_data')


def load_data(data_path):
    logger.info('Start reading data to pandas.')
    data_pd = pd.read_json(data_path, lines=True)
    data_pd = data_pd.rename(index=int, columns={'asin': 'item',
                                                 'overall': 'rating',
                                                 'reviewText': 'review_text',
                                                 'reviewerID': 'user',
                                                 'unixReviewTime': 'time'})
    data_pd['review_id'] = data_pd.index

    del data_pd['helpful']
    del data_pd['summary']

    return data_pd


def load_data_deepconn(data_path, train_ratio=.8, test_ratio=.2, rebuild=False):
    """
    数据预处理, 并将review信息存入以lmdb保存在 data_path/../DeepCoNN/lmdb/
    保存至本地文件：数据集基本信息,
    train_user-item-rating-review_id, test_user-item-rating-review_id,
    :param data_path: 源数据路径
    :param train_ratio:
    :param test_ratio:
    :param rebuild:
    :return:
    """

    data_pd = load_data(data_path)
    folder = os.path.dirname(data_path)

    if not rebuild:
        if os.path.exists(os.path.join(folder,
                                       'DeepCoNN',
                                       'train_user_item_rating.csv')):
            return

    user_size = len(data_pd['user'].drop_duplicates())
    item_size = len(data_pd['item'].drop_duplicates())
    dataset_size = len(data_pd)

    dataset_meta_info = {'dataset_size': dataset_size,
                         'user_size': user_size,
                         'item_size': item_size}

    logger.info('convert review sentences into tokens and token ids')
    review = data_pd['review_text']
    review_tokens = review.apply(lambda x: split_sentence_to_word_list(x))
    review_token_ids = review_tokens.apply(lambda x: words2ids(x))
    review = review_tokens.combine(review_token_ids,
                                   lambda x, y: json.dumps({'tokens': x,
                                                            'token_ids': y}))
    review = review.apply(lambda x: x.encode())
    review_memory = review.memory_usage(index=True, deep=True)
    review_memory = int(review_memory * 1.5)
    review = review.to_dict()

    logger.info('Save data to lmdb')
    lmdb_path = '{}/DeepCoNN/lmdb'.format(folder)
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)

    env = lmdb.open(lmdb_path, map_size=review_memory)
    with env.begin(write=True) as txn:
        for k, v in review.items():
            txn.put(str(k).encode(), v)

    user_to_review_ids = data_pd.groupby(['user'])['review_id'] \
        .apply(list).to_dict()
    item_to_review_ids = data_pd.groupby(['item'])['review_id'] \
        .apply(list).to_dict()

    with open('{}/DeepCoNN/user_to_review_ids.json'.format(folder), 'w') as f:
        json.dump(user_to_review_ids, f)
    with open('{}/DeepCoNN/item_to_review_ids.json'.format(folder), 'w') as f:
        json.dump(item_to_review_ids, f)

    data_pd = data_pd.loc[:, ['user', 'item', 'rating']]
    logger.info('Split training and test dataset')
    train_uir, test_uir = data_split_pandas(data_pd, train_ratio, test_ratio)
    train_uir.to_csv('{}/DeepCoNN/train_user_item_rating.csv'.format(folder))
    test_uir.to_csv('{}/DeepCoNN/test_user_item_rating.csv'.format(folder))

    with open('{}/DeepCoNN/dataset_meta_info.json'.format(folder), 'w') as f:
        json.dump(dataset_meta_info, f)

    logger.info('Load data finished')


def load_data_fm(data_path, train_ratio, test_ratio):
    """
    保存以下信息：数据集基本信息, #todo
    :param data_path:
    :param train_ratio:
    :param test_ratio:
    :return:
    """
    # review data_frame, keys: review_text, token_ids
    data_pd = load_data(data_path)

    user_ids = get_unique_id(data_pd, 'user')
    item_ids = get_unique_id(data_pd, 'item')

    user_size = len(user_ids)
    item_size = len(item_ids)
    dataset_size = len(data_pd)
    min_date = data_pd['time'].min().item()
    dataset_meta_info = {'dataset_size': dataset_size,
                         'user_size': user_size,
                         'item_size': item_size,
                         'min_date': min_date}

    logger.info('Split sentence into word ids.')

    # user_item_rating, keys: user, item, rating
    logger.info('Get <user, item, rating> triplet.')
    user_item_rating = data_pd \
        .loc[:, ['user', 'user_id', 'item', 'item_id', 'time', 'rating']]

    # user_to_review_ids, keys: user, review_ids
    # item_to_review_ids, keys: item, review_ids
    logger.info('Get user_to_review_id and item_to_review_id.')
    data_pd['review_ids'] = data_pd.index

    user_to_review_ids = data_pd.groupby(['user'])['review_ids']\
        .apply(list).to_frame()

    item_to_review_ids = data_pd.groupby(['item'])['review_ids']\
        .apply(list).to_frame()

    logger.info('Split data into train, valid and test sets')
    train_user_item_rating, test_user_item_rating = \
        data_split_pandas(user_item_rating, train_ratio, test_ratio)

    logger.info('Data pre-handle finished.')


def dataset_statistic(data_frame: pd.DataFrame, folder: str):
    """
    statistics of data: #user, #item, #review, #word,
    #review per user, #word per user
    :param data_frame: columns: user, item, review_text
    :param folder: fig save folder path
    :return: None
    """
    data_frame['review_word_count'] = \
        data_frame.apply(lambda x: len(x['review_text'].split()), axis=1)
    user_size = data_frame['user'].drop_duplicates().size
    item_size = data_frame['item'].drop_duplicates().size
    review_size = data_frame.size
    word_count = data_frame['review_word_count'].sum()

    attr_list = ('#user', '#item', '#review', '#words',
                 '#review per user', '#words per user')
    rows_format = '{:<10}' * 4 + '{:<20}' * 2
    print(rows_format.format(*attr_list))
    dash = '-'*40
    print(rows_format.format(user_size,
                             item_size,
                             review_size,
                             word_count,
                             word_count/user_size,
                             word_count/review_size))
    # data_frame['review_word_count'].plot.box()
    # plt.savefig(folder + '\\review_word_count_box.jpg')


def get_unique_id(data_pd: pd.DataFrame, column: str) -> (dict, pd.DataFrame):
    """
    获取指定列的唯一id
    :param data_pd: pd.DataFrame 数据
    :param column: 指定列
    :return: dict: {value: id}
    """
    new_column = '{}_id'.format(column)
    assert new_column not in data_pd.columns
    temp = data_pd.loc[:, [column]].drop_duplicates().reset_index(drop=True)
    temp[new_column] = temp.index
    temp.index = temp[column]
    del temp[column]
    # data_pd.merge()
    data_pd = pd.merge(left=data_pd,
                       right=temp,
                       left_on=column,
                       right_index=True,
                       how='left')

    return temp[new_column].to_dict(), data_pd

