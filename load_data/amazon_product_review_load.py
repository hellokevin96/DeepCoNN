# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from util import split_sentence_to_word_list, words2ids
import logging
import os

logger = logging.getLogger('DeepCoNN.load_data')


def load_data(data_path):
    data_pd = pd.read_json(data_path, lines=True)
    data_pd = data_pd.rename(index=int, columns={'asin': 'item',
                                                 'overall': 'rating',
                                                 'reviewText': 'review_text',
                                                 'reviewerID': 'user'})

    del data_pd['helpful']
    del data_pd['summary']
    del data_pd['unixReviewTime']

    logger.info('Compile data statistic')
    dataset_statistic(data_pd, os.path.dirname(data_path))

    # review data_frame, keys: review_text, token_ids
    logger.info("Get review data frame.")
    review_pd = data_pd['review_text'].to_frame()

    logger.info('Split sentence into word ids.')

    review_pd['tokens'] = review_pd \
        .apply(lambda x: split_sentence_to_word_list(x['review_text']), axis=1)
    review_pd['token_ids'] = review_pd\
        .apply(lambda x: words2ids(x['tokens']), axis=1)

    # user_item_rating, keys: user, item, rating
    logger.info('Get <user, item, rating> triplet.')
    user_item_rating = data_pd.loc[:, ['user', 'item', 'rating']]

    # user_to_review_ids, keys: user, review_ids
    # item_to_review_ids, keys: item, review_ids
    logger.info('Get user_to_review_id and item_to_review_id.')
    data_pd['review_ids'] = data_pd.index

    user_to_review_ids = data_pd.groupby(['user'])['review_ids']\
        .apply(list).to_frame()

    item_to_review_ids = data_pd.groupby(['item'])['review_ids'] \
        .apply(list).to_frame()

    logger.info('Data pre-handle finished.')
    return user_item_rating, review_pd, user_to_review_ids, item_to_review_ids


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

    # dash = '-'*40
    attr_list = ('#user', '#item', '#review', '#words',
                 '#review per user', '#words per user')
    rows_format = '{:<10}' * 4 + '{:<20}' * 2
    print(rows_format.format(*attr_list))
    print(rows_format.format(user_size,
                             item_size,
                             review_size,
                             word_count,
                             word_count/user_size,
                             word_count/review_size))
    data_frame['review_word_count'].plot.box()
    plt.savefig(folder + '\\review_word_count_box.jpg')

