# -*- coding: utf-8 -*-

import logging
from load_data.amazon_product_review_load import load_data
from model import DeepCoNNTrainTest
import os

if __name__ == '__main__':
    data_path = 'data/music_instruments/Musical_Instruments_5.json'
    folder = os.path.dirname(data_path)

    if not os.path.exists(os.path.join(folder, 'user_item_rating.json')):
        user_item_rating, review_pd, user_to_review_ids, item_to_review_ids \
            = load_data(data_path)
        user_item_rating.to_json('{:}/user_item_rating.json'.format(folder))
        review_pd.to_json('{:}/review.json'.format(folder))
        user_to_review_ids.to_json('{:}/user_to_review_ids.json'.format(folder))
        item_to_review_ids.to_json('{:}/item_to_review_ids.json'.format(folder))

    train_test = DeepCoNNTrainTest(epoch=1,
                                   batch_size=100,
                                   review_length=100,
                                   data_folder=folder,
                                   is_cuda=False)

    # train_test.train()
    train_test.test()
