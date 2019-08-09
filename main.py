# -*- coding: utf-8 -*-
# import pickle
import util.log_util
import logging
from load_data.amazon_product_review_load import load_data
from model import DeepCoNNTrainTest

if __name__ == '__main__':
    # data_path = 'data/music/Digital_Music_5.json'
    # user_item_rating, review_pd, user_to_review_ids, item_to_review_ids \
    #     = load_data(data_path)
    # user_item_rating.to_json('data/music/user_item_rating.json')
    # review_pd.to_json('data/music/review.json')
    # user_to_review_ids.to_json('data/music/user_to_review_ids.json')
    # item_to_review_ids.to_json('data/music/item_to_review_ids.json')
    # train(epoch=100, batch_size=512)
    train_test = DeepCoNNTrainTest(epoch=1,
                                   batch_size=12,
                                   review_length=1000,
                                   data_folder='music',
                                   is_cuda=False)

    # train_test.train()
    train_test.test()

