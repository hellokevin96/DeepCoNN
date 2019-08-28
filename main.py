# -*- coding: utf-8 -*-
import logging
from load_data import amazon_deepconn_load
from model import DeepCoNNTrainTest
import os
import yaml


def deepconn():
    with open('args.yml', 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    model_args = args['model']
    training_args = args['training']
    data_handle_args = args['data_handle']

    data_path = args['dataset_path']
    dir_path = os.path.dirname(data_path)

    save_folder = args['save_folder']

    # 预处理数据
    amazon_deepconn_load(data_path,
                         train_ratio=data_handle_args['train_ratio'],
                         test_ratio=data_handle_args['test_ratio'],
                         rebuild=data_handle_args['rebuild'])

    # 训练模型
    train_test = DeepCoNNTrainTest(epoch=training_args['epoch'],
                                   batch_size=training_args['batch_size'],
                                   dir_path=dir_path,
                                   device=training_args['device'],
                                   model_args=model_args,
                                   learning_rate=training_args['learning_rate'],
                                   save_folder=save_folder)

    train_test.train()

    with open(os.path.join(dir_path,
                           'DeepCoNN',
                           save_folder,
                           'args.yml'), 'w') as f:
        yaml.dump(args, f)

    train_test.test()


if __name__ == '__main__':
    deepconn()
